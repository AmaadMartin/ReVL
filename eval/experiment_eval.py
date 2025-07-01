import argparse
import json
import os
import time
import torch
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
from PIL import Image, ImageDraw
import wandb

# === Global parameters / path templates ===
MODEL_DIRECTORY = "./finetune/{model_name}"
# When loading images from test_data, use:
IMAGE_PATH = "./data/images/{image_name}"
# For temporary saving of cropped images:
TEMP_PATH = "./eval/temp2/partition{num}.png"  # used in experiments 1 & 2
# (For experiment 3, we will write temp files with a different naming scheme.)

# === Helper Functions ===

def add_quadrant_marks(image_path, new_image_path):
    """Draw quadrant lines on an image and save it to new_image_path."""
    with Image.open(image_path) as img:
        width, height = img.size
        draw = ImageDraw.Draw(img)
        # Draw two thick green lines at the midpoints
        draw.line((width // 2, 0, width // 2, height), fill=(0, 255, 0), width=3)
        draw.line((0, height // 2, width, height // 2), fill=(0, 255, 0), width=3)
        img.save(new_image_path)
        return img

def split_into_quadrants(image_path, step):
    """
    Given an image (path) and the current step (used in naming),
    split it into four quadrants using the same cropping as in the training/inference code.
    Returns a dictionary mapping quadrant number (1-4) to the temporary file path.
    Mapping (as in our eval script):
      - Quadrant 1: top–right: (width//2, 0, width, height//2)
      - Quadrant 2: top–left: (0, 0, width//2, height//2)
      - Quadrant 3: bottom–left: (0, height//2, width//2, height)
      - Quadrant 4: bottom–right: (width//2, height//2, width, height)
    """
    quadrants = {}
    with Image.open(image_path) as im:
        width, height = im.size
        crop_coords = {
            1: (width // 2, 0, width, height // 2),
            2: (0, 0, width // 2, height // 2),
            3: (0, height // 2, width // 2, height),
            4: (width // 2, height // 2, width, height)
        }
        for q in [1, 2, 3, 4]:
            cropped = im.crop(crop_coords[q])
            temp_path = f"./eval/temp3/exp3_step{step}_quad{q}.png"
            cropped.save(temp_path)
            quadrants[q] = temp_path
    return quadrants

def get_gt_quadrants(target, k):
    """
    Given the ground–truth target point (normalized coordinates in [0,1]×[0,1]) and number of steps k,
    compute the sequence of quadrant decisions that would be made by a “perfect” splitter.
    We assume the following rule:
       if x < 0.5 and y < 0.5: quadrant 2
       if x >= 0.5 and y < 0.5: quadrant 1
       if x < 0.5 and y >= 0.5: quadrant 3
       if x >= 0.5 and y >= 0.5: quadrant 4
    Then update the target to be relative to the chosen quadrant.
    Returns:
       gt_quadrants: a list of length k with the ground–truth quadrant (one per step)
       final_target: the final normalized target (in the cropped image after k steps)
    """
    gt_quadrants = []
    current = target  # current normalized target in the current image (starts as full image coordinates)
    for _ in range(k):
        x, y = current
        if x < 0.5 and y < 0.5:
            gt = 2
            new = (x / 0.5, y / 0.5)
        elif x >= 0.5 and y < 0.5:
            gt = 1
            new = ((x - 0.5) / 0.5, y / 0.5)
        elif x < 0.5 and y >= 0.5:
            gt = 3
            new = (x / 0.5, (y - 0.5) / 0.5)
        else:  # x >= 0.5 and y >= 0.5
            gt = 4
            new = ((x - 0.5) / 0.5, (y - 0.5) / 0.5)
        gt_quadrants.append(gt)
        current = new
    return gt_quadrants, current

def reverse_transform(final_coord, partitions):
    """
    Given the predicted final coordinate (normalized in the final cropped image)
    and the list of predicted partition decisions (one per step),
    reverse–apply the cropping transformation (in reverse order)
    to get the coordinate in the original full–sized image.
    (This reverse mapping follows the cropping logic used during inference.)
    """
    x, y = final_coord
    for p in reversed(partitions):
        if p == 1:
            x = x / 2 + 0.5
            y = y / 2
        elif p == 2:
            x = x / 2
            y = y / 2
        elif p == 3:
            x = x / 2
            y = y / 2 + 0.5
        elif p == 4:
            x = x / 2 + 0.5
            y = y / 2 + 0.5
    return x, y

# === The Main Evaluation Function ===

def eval_model(model, tokenizer, test_data, exp_mode, k=3, context=True, visualize=False):
    """
    Evaluate the model on test_data.
    
    Parameters:
      - exp_mode: 1 or 2 use a single image (with/without quadrant marks), while 3 uses separate quadrant images.
      - k: number of partition (quadrant) steps.
      - context: if True, the full dialogue (all previous turns) is included in each query.
      - visualize: if True, log visualizations via wandb.
    
    Returns:
      quadrant_accuracy: fraction of all partition decisions that match the ground truth.
      final_accuracy: fraction of samples for which the final predicted coordinate is inside the bbox.
    """
    total_quadrant_steps = 0
    correct_quadrant_steps = 0
    total_samples = 0
    final_correct = 0
    total_time = 0
    vis_images = []

    for i, data in enumerate(test_data):
        print(f"\n--- Sample {i+1}/{len(test_data)} ---")
        # Each test sample is assumed to have keys:
        #   'img_filename': the image file name (to be formatted by IMAGE_PATH)
        #   'instruction': the referring command text
        #   'bbox': a list [x_min, y_min, x_max, y_max] (normalized)
        image_path = IMAGE_PATH.format(image_name=data['img_filename'])
        command = data['instruction']
        bbox = data['bbox']
        # For quadrant evaluation, we use the center of the bbox as the target point.
        target_point = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        gt_quads, gt_final = get_gt_quadrants(target_point, k)
        print("Ground–truth quadrant sequence:", gt_quads)
        
        # For partitioning, we start with the full image.
        current_image = image_path
        predicted_partitions = []
        sample_start = time.time()
        
        # Initialize conversation history if context is enabled.
        history = [] if context else None
        
        for j in range(k):
            # Build the prompt for the partition step.
            if exp_mode in [1, 2]:
                # For experiments 1 and 2, we supply a single image.
                if exp_mode == 2:
                    # add quadrant marks if desired
                    temp_marked = TEMP_PATH.format(num=0)
                    add_quadrant_marks(current_image, temp_marked)
                    image_for_prompt = temp_marked
                else:
                    image_for_prompt = current_image
                user_prompt = f"<img>{image_for_prompt}</img>\nIn this UI screenshot, what is the partition of the element corresponding to the command \"{command}\" (with quadrant number)?"
            elif exp_mode == 3:
                # For experiment 3, split the current image into four quadrant images and include all in the prompt.
                quads = split_into_quadrants(current_image, j)
                prompt_lines = []
                for q in [1, 2, 3, 4]:
                    prompt_lines.append(f"Quadrant {q}: <img>{quads[q]}</img>")
                prompt_lines.append(f"In this UI screenshot, what is the partition of the element corresponding to the command \"{command}\" (with quadrant number)?")
                user_prompt = "\n".join(prompt_lines)
            else:
                raise ValueError("Invalid experiment mode.")
            
            print(f"Step {j+1} partition prompt:\n{user_prompt}")
            # Send the prompt. If context is enabled, include the conversation history.
            if context:
                response, history = model.chat(tokenizer, query=user_prompt, history=history)
            else:
                response, _ = model.chat(tokenizer, query=user_prompt, history=None)
            print("Raw partition response:", response)
            try:
                # Attempt to extract the first token as an integer (the predicted quadrant)
                pred_quad = int(response.strip().split()[0])
            except Exception as e:
                print("Error parsing partition response:", e)
                pred_quad = -1  # indicate an error
            predicted_partitions.append(pred_quad)
            total_quadrant_steps += 1
            if pred_quad == gt_quads[j]:
                correct_quadrant_steps += 1
                print("Partition step correct.")
            else:
                print(f"Partition step incorrect: predicted {pred_quad} vs. expected {gt_quads[j]}")
            
            # Now update the current image by cropping according to the predicted quadrant.
            try:
                with Image.open(current_image) as img:
                    width, height = img.size
                    if pred_quad == 1:
                        box = (width // 2, 0, width, height // 2)
                    elif pred_quad == 2:
                        box = (0, 0, width // 2, height // 2)
                    elif pred_quad == 3:
                        box = (0, height // 2, width // 2, height)
                    elif pred_quad == 4:
                        box = (width // 2, height // 2, width, height)
                    else:
                        # if there was an error, do not crop
                        box = (0, 0, width, height)
                    cropped = img.crop(box)
                    temp_crop = TEMP_PATH.format(num=j+1)
                    cropped.save(temp_crop)
                    # For exp_mode 2, we may add quadrant marks (except for final step)
                    if exp_mode == 2 and j < k - 1:
                        add_quadrant_marks(temp_crop, temp_crop)
                    current_image = temp_crop
                    vis_images.append(wandb.Image(cropped, caption=f"Step {j+1}: Predicted quadrant {pred_quad}"))
            except Exception as e:
                print("Error cropping image:", e)
        
        # Final coordinate step: ask for the precise point.
        if exp_mode in [1, 2]:
            if exp_mode == 2:
                temp_marked = TEMP_PATH.format(num=0)
                add_quadrant_marks(current_image, temp_marked)
                final_img = temp_marked
            else:
                final_img = current_image
            final_prompt = f"<img>{final_img}</img>\nIn this UI screenshot, what is the position of the element corresponding to the command \"{command}\" (with point)?"
        elif exp_mode == 3:
            final_prompt = f"<img>{current_image}</img>\nIn this UI screenshot, what is the position of the element corresponding to the command \"{command}\" (with point)?"
        print("Final coordinate prompt:\n", final_prompt)
        if context:
            response, history = model.chat(tokenizer, query=final_prompt, history=history)
        else:
            response, _ = model.chat(tokenizer, query=final_prompt, history=None)
        print("Raw coordinate response:", response)
        try:
            # Expecting a coordinate in the form (x, y)
            coord_str = response.strip()
            x = float(coord_str.split(",")[0].split("(")[-1])
            y = float(coord_str.split(",")[1].split(")")[0])
            pred_coord = (x, y)
        except Exception as e:
            print("Error parsing coordinate response:", e)
            pred_coord = (0.0, 0.0)
        # Reverse the transformations (apply the predicted partitions in reverse)
        full_pred_coord = reverse_transform(pred_coord, predicted_partitions)
        print("Rescaled final coordinate:", full_pred_coord)
        
        # Check if the final coordinate lies within the bbox.
        if bbox[0] <= full_pred_coord[0] <= bbox[2] and bbox[1] <= full_pred_coord[1] <= bbox[3]:
            final_correct += 1
            print("Final coordinate is CORRECT.")
        else:
            print("Final coordinate is INCORRECT.")
        
        sample_time = time.time() - sample_start
        total_time += sample_time
        total_samples += 1
        
        wandb.log({
            "sample": i+1,
            "sample_time": sample_time,
            "final_correct_so_far": final_correct,
        })
        
        # (Optional) Visualize the original image with the bbox and final point.
        if visualize:
            with Image.open(IMAGE_PATH.format(image_name=data['img_filename'])) as orig_img:
                draw = ImageDraw.Draw(orig_img)
                w, h = orig_img.size
                draw.rectangle([bbox[0]*w, bbox[1]*h, bbox[2]*w, bbox[3]*h], outline="red", width=2)
                pt = (full_pred_coord[0]*w, full_pred_coord[1]*h)
                draw.ellipse([pt[0]-5, pt[1]-5, pt[0]+5, pt[1]+5], fill="red")
                vis_images.append(wandb.Image(orig_img, caption=command))
                wandb.log({"visualizations": vis_images})
                vis_images = []  # reset for next sample

    avg_inference_time = total_time / total_samples if total_samples else 0
    quadrant_accuracy = correct_quadrant_steps / total_quadrant_steps if total_quadrant_steps else 0
    final_accuracy = final_correct / total_samples if total_samples else 0
    wandb.log({
        "avg_inference_time": avg_inference_time,
        "quadrant_accuracy": quadrant_accuracy,
        "final_accuracy": final_accuracy,
    })
    return quadrant_accuracy, final_accuracy

# === Main Script ===

def main():
    parser = argparse.ArgumentParser(description="Evaluate GUI grounding model for experiments 1, 2, and 3")
    parser.add_argument("--data", type=str, default="./data/json_data/{data_name}.json",
                        help="Path pattern to test data JSON file (use {data_name} placeholder)")
    parser.add_argument("--data_name", type=str, default="screenspot_bbox_test",
                        help="Name of test data file (without extension)")
    parser.add_argument("--model", type=str, default="base_experiment_out",
                        help="Name of the model directory (inside finetune/)")
    parser.add_argument("--k", type=int, default=3, help="Number of partition (quadrant) steps")
    parser.add_argument("--context", type=bool, default=True, help="Whether to include the full dialogue history in each turn")
    parser.add_argument("--exp_mode", type=int, choices=[1, 2, 3], default=1,
                        help="Experiment mode: 1 = original conversation, 2 = with quadrant marks, 3 = separate quadrant images")
    parser.add_argument("--visualize", action="store_true", help="Log visualizations with wandb")
    args = parser.parse_args()

    wandb.init(project="ReVL_eval", name=f"{args.model}-{args.data_name}-k{args.k}-exp{args.exp_mode}" + ("-context" if args.context else ""))
    
    model = AutoPeftModelForCausalLM.from_pretrained(
        MODEL_DIRECTORY.format(model_name=args.model),
        device_map="cuda",
        trust_remote_code=True,
        fp16=True
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIRECTORY.format(model_name=args.model),
        model_max_length=2048,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    tokenizer.pad_token_id = tokenizer.eod_id

    test_data_path = "./data/json_data/{data_name}.json".format(data_name=args.data_name)
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    # Optionally limit test_data (for quick debugging)
    test_data = test_data
    
    quad_acc, final_acc = eval_model(model, tokenizer, test_data, args.exp_mode, k=args.k, context=args.context, visualize=True)
    print(f"\nOverall quadrant accuracy: {quad_acc:.2f}")
    print(f"Overall final coordinate accuracy: {final_acc:.2f}")
    wandb.log({
        "quadrant_accuracy": quad_acc,
        "final_accuracy": final_acc
    })

if __name__ == "__main__":
    main()
