from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import transformers
import json
from PIL import Image, ImageDraw
import wandb
import time
import argparse
import re
from typing import List
from qwen2_vl.model import Qwen2VLChat
from qwen_vl_utils import process_vision_info
torch.manual_seed(1234)

def lines_demarcation(image: Image.Image) -> List[Image.Image]:
    """Return a list containing the input image augmented with green lines
    demarcating the vertical and horizontal midlines.

    A vertical green line is drawn through the center column and a horizontal
    green line through the center row.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("lines_demarcation expects a PIL.Image.Image")

    width, height = image.size
    center_x, center_y = width // 2, height // 2

    augmented = image.copy()
    draw = ImageDraw.Draw(augmented)

    green = (0, 255, 0)
    line_width = max(1, min(width, height) // 400 or 1)

    # Vertical center line
    draw.line([(center_x, 0), (center_x, height)], fill=green, width=line_width)
    # Horizontal center line
    draw.line([(0, center_y), (width, center_y)], fill=green, width=line_width)

    return [augmented]


def quadrants_demarcation(image: Image.Image) -> List[Image.Image]:
    """Return the four quadrants of the image in graph-quadrant order.

    Order:
    1) Top-right, 2) Top-left, 3) Bottom-left, 4) Bottom-right
    """
    if not isinstance(image, Image.Image):
        raise TypeError("quadrants_demarcation expects a PIL.Image.Image")

    width, height = image.size
    center_x, center_y = width // 2, height // 2

    # Define crop boxes: (left, upper, right, lower)
    top_right = image.crop((center_x, 0, width, center_y))
    top_left = image.crop((0, 0, center_x, center_y))
    bottom_left = image.crop((0, center_y, center_x, height))
    bottom_right = image.crop((center_x, center_y, width, height))

    return [top_right, top_left, bottom_left, bottom_right]

DEMARCATION_MAP = {
    "lines": lines_demarcation,
    "quadrants": quadrants_demarcation,
    "none": lambda x: [x]
}

COORDINATE_PROMPT = 'In this UI screenshot, what is the position of the element corresponding to the command \"{command}\" (with point)?'

PARTITION_PROMPT = 'In this UI screenshot, what is the partition of the element corresponding to the command \"{command}\" (with quadrant number)?'

MODEL_DIRECTORY = "{model_path}"
MODEL = "base_experiment_out"
K = 3
CONTEXT = True
RESOLUTION = "dynamic"
DEMARCATION = "none"

TEST_DATA_PATH = "/ocean/projects/cis240092p/amartin1/ReVL/data/datasets/benchmarks/screenspot/{data_name}.json"
TEST_DATA = "screenspot_web"

IMAGE_PATH = "/ocean/projects/cis240092p/amartin1/ReVL/data/images/benchmarks/screenspot/images/{image_name}"

def generate_with_history(model, query, current_images, history=None, history_images=None):
    """
    Custom wrapper to support conversation history with Qwen2VLChat.
    
    Args:
        model: Qwen2VLChat model instance
        query: Text query string
        current_images: List of PIL Image objects (or single PIL Image) for current query
        history: List of conversation history in format [{'from': 'human'/'gpt', 'value': str}, ...]
        history_images: List of PIL Image objects corresponding to each 'human' turn in history
    
    Returns:
        Generated response string
    """
    # Build messages list
    messages = []
    if model.system_prompt is not None:
        messages.append({'role': 'system', 'content': model.system_prompt})
    
    # Add history if provided
    if history and history_images:
        image_idx = 0
        for msg in history:
            if msg['from'] == 'human':
                # Parse out the query text (remove <image> token)
                text_content = msg['value'].replace('<image>\n', '').replace('<image>', '')
                # Add the corresponding image from history
                if image_idx < len(history_images):
                    # Add images to content if history_images entry is a list or single item
                    img_entry = history_images[image_idx]
                    content = []
                    if isinstance(img_entry, list):
                        for img in img_entry:
                             content.append({'type': 'image', 'image': img})
                    else:
                        content.append({'type': 'image', 'image': img_entry})
                    
                    content.append({'type': 'text', 'text': text_content})
                    image_idx += 1
                else:
                    content = text_content
                messages.append({'role': 'user', 'content': content})
            elif msg['from'] == 'gpt':
                messages.append({'role': 'assistant', 'content': msg['value']})
    elif history:
        # Fallback if no history_images provided
        for msg in history:
            if msg['from'] == 'human':
                text_content = msg['value'].replace('<image>\n', '').replace('<image>', '')
                messages.append({'role': 'user', 'content': text_content})
            elif msg['from'] == 'gpt':
                messages.append({'role': 'assistant', 'content': msg['value']})
    
    # Add current query with image(s)
    content = []
    if isinstance(current_images, list):
        for img in current_images:
            content.append({'type': 'image', 'image': img})
    else:
         content.append({'type': 'image', 'image': current_images})
    
    content.append({'type': 'text', 'text': query})
    messages.append({'role': 'user', 'content': content})
    
    if model.verbose:
        print(f'\033[31m{messages}\033[0m')
    
    print(f"messages: {messages}")
    
    # Process messages through the model
    text = model.processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info([messages])
    inputs = model.processor(text=text, images=images, videos=videos, padding=True, return_tensors='pt')
    inputs = inputs.to('cuda')
    
    generated_ids = model.model.generate(
        **inputs,
        **model.generate_kwargs,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    out = model.processor.tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    response = out[0]
    
    if model.verbose:
        print(f'\033[32m{response}\033[0m')
    
    return response

def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark"
    )

    parser.add_argument(
        "--data",
        type=str,
        default=TEST_DATA,
        help="The dataset to evaluate on",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=MODEL,
        help="The model to evaluate",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=K,
        help="The number of partitions to use",
    )

    parser.add_argument(
        "--context",
        type=bool,
        default=CONTEXT,
        help="Whether to keep context images",
    )

    parser.add_argument(
        "--resolution",
        type=str,
        default=RESOLUTION,
        choices=["dynamic", "static"],
        help="Image resolution strategy: 'dynamic' or 'static' (mirrors finetuning behavior)",
    )

    parser.add_argument(
        "--demarcation",
        type=str,
        default=DEMARCATION,
        choices=list(DEMARCATION_MAP.keys()),
        help="Demarcation strategy: 'none', 'lines', or 'quadrants'",
    )

    return parser.parse_args()

def eval(model, tokenizer, test_data, visualize=False, k=0, keep_context=False, demarcation="none"):
    acc = 0
    imgs = []
    total_time = 0
    demarcation_fn = DEMARCATION_MAP[demarcation]

    for i, data in enumerate(test_data):
        
        print(f"Test {i+1}/{len(test_data)}")
        image_path = IMAGE_PATH.format(image_name=data['img_filename'])

        command = data['instruction']
        bbox = data['bbox']
        print(image_path)
        print(command)
        print(bbox)

        # Load original image as PIL object
        original_image = Image.open(image_path)
        original_width, original_height = original_image.size
        
        # Current working image (PIL object)
        current_image = original_image.copy()
        
        # Normalize dataset bbox from [x, y, w, h] pixels to [x0, y0, x1, y1] normalized
        x0_px, y0_px, w_px, h_px = bbox
        x1_px = x0_px + w_px
        y1_px = y0_px + h_px
        x0 = x0_px / original_width
        y0 = y0_px / original_height
        x1 = x1_px / original_width
        y1 = y1_px / original_height
        # Clamp to [0,1] and ensure ordering
        x0 = max(0.0, min(1.0, x0))
        x1 = max(0.0, min(1.0, x1))
        y0 = max(0.0, min(1.0, y0))
        y1 = max(0.0, min(1.0, y1))
        bbox = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
        
        partition_imgs = []
        partitions = []
        time_start = time.time()
        history = []
        history_images = []  # Track PIL images for each turn
        try:
            for j in range(k):
                query = PARTITION_PROMPT.format(command=command)
                print("Query:", query)
                
                # Apply demarcation for this step
                demarcated_images = demarcation_fn(current_image)

                if keep_context:
                    # Generate with history
                    response = generate_with_history(model, query, demarcated_images, history=history, history_images=history_images)
                    # Add human message to history
                    # Prepend <image> for each image
                    image_tokens = "".join(["<image>\n" for _ in demarcated_images])
                    history.append({'from': 'human', 'value': f'{image_tokens}{query}'})
                    # Add current image(s) to history_images
                    history_images.append([img.copy() for img in demarcated_images])
                    
                    # Add gpt response to history
                    history.append({'from': 'gpt', 'value': response})
                else:
                    # Generate without history using PIL image
                    response = generate_with_history(model, query, demarcated_images, history=None, history_images=None)
                
                print("Partition Response:", response)

                # parse partition from response
                try:
                    partition = int(response.split(" ")[-1])
                except ValueError:
                    print(f"Failed to parse partition from response: {response}")
                    partition = 0 # Default or handle error
                
                partitions.append(partition)
                
                # Update current_image for next step (for Lines, we accumulate lines)
                if demarcation == "lines":
                    # lines_demarcation returns 1 image
                    current_image = demarcated_images[0]

                # Get cropped image of the partition from the current_image
                width, height = current_image.size
                if partition == 1:
                    current_image = current_image.crop((width // 2, 0, width, height // 2))
                elif partition == 2:
                    current_image = current_image.crop((0, 0, width // 2, height // 2))
                elif partition == 3:
                    current_image = current_image.crop((0, height // 2, width // 2, height))
                elif partition == 4:
                    current_image = current_image.crop((width // 2, height // 2, width, height))

                partition_imgs.append(wandb.Image(current_image, caption=f"{command}: {j + 1}, {partition}"))

            query = COORDINATE_PROMPT.format(command=command)
            print("Query:", query)
            
            # For final coordinate step, apply demarcation
            demarcated_images = demarcation_fn(current_image)

            if keep_context:
                # Generate final coordinate response with history
                response = generate_with_history(model, query, demarcated_images, history=history, history_images=history_images)
                # Add final human message to history
                image_tokens = "".join(["<image>\n" for _ in demarcated_images])
                history.append({'from': 'human', 'value': f'{image_tokens}{query}'})
                # Add final image to history_images
                history_images.append([img.copy() for img in demarcated_images])
                # Add final gpt response to history (for completeness)
                history.append({'from': 'gpt', 'value': response})
            else:
                # Generate without history using PIL image
                response = generate_with_history(model, query, demarcated_images, history=None, history_images=None)
            
            print("Coordinate Response:", response)
        
            # parse point from response (point has format (x, y))

            x = float(response.split(",")[0].split("(")[1])
            y = float(response.split(",")[1].split(")")[0])
            print("point:", x, y)
            time_end = time.time()
            total_time += time_end - time_start

            for partition in partitions[::-1]:
                if partition == 1:
                    x = x/2 + 0.5
                    y = y/2
                elif partition == 2:
                    x = x/2
                    y = y/2
                elif partition == 3:
                    x = x/2
                    y = y/2 + 0.5
                elif partition == 4:
                    x = x/2 + 0.5
                    y = y/2 + 0.5
            print("rescaled point:", x, y)
            
            # check if point is inside bbox
            if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
                acc += 1
                print("Correct")
            else:
                print("Incorrect")
            print()
            if visualize:
                vis_img = original_image.copy()
                draw = ImageDraw.Draw(vis_img)
                draw.rectangle([bbox[0] * original_width, bbox[1] * original_height, bbox[2] * original_width, bbox[3] * original_height], outline="red")

                point = (x*original_width, y*original_height)
                draw.rectangle([point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5], fill="red")

                imgs.append(wandb.Image(vis_img, caption=command))
                wandb.log({"images": imgs + partition_imgs})
                imgs = []
            
            # Clean up - close the original image
            original_image.close()
        except Exception as e:
            print("Error trying to run inference:", e)
            # print traceback
            import traceback
            traceback.print_exc()
            print()
            continue
        wandb.log({"step": i+1, "num_correct": acc, "time": time_end - time_start})

    wandb.log({"average_inference_time": total_time / len(test_data)})
    return acc / len(test_data)


if __name__ == '__main__':
    args = config()
    MODEL = args.model
    TEST_DATA = args.data
    K = args.k
    CONTEXT = args.context
    RESOLUTION = args.resolution
    DEMARCATION = args.demarcation

    # Mirror finetuning resolution behavior:
    # dynamic  -> max_pixels=50176, min_pixels=784
    # static   -> 448x448 fixed resolution
    if RESOLUTION.lower() == "dynamic":
        max_pixels = 50176
        min_pixels = 784
    elif RESOLUTION.lower() == "static":
        px = 448 * 448
        max_pixels = px
        min_pixels = px
    else:
        raise ValueError(f"Invalid resolution '{RESOLUTION}'. Use 'dynamic' or 'static'.")
    
    wandb.init(
        project="ReVL_eval",
        name=MODEL
        + "-"
        + TEST_DATA
        + "-k"
        + str(K)
        + ("-context" if CONTEXT else "")
        + f"-res-{RESOLUTION}"
        + f"-demarc-{DEMARCATION}",
    )


    print(f"Loading HuggingFace model from {MODEL_DIRECTORY.format(model_path=MODEL)}")
    
    model = Qwen2VLChat(
        model_path=MODEL_DIRECTORY.format(model_path=MODEL),
        temperature=0.01,
        top_p=0.001,
        top_k=1,
        use_custom_prompt=True,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    with open(TEST_DATA_PATH.format (data_name=TEST_DATA), 'r') as f:
        test_data = json.load(f)
    acc = eval(model, None, test_data, visualize=True, k = K, keep_context=CONTEXT, demarcation=DEMARCATION)
    wandb.log({"accuracy": acc})
    print(f"Accuracy: {acc}")
