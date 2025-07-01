from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import transformers
from peft import AutoPeftModelForCausalLM
import json
from PIL import Image, ImageDraw
import wandb
import time
import argparse
torch.manual_seed(1234)

COORDINATE_PROMPT = 'In this UI screenshot, what is the position of the element corresponding to the command \"{command}\" (with point)?'

PARTITION_PROMPT = 'In this UI screenshot, what is the partition of the element corresponding to the command \"{command}\" (with quadrant number)?'

MODEL_DIRECTORY = "./finetune/output/{model_name}"
MODEL = "base_experiment_out"
K = 3
CONTEXT = True

TEST_DATA_PATH = "./data/json_data/{data_name}.json"
TEST_DATA = "screenspot_bbox_test"

IMAGE_PATH = "./data/images/{image_name}"

# TEMP_PATH = "./eval/temp/partition{num}.png"
TEMP_PATH = "./eval/temp/partition{num}.png"

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

    return parser.parse_args()

def eval(model, tokenizer, test_data, visualize=False, k=0, keep_context=False):
    acc = 0
    imgs = []
    total_time = 0
    for i, data in enumerate(test_data):
        
        print(f"Test {i+1}/{len(test_data)}")
        image = IMAGE_PATH.format(image_name=data['img_filename'])

        command = data['instruction']
        bbox = data['bbox']
        print(image)
        print(command)
        print(bbox)

        with Image.open(image) as img:
            original_image = img.copy()
            original_width, original_height = img.size
        
        partition_imgs = []
        partitions = []
        images = [image]
        time_start = time.time()
        history = None
        try:
            for j in range(k):
                query = tokenizer.from_list_format([{'image': image}, {'text': PARTITION_PROMPT.format(command=command)}])
                print("Query:", query)
                if keep_context:
                    response, history = model.chat(tokenizer, query=query, history=history)
                else:
                    response, _ = model.chat(tokenizer, query=query, history=None)
                print("Partition Response:", response)

                # parse partition from response
                partition = int(response.split(" ")[-1])
                partitions.append(partition)
                
                # get cropped image of the partition
                with Image.open(image) as img:
                    width, height = img.size
                    if partition == 1:
                        img = img.crop((width // 2, 0, width, height // 2))
                    elif partition == 2:
                        img = img.crop((0, 0, width // 2, height // 2))
                    elif partition == 3:
                        img = img.crop((0, height // 2, width // 2, height))
                    elif partition == 4:
                        img = img.crop((width // 2, height // 2, width, height))

                    new_image = TEMP_PATH.format(num=j+1)
                    img.save(new_image)
                    image = new_image
                    images.append(image)

                    partition_imgs.append(wandb.Image(img, caption=f"{command}: {j + 1}, {partition}"))

            query = tokenizer.from_list_format([{'image': image}, {'text': COORDINATE_PROMPT.format(command=command)}])
            print("Query:", query)
            if keep_context:
                response, history = model.chat(tokenizer, query=query, history=history)
            else:
                response, _ = model.chat(tokenizer, query=query, history=None)
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
                img = original_image.copy()
                draw = ImageDraw.Draw(img)
                width, height = img.size
                draw.rectangle([bbox[0] * original_width, bbox[1] * original_height, bbox[2] * original_width, bbox[3] * original_height], outline="red")

                point = (x*original_width, y*original_height)
                draw.rectangle([point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5], fill="red")

                imgs.append(wandb.Image(img, caption=command))
                wandb.log({"images": imgs + partition_imgs})
                imgs = []
        except Exception as e:
            print("Error trying to run inference:", e)
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
    
    wandb.init(project="ReVL_eval", name = MODEL + "-" + TEST_DATA + "-k" + str(K) + ("-context" if CONTEXT else ""))

    model = AutoPeftModelForCausalLM.from_pretrained(
    MODEL_DIRECTORY.format(model_name=MODEL),
    device_map="cuda",
    trust_remote_code=True,
    fp16=True).eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_DIRECTORY.format(model_name=MODEL),
        cache_dir=None,
        model_max_length=2048,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    tokenizer.pad_token_id = tokenizer.eod_id

    with open(TEST_DATA_PATH.format (data_name=TEST_DATA), 'r') as f:
        test_data = json.load(f)
    acc = eval(model, tokenizer, test_data, visualize=True, k = K, keep_context=CONTEXT)
    wandb.log({"accuracy": acc})
    print(f"Accuracy: {acc}")
    
