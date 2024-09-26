from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import transformers
from peft import AutoPeftModelForCausalLM
import json
from PIL import Image, ImageDraw
import wandb
import time
torch.manual_seed(1234)

COORDINATE_PROMPT = 'In this UI screenshot, what is the position of the element corresponding to the command \"{command}\" (with point)?'

PARTITION_PROMPT = 'In this UI screenshot, what is the partition of the element corresponding to the command \"{command}\" (with quadrant number)?'

MODEL_DIRECTORY = "./finetune/output/{model_name}"
MODEL = "baseline_model"
K = 0
CONTEXT = False

TEST_DATA_PATH = "./data/json_data/{data_name}.json"
TEST_DATA = "screenspot_bbox_test_text"

IMAGE_PATH = "./data/images/{image_name}"

TEMP_PATH = "./eval/temp/partition{num}.png"


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

        # parse top, left, bottom, right from bbox string
        # bbox = [float(x) for x in bbox.replace("(", "").replace(")", "").split(",")]
        print(bbox)

        with Image.open(image) as img:
            original_image = img.copy()
            original_width, original_height = img.size
        
        partition_imgs = []
        partitions = []
        images = [image]
        time_start = time.time()
        try:
            for j in range(k):
                query = tokenizer.from_list_format(([{ 'image': context_image } for context_image in images] if keep_context else [{'image': image}]) + 
                [{'text': PARTITION_PROMPT.format(command=command)}])
                print("Query:", query)
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
                    img.save(TEMP_PATH.format(num=j))
                    image = TEMP_PATH.format(num=j)
                    images.append(image)

                    partition_imgs.append(wandb.Image(img, caption=f"{command}: {j + 1}, {partition}"))

            query = tokenizer.from_list_format(([{ 'image': context_image } for context_image in images] if keep_context else [{'image': image}]) + 
            [{'text': COORDINATE_PROMPT.format(command=command)}])
            print("Query:", query)
            response, _ = model.chat(tokenizer, query=query, history=None)
            print("Coordinate Response:", response)
        
            # parse point from response (point has format (x, y))

            x = float(response.split(",")[0].split("(")[1])
            y = float(response.split(",")[1].split(")")[0])
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
    # init wandb
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

    with open(TEST_DATA_PATH.format(data_name=TEST_DATA), 'r') as f:
        test_data = json.load(f)

    acc = eval(model, tokenizer, test_data, visualize=True, k = K, keep_context=CONTEXT)
    wandb.log({"accuracy": acc})
    print(f"Accuracy: {acc}")
    
