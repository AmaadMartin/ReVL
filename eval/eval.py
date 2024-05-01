from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import transformers
from peft import AutoPeftModelForCausalLM
import json
from PIL import Image, ImageDraw
import wandb
torch.manual_seed(1234)

COORDINATE_PROMPT = 'In this UI screenshot, what is the position of the element corresponding to the command \"{command}\" (with point)?'

PARTITION_PROMPT = 'In this UI screenshot, what is the partition of the element corresponding to the command \"{command}\" (with quadrant number)?'

MODEL_DIRECTORY = "./output_qwen/{model_name}"
MODEL = "k_1_model"

def eval(model, tokenizer, test_data, visualize=False, k=0, keep_context=False):
    acc = 0
    imgs = []
    for i, data in enumerate(test_data):
        print(f"Test {i+1}/{len(test_data)}")
        image = data['image']
        command = data['command']
        bbox = data['bbox']
        print(image)
        print(command)

        # parse top, left, bottom, right from bbox string
        bbox = [float(x) for x in bbox.replace("(", "").replace(")", "").split(",")]
        print(bbox)

        with Image.open(image) as img:
            original_image = img.copy()
            original_width, original_height = img.size
        
        partition_imgs = []
        history = None
        partitions = []
        try:
            for j in range(k):
                query = tokenizer.from_list_format([
                    {'image': image},
                    {'text': PARTITION_PROMPT.format(command=command)},
                ])
                response, history = model.chat(tokenizer, query=query, history=history)
                if not keep_context:
                    history = None
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
                    img.save("temp/partition.png")
                    image = "temp/partition.png"

                    partition_imgs.append(wandb.Image(img, caption=f"{command}: {j + 1}, {partition}"))

            query = tokenizer.from_list_format([
                {'image': image},
                {'text': COORDINATE_PROMPT.format(command=command)},
            ])
            response, _ = model.chat(tokenizer, query=query, history=history)
            print("Coordinate Response:", response)
        
            # parse point from response (point has format (x, y))

            x = float(response.split(",")[0].split("(")[1])
            y = float(response.split(",")[1].split(")")[0])

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
        except:
            print("Invalid response")
            print()
            continue
        wandb.log({"step": i+1, "num_correct": acc})

    return acc / len(test_data)


if __name__ == '__main__':
    # init wandb
    wandb.init(project=MODEL + "_eval")

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

    with open("./json_data/screenspot_web_test.json", "r") as f:
        test_data = json.load(f)

    acc = eval(model, tokenizer, test_data, visualize=True, k = 2, keep_context=False)
    print(f"Accuracy: {acc}")
    
