import json
import random
import argparse

WEB_SEECLICK_DATASET_PATH = '/ocean/projects/cis240092p/amartin1/ReVL/data/datasets/web/seeclick_web/seeclick_web.json'
WEB_IMAGES_PATH = '/ocean/projects/cis240092p/amartin1/ReVL/data/images/web/seeclick_web_images/images/cpfs01/user/chengkanzhi/seeclick_web_imgs/'

MOBILE_RICOSCA_DATASET_PATH = '/ocean/projects/cis240092p/amartin1/ReVL/data/datasets/mobile/ricosca/ricosca.json'
MOBILE_SCREEN_CAPTIONING_DATASET_PATH = '/ocean/projects/cis240092p/amartin1/ReVL/data/datasets/mobile/screen_captioning/screen_captioning.json'
MOBILE_WIDGET_CAPTIONING_DATASET_PATH = '/ocean/projects/cis240092p/amartin1/ReVL/data/datasets/mobile/widget_captioning/widget_captioning.json'
MOBILE_IMAGES_PATH = '/ocean/projects/cis240092p/amartin1/ReVL/data/images/mobile/RICO/images/combined/'

GENERAL_LLAVA_DATASET_PATH = '/ocean/projects/cis240092p/amartin1/ReVL/data/datasets/general/llava-instruct-150k/llava-instruct-150k.json'
GENERAL_IMAGES_PATH = '/ocean/projects/cis240092p/amartin1/ReVL/data/images/general/COCO/images/train2014/COCO_train2014_'

ROLE_MAPPING = {
    "assistant": "gpt",
    "gpt": "gpt",
    "user": "human",
    "human": "human"
}

TEXT_TO_QUADRANT_PROMPT = "<image>\nIn this UI screenshot, what is the partition of the element corresponding to the command \"{task}\" (with quadrant number)?"
TEXT_TO_POINT_PROMPT = "<image>\nIn this UI screenshot, what is the position of the element corresponding to the command \"{task}\" (with point)?"
TEXT_TO_BBOX_PROMPT = "<image>\nIn this UI screenshot, what is the position of the element corresponding to the command \"{task}\" (with bbox)?"
POINT_TO_TEXT_PROMPT = "<image>\nIn this UI screenshot, what is the element corresponding to the point \"({x}, {y})\" on the screen?"
BBOX_TO_TEXT_PROMPT = "<image>\nIn this UI screenshot, what is the element corresponding to the bounding box \"({bbox_0}, {bbox_1}, {bbox_2}, {bbox_3})\" on the screen?"
SCREEN_CAPTIONING_PROMPT = "<image>\nCan you provide a detailed description of the interface screenshot shown?"
WIDGET_CAPTIONING_PROMPT = "<image>\nFrom this UI screenshot, can you caption the widget corresponding to the point \"({x}, {y})\" on the screen?"

parser = argparse.ArgumentParser()
parser.add_argument("--num_samples", type=int, default=1000)
parser.add_argument("--num_web_text_to_point_percentage", type=float, default=0.272)
parser.add_argument("--num_web_text_to_bbox_percentage", type=float, default=0.054)
parser.add_argument("--num_point_to_text_percentage", type=float, default=0.054)
parser.add_argument("--num_bbox_to_text_percentage", type=float, default=0.054)
parser.add_argument("--num_mobile_text_to_point_percentage", type=float, default=0.275)
parser.add_argument("--num_mobile_text_to_bbox_percentage", type=float, default=0.056)
parser.add_argument("--num_ui_summarization_percentage", type=float, default=0.048)
parser.add_argument("--num_widget_captioning_percentage", type=float, default=0.042)
parser.add_argument("--num_general_llava_percentage", type=float, default=0.145)
args = parser.parse_args()

assert abs(args.num_web_text_to_point_percentage + args.num_web_text_to_bbox_percentage + args.num_point_to_text_percentage + args.num_bbox_to_text_percentage + args.num_mobile_text_to_point_percentage + args.num_mobile_text_to_bbox_percentage + args.num_ui_summarization_percentage + args.num_widget_captioning_percentage + args.num_general_llava_percentage - 1) < 1e-6, f"The sum of the percentages must be 1, got {args.num_web_text_to_point_percentage + args.num_web_text_to_bbox_percentage + args.num_point_to_text_percentage + args.num_bbox_to_text_percentage + args.num_mobile_text_to_point_percentage + args.num_mobile_text_to_bbox_percentage + args.num_ui_summarization_percentage + args.num_widget_captioning_percentage + args.num_general_llava_percentage}"

def create_conversation_message(role, content):
    return {
        "from": role,
        "value": content
    }

def bbox_to_point(bbox):
    # get the center of the bbox
    x = (bbox[0] + bbox[2]) / 2
    y = (bbox[1] + bbox[3]) / 2
    return (round(x, 2), round(y, 2))

def truncate_bbox(bbox):
    return [round(coord, 2) for coord in bbox]

def bbox_to_text(bbox):
    truncated_bbox = truncate_bbox(bbox)
    return f"({truncated_bbox[0]}, {truncated_bbox[1]}, {truncated_bbox[2]}, {truncated_bbox[3]})"

def point_to_text(x, y):
    return f"({x}, {y})"

def process_web_text_to_point(data, count):
    datapoints = []
    for item in random.sample(data, count):
        image = WEB_IMAGES_PATH + item["img_filename"]
        for element in item.get("elements", []):
            coordinates = bbox_to_point(element["bbox"])
            datapoint = {
                "type": "text_to_point",
                "image": [image],
                "quadrant_template": TEXT_TO_QUADRANT_PROMPT,
                "point_template": TEXT_TO_POINT_PROMPT,
                "prompt": element["instruction"],
                "coordinates": coordinates,
            }
            datapoints.append(datapoint)
    return random.sample(datapoints, count)

def process_mobile_text_to_point(data, count):
    datapoints = []
    for item in random.sample(data, count):
        image = MOBILE_IMAGES_PATH + item["img_filename"]
        coordinates = bbox_to_point(item["bbox"])
        datapoint = {
            "type": "text_to_point",
            "image": [image],
            "quadrant_template": TEXT_TO_QUADRANT_PROMPT,
            "point_template": TEXT_TO_POINT_PROMPT,
            "prompt": item["instruction"],
            "coordinates": coordinates,
        }
        datapoints.append(datapoint)
    return random.sample(datapoints, count)

def process_web_text_to_bbox(data, count):
    datapoints = []
    for item in random.sample(data, count):
        image = WEB_IMAGES_PATH + item["img_filename"]
        for element in item.get("elements", []):
            instruction = TEXT_TO_BBOX_PROMPT.format(task=element["instruction"])
            conversation = [
                create_conversation_message(ROLE_MAPPING["human"], instruction),
                create_conversation_message(ROLE_MAPPING["assistant"], bbox_to_text(element["bbox"])),
            ]
            datapoint = {
                "type": "sft",
                "image": [image],
                "conversations": conversation,
            }
            datapoints.append(datapoint)
    return random.sample(datapoints, count)

def process_mobile_text_to_bbox(data, count):
    datapoints = []
    for item in random.sample(data, count):
        image = MOBILE_IMAGES_PATH + item["img_filename"]
        instruction = TEXT_TO_BBOX_PROMPT.format(task=item["instruction"])
        
        conversation = [
            create_conversation_message(ROLE_MAPPING["human"], instruction),
            create_conversation_message(ROLE_MAPPING["assistant"], bbox_to_text(item["bbox"])),
        ]
        
        datapoint = {
            "type": "sft",
            "image": [image],
            "conversations": conversation,
        }
        datapoints.append(datapoint)
    return random.sample(datapoints, count)

def process_web_point_to_text(data, count):
    datapoints = []
    for item in random.sample(data, count):
        image = WEB_IMAGES_PATH + item["img_filename"]
        for element in item.get("elements", []):
            coordinates = bbox_to_point(element["bbox"])
            instruction = POINT_TO_TEXT_PROMPT.format(x=coordinates[0], y=coordinates[1])
            
            conversation = [
                create_conversation_message(ROLE_MAPPING["human"], instruction),
                create_conversation_message(ROLE_MAPPING["assistant"], element["instruction"]),
            ]
            
            datapoint = {
                "type": "sft",
                "image": [image],
                "conversations": conversation,
            }
            datapoints.append(datapoint)
    return random.sample(datapoints, count)

def process_web_bbox_to_text(data, count):
    datapoints = []
    for item in random.sample(data, count):
        image = WEB_IMAGES_PATH + item["img_filename"]
        for element in item.get("elements", []):
            bbox = truncate_bbox(element["bbox"])
            instruction = BBOX_TO_TEXT_PROMPT.format(bbox_0=bbox[0], bbox_1=bbox[1], bbox_2=bbox[2], bbox_3=bbox[3])
            conversation = [
                create_conversation_message(ROLE_MAPPING["human"], instruction),
                create_conversation_message(ROLE_MAPPING["assistant"], element["instruction"]),
            ]
            
            datapoint = {
                "type": "sft",
                "image": [image],
                "conversations": conversation,
            }
            datapoints.append(datapoint)
    return random.sample(datapoints, count)

def process_mobile_screen_captioning(data, count):
    datapoints = []
    for item in random.sample(data, count):
        image = MOBILE_IMAGES_PATH + item["img_filename"]
        instruction = SCREEN_CAPTIONING_PROMPT
        conversation = [
            create_conversation_message(ROLE_MAPPING["human"], instruction),
            create_conversation_message(ROLE_MAPPING["assistant"], random.choice(item["captions"])),
        ]
        
        datapoint = {
            "type": "sft",
            "image": [image],
            "conversations": conversation,
        }
        datapoints.append(datapoint)
    return random.sample(datapoints, count)

def process_mobile_widget_captioning(data, count):
    datapoints = []
    for item in random.sample(data, count):
        image = MOBILE_IMAGES_PATH + item["img_filename"]
        coordinates = bbox_to_point(item["bbox"])
        instruction = WIDGET_CAPTIONING_PROMPT.format(x=coordinates[0], y=coordinates[1])
        conversation = [
            create_conversation_message(ROLE_MAPPING["human"], instruction),
            create_conversation_message(ROLE_MAPPING["assistant"], item["instruction"]),
        ]
        
        datapoint = {
            "type": "sft",
            "image": [image],
            "conversations": conversation,
        }
        datapoints.append(datapoint)
    return random.sample(datapoints, count)

def process_general_llava(data, count):
    datapoints = []

    for item in random.sample(data, count):
        conversation = []
        for message in item["conversations"]:
            role = ROLE_MAPPING[message["from"]]
            content = message["value"]
            conversation.append(create_conversation_message(role, content))
        image = GENERAL_IMAGES_PATH + item["image"]
        datapoint = {
            "type": "sft",
            "image": [image],
            "conversations": conversation,
        }
        datapoints.append(datapoint)
    return random.sample(datapoints, count)

def create_combined_dataset():
  with open(WEB_SEECLICK_DATASET_PATH, "r") as web_file, open(MOBILE_RICOSCA_DATASET_PATH, "r") as mobile_file, open(MOBILE_SCREEN_CAPTIONING_DATASET_PATH, "r") as screen_file, open(MOBILE_WIDGET_CAPTIONING_DATASET_PATH, "r") as widget_file, open(GENERAL_LLAVA_DATASET_PATH, "r") as llava_file:

        web_data = json.load(web_file)
        mobile_data = json.load(mobile_file)
        screen_captioning_data = json.load(screen_file)
        widget_captioning_data = json.load(widget_file)
        llava_data = json.load(llava_file)

        # randomize the data
        random.shuffle(web_data)
        random.shuffle(mobile_data)
        random.shuffle(screen_captioning_data)
        random.shuffle(widget_captioning_data)
        random.shuffle(llava_data)

        num_samples = args.num_samples
        num_web_text_to_point = int(num_samples * args.num_web_text_to_point_percentage)
        num_web_text_to_bbox = int(num_samples * args.num_web_text_to_bbox_percentage)
        num_point_to_text = int(num_samples * args.num_point_to_text_percentage)
        num_bbox_to_text = int(num_samples * args.num_bbox_to_text_percentage)
        num_mobile_text_to_point = int(num_samples * args.num_mobile_text_to_point_percentage)
        num_mobile_text_to_bbox = int(num_samples * args.num_mobile_text_to_bbox_percentage)
        num_ui_summarization = int(num_samples * args.num_ui_summarization_percentage)
        num_widget_captioning = int(num_samples * args.num_widget_captioning_percentage)
        num_general_llava = int(num_samples * args.num_general_llava_percentage)

        combined_dataset = []
        combined_dataset += process_web_text_to_point(web_data, num_web_text_to_point)
        combined_dataset += process_web_text_to_bbox(web_data, num_web_text_to_bbox)
        combined_dataset += process_web_point_to_text(web_data, num_point_to_text)
        combined_dataset += process_web_bbox_to_text(web_data, num_bbox_to_text)
        combined_dataset += process_mobile_text_to_point(mobile_data, num_mobile_text_to_point)
        combined_dataset += process_mobile_text_to_bbox(mobile_data, num_mobile_text_to_bbox)
        combined_dataset += process_mobile_screen_captioning(screen_captioning_data, num_ui_summarization)
        combined_dataset += process_mobile_widget_captioning(widget_captioning_data, num_widget_captioning)
        combined_dataset += process_general_llava(llava_data, num_general_llava)
        
        random.shuffle(combined_dataset)
        
        file_name = f"/ocean/projects/cis240092p/amartin1/ReVL/data/datasets/ReVL/ReVL_{num_samples}_{num_web_text_to_point}_{num_web_text_to_bbox}_{num_point_to_text}_{num_bbox_to_text}_{num_mobile_text_to_point}_{num_mobile_text_to_bbox}_{num_ui_summarization}_{num_widget_captioning}_{num_general_llava}.json"
        with open(file_name, "w") as output_file:
            json.dump(combined_dataset, output_file, indent=2)

        print(f"Dataset successfully created as {file_name}")

# Call the function to create the dataset
create_combined_dataset()
