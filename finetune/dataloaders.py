from dataclasses import dataclass, field
import json
import math
import logging
import os
from typing import Dict, Optional, List
import torch
from torch.utils.data import Dataset
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import transformers
from transformers import Trainer, GPTQConfig, deepspeed
from transformers.trainer_pt_utils import LabelSmoother
from torch.utils.data import DataLoader as Dataloader
import random
from PIL import Image, ImageDraw
import pandas as pd
import argparse
import csv
import json
import tempfile

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
# CHECKPOINT = "./finetune/output/checkpoint-2750"
IMAGE_PATH = './data/images'
CACHE_PATH = './finetune/state/cached_data.csv'
K = 3
# WANDB_PROJECT = f"revl_k_{K}"
CONTEXT = True
DATA_SEPERATOR = "`"
# EVAL_DATA_PATH = "./data/json_data/full_coordinate_test.json"
STEPS_LEFT_PATH = "./finetune/state/steps_left.json"

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def atomic_write(filepath, content):
    with tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(filepath)) as temp:
        temp.write(content.encode())  # Assuming content is a string
    os.rename(temp.name, filepath)  # Atomic on POSIX systems

def cache_data(task, id, images, points, quadrants):
    # if task and id already in cache, add the extra points and quadrants to their respective lists
    # csv format: task, id, points, quadrants
    with open(CACHE_PATH, "r+") as f:
        # find the index of the task and id in the csv file
        # replace \n with blank space
        reformatted_task = task.replace("\n", " ")
        reformatted_task = reformatted_task.replace('"', '""')
        reformatted_task = reformatted_task.rstrip("\n")
        reformatted_points = DATA_SEPERATOR.join([str(round(x[0], 2)) + "~" + str(round(x[1], 2)) for x in points])
        reformmateed_quadrants = DATA_SEPERATOR.join([str(quadrant) for quadrant in quadrants])
        reformatted_images = DATA_SEPERATOR.join(images)
        index, _ = check_cache(id)

        if index == -1:
            lines = f.readlines()
            lines.append(f"{id},\"{reformatted_task}\",{reformatted_points},{reformmateed_quadrants},{reformatted_images}\n")
            atomic_write(CACHE_PATH, "".join(lines))
            index = len(f.readlines()) - 1
        else:
            lines = f.readlines()
            lines[index] = f"{id},\"{reformatted_task}\",{reformatted_points},{reformmateed_quadrants},{reformatted_images}\n"
            atomic_write(CACHE_PATH, "".join(lines))
    return index

def check_cache(id, step = None):
    try:
        with open(CACHE_PATH, "r") as f:
            # use pandas to make a query for the id
            df = pd.read_csv(f, quoting=csv.QUOTE_MINIMAL, on_bad_lines='skip')
            index = df.index[df["id"] == id].tolist()
            if len(index) == 0:
                return -1, None
            data_point = index[0]

            # check if length of datapoint's quadrants list is greater than the step
            if step is not None:
                if len(df["quadrants"][data_point].split(DATA_SEPERATOR)) > step:
                    return data_point, True
                return data_point, False
            return data_point, None
    except Exception as e:
        logging.error(f"Error reading cache file: {e}")
    return -1, None

def update_cached_data():
    # create json data from csv where the order is task,id,points,quadrants,images
    data = []
    try:
        with open(CACHE_PATH, "r") as f:
            df = pd.read_csv(f, quoting=csv.QUOTE_MINIMAL, on_bad_lines='skip')
            for i in range(len(df)):
                data.append({
                    "task": df["task"][i],
                    "id": df["id"][i],
                    "points": [tuple([round(float(x), 2) for x in point.split("~")]) for point in df["points"][i].split(DATA_SEPERATOR)],
                    "quadrants": [int(quadrant) for quadrant in df["quadrants"][i].split(DATA_SEPERATOR)],
                    "images": [image for image in df["images"][i].split(DATA_SEPERATOR)]
                })
            return data
    except Exception as e:
        logging.error(f"Error reading cache file: {e}")
    return data

def draw_quadrant_lines(img, line_color=(0, 255, 0), line_width=1):
    draw = ImageDraw.Draw(img)
    width, height = img.size
    draw.line((width/2, 0, width/2, height), fill=line_color, width=line_width)
    draw.line((0, height/2, width, height/2), fill=line_color, width=line_width)
    return img

def create_augmented_data(id, task, image, coordinate, draw_lines = False):
    img = Image.open(IMAGE_PATH + "/" + image)
    w, h = img.size
    x = coordinate[0] * w
    y = coordinate[1] * h
    original_path = image.split(".")[0]

    quadrants = []
    points = [(coordinate[0], coordinate[1])]

    images = []
    img_crop = [0, 0, w, h]
    for step_idx in range(K + 1):
        if draw_lines and step_idx < K:
            draw_quadrant_lines(img)
        w, h = img.size
        mid_w, mid_h = w / 2, h / 2
        new_x, new_y = x, y
        new_crop = img_crop
        # Determine the quadrant and adjust the bounds
        if x < mid_w and y < mid_h:  # Quadrant 2
            quadrant = 2
            new_crop = (0, 0, mid_w, mid_h)
        elif x >= mid_w and y < mid_h:  # Quadrant 1
            quadrant = 1
            new_crop = (mid_w, 0, w, mid_h)
            new_x -= mid_w  # Adjust x relative to the new image bounds
        elif x < mid_w and y >= mid_h:  # Quadrant 3
            quadrant = 3
            new_crop = (0, mid_h, mid_w, h)
            new_y -= mid_h  # Adjust y relative to the new image bounds
        else:  # x >= mid_w and y >= mid_h, Quadrant 4
            quadrant = 4
            new_crop = (mid_w, mid_h, w, h)
            new_x -= mid_w  # Adjust x relative to the new image bounds
            new_y -= mid_h  # Adjust y relative to the new image bounds
        
        if draw_lines:
            new_name = f"{original_path}_{' '.join([str(quadrant) for quadrant in quadrants])}_lines.png"
        else:
            new_name = f"{original_path}_{' '.join([str(quadrant) for quadrant in quadrants])}.png"

        quadrants.append(quadrant)
        points.append((new_x / (new_crop[2] - new_crop[0]), new_y / (new_crop[3] - new_crop[1])))
        
        path = os.path.join(IMAGE_PATH, new_name)
        images.append(path)
        if not os.path.exists(path): 
            img.save(path)
        img = img.crop(new_crop)
        x, y = new_x, new_y

    # print("images: " + str(images))
    # print("points: " + str(points))
    # print("quadrants: " + str(quadrants))
    return cache_data(task, id, images, points, quadrants)

def preprocess_tokens(
    source,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
    system_message: str = "You are a helpful assistant."
) -> Dict:
    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

    im_start = tokenizer.im_start_id
    im_end = tokenizer.im_end_id
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids + nl_tokens
    _user = tokenizer('user').input_ids + nl_tokens
    _assistant = tokenizer('assistant').input_ids + nl_tokens
    
    input_ids, targets = [], []
    # Apply prompt templates
    input_id, target = [], []
    system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
    input_id += system
    target += [im_start] + [IGNORE_TOKEN_ID] * (len(system)-3) + [im_end] + nl_tokens

    assert len(input_id) == len(target)
    for j, sentence in enumerate(source):
        role = roles[sentence["from"]]
        _input_id = tokenizer(role).input_ids + nl_tokens + \
            tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
        input_id += _input_id
        if role == '<|im_start|>user':
            _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id)-3) + [im_end] + nl_tokens
        elif role == '<|im_start|>assistant':
            _target = [im_start] + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                _input_id[len(tokenizer(role).input_ids)+1:-2] + [im_end] + nl_tokens
        else:
            raise NotImplementedError
        target += _target
    assert len(input_id) == len(target)

    input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
    target += [IGNORE_TOKEN_ID] * (max_len - len(target))
    input_ids.append(input_id[:max_len])
    targets.append(target[:max_len])

    input_ids = torch.tensor(input_ids, dtype=torch.int)
    targets = torch.tensor(targets, dtype=torch.int)
    ret = dict(
                input_ids=input_ids,
                labels=targets,
                attention_mask=(input_ids.ne(tokenizer.pad_token_id)),
            )
    ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
    return ret

def preprocess_sources(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
    system_message: str = "You are a helpful assistant."
) -> Dict:
    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

    im_start = tokenizer.im_start_id
    im_end = tokenizer.im_end_id
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids + nl_tokens
    _user = tokenizer('user').input_ids + nl_tokens
    _assistant = tokenizer('assistant').input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["user"]:
            source = source[1:]

        input_id, target = [], []
        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system
        target += [im_start] + [IGNORE_TOKEN_ID] * (len(system)-3) + [im_end] + nl_tokens
        assert len(input_id) == len(target)
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            _input_id = tokenizer(role).input_ids + nl_tokens + \
                tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
            input_id += _input_id
            if role == '<|im_start|>user':
                _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id)-3) + [im_end] + nl_tokens
            elif role == '<|im_start|>assistant':
                _target = [im_start] + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                    _input_id[len(tokenizer(role).input_ids)+1:-2] + [im_end] + nl_tokens
            else:
                raise NotImplementedError
            target += _target
        assert len(input_id) == len(target)
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))
        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])
    input_ids = torch.tensor(input_ids, dtype=torch.int)
    targets = torch.tensor(targets, dtype=torch.int)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )

def tokenize_cached_data(data, data_dict, step, tokenizer, max_len, context = False):
    # convert data from string json to dictionary
    task = data["task"]
    task.replace("\n", " ")
    # remove commas from task if there are
    if step == K:
        prompt = f"In this UI screenshot, what is the position of the element corresponding to the command \"{task}\" (with point)?"
    else:
        prompt = f"In this UI screenshot, what is the partition of the element corresponding to the command \"{task}\" (with quadrant number)?"

    points = data["points"]
    # print("before :" + str(points))
    points = [(round(p[0], 2), round(p[1], 2)) for p in points]
    # print("after :" + str(points))
    quadrants = data["quadrants"]
    images = data["images"]
    # print("images", images)

    if context:
        prompt_images = []
        # print("step", step)
        # print("num images", len(images))
        for i in range(step + 1):
            # print(i, images[i])
            prompt_images.append(images[i])
    else:
        prompt_images = [images[step]]

    images_and_prompt = "".join([f"Picture {i + 1}: <img>{image}</img>\n" for i, image in enumerate(prompt_images)]) + "" + prompt

    target_output = str(points[step] if step == K else quadrants[step])

    # print(images_and_prompt)
    # print(target_output)

    new_source =  \
    [
        {
            "from": "user",
            "value": images_and_prompt
        },
        {
            "from": "assistant",
            "value": target_output
        }
    ]

    data_dict[(data["id"], step)] = preprocess_tokens(new_source, tokenizer, max_len)
    return data_dict[(data["id"], step)]

class ReVL_loader_old(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int, draw_lines = False):
        super(ReVL_loader, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data = update_cached_data()
        self.cached_data_dict = {}

        self.steps_left = {}
        self.draw_lines = draw_lines
        if os.path.exists(STEPS_LEFT_PATH):
            if os.stat(STEPS_LEFT_PATH).st_size != 0:
                with open(STEPS_LEFT_PATH, "r") as f:
                    try:
                        self.steps_left = json.load(f)
                    except:
                        self.steps_left = {}

    def __len__(self):
        return len(self.raw_data)
            
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # get random step
        # print("i: " + str(i))
        if i not in self.steps_left or len(self.steps_left[i]) == 0:
            self.steps_left[i] = [k for k in range(K+1)]
        step_index = random.choice(range(len(self.steps_left[i])))
        # print("step index: " + str(step_index))
        step = self.steps_left[i].pop(step_index)
        # print("step: " + str(step))
        
        # upload steps_left to STEPS_LEFT_PATH

        steps_left_json = json.dumps(self.steps_left)
        atomic_write(STEPS_LEFT_PATH, steps_left_json)

        # print(str(i) + " " + str(step))
        if (i, step) in self.cached_data_dict:
            # print("cached already")
            return self.cached_data_dict[(i, step)]

        data_index, step_cached = check_cache(self.raw_data[i]["id"], step)
        if data_index != -1 and step_cached:
            self.cached_data = update_cached_data()
            if data_index >= len(self.cached_data):
                logging.error("Data Index out of bounds")
                data_index = random.randint(0, len(self.cached_data) - 1)
            return tokenize_cached_data(self.cached_data[data_index], self.cached_data_dict, step, self.tokenizer, self.max_len, context = CONTEXT)
        
        index = create_augmented_data(self.raw_data[i]["id"], self.raw_data[i]["task"], self.raw_data[i]["image"], self.raw_data[i]["coordinate"], self.draw_lines)
        self.cached_data = update_cached_data()
        if index >= len(self.cached_data):
            logging.error("Index out of bounds")
            index = random.randint(0, len(self.cached_data) - 1)
        return tokenize_cached_data(self.cached_data[index], self.cached_data_dict, step, self.tokenizer, self.max_len, context = CONTEXT)
    
class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess_sources(sources, tokenizer, max_len)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        print("num tokens: " + str(len(self.input_ids[i])))
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )