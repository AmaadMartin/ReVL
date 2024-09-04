# This code is based on the revised code from fastchat based on tatsu-lab/stanford_alpaca.


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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate.utils import DistributedType
import random
from PIL import Image
import pandas as pd
import argparse
import csv
import json
from atomicwrites import atomic_write

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
CHECKPOINT = "output_qwen/checkpoint-1200"
IMAGE_PATH = './data/images'
CACHE_PATH = './data/cached_data.csv'
K = 3
WANDB_PROJECT = f"revl_k_{K}"
CONTEXT = True
DATA_SEPERATOR = "`"
EVAL_DATA_PATH = "./json_data/seeclick_test_split_new_format.json"
STEPS_LEFT_PATH = "./finetune/state/steps_left.json"

os.environ["WANDB_PROJECT"] = WANDB_PROJECT

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen-7B")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False
    fix_vit: bool = True


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["c_attn", "attn.c_proj", "w1", "w2"] ##["in_proj","out_proj","c_fc"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str, bias="none"):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        if trainer.args.use_lora:
            state_dict = get_peft_state_maybe_zero_3(
                trainer.model.named_parameters(), bias
            )
        else:
            state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)


def create_augmented_data(id, task, image, coordinate):
    img = Image.open(IMAGE_PATH + "/" + image)
    w, h = img.size
    x = coordinate[0] * w
    y = coordinate[1] * h
    original_path = image.split(".")[0]

    quadrants = []
    points = [(coordinate[0], coordinate[1])]

    images = []
    img_crop = [0, 0, w, h]
    for _ in range(K + 1):
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

def check_cache(id, step = None):
    with open(CACHE_PATH, "r") as f:
        # use pandas to make a query for the id
        df = pd.read_csv(f)
        index = df.index[df["id"] == id].tolist()
        if len(index) == 0:
            return -1
        data_point = index[0]

        # check if length of datapoint's quadrants list is greater than the step
        if step is not None:
            if len(df["quadrants"][data_point].split(DATA_SEPERATOR)) > step:
                return data_point, True
            return data_point, False
        return data_point, None

def cache_data(task, id, images, points, quadrants):
    # if task and id already in cache, add the extra points and quadrants to their respective lists
    # csv format: task, id, points, quadrants
    with open(CACHE_PATH, "a+") as f:
        # find the index of the task and id in the csv file
        # replace \n with blank space
        reformatted_task = task.replace("\n", " ")
        reformatted_task = reformatted_task.replace('"', '""')
        reformatted_task = reformatted_task.rstrip("\n")
        reformatted_points = DATA_SEPERATOR.join([str(x[0]) + "~" + str(x[1]) for x in points])
        reformmateed_quadrants = DATA_SEPERATOR.join([str(quadrant) for quadrant in quadrants])
        reformatted_images = DATA_SEPERATOR.join(images)
        index, _ = check_cache(id)
        if index == -1:
            f.write(f"{id},\"{reformatted_task}\",{reformatted_points},{reformmateed_quadrants},{reformatted_images}\n")
            index = len(f.readlines()) - 1
        else:
            f.seek(0)
            lines = f.readlines()
            lines[index] = f"{id},\"{reformatted_task}\",{reformatted_points},{reformmateed_quadrants},{reformatted_images}\n"
            f.seek(0)
            f.writelines(lines)
    return index

def update_cached_data():
    # create json data from csv where the order is task,id,points,quadrants,images
    data = []
    with open(CACHE_PATH, "r") as f:
        df = pd.read_csv(f)
        for i in range(len(df)):
            data.append({
                "task": df["task"][i],
                "id": df["id"][i],
                "points": [tuple([float(x) for x in point.split("~")]) for point in df["points"][i].split(DATA_SEPERATOR)],
                "quadrants": [int(quadrant) for quadrant in df["quadrants"][i].split(DATA_SEPERATOR)],
                "images": [image for image in df["images"][i].split(DATA_SEPERATOR)]
            })
    return data

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



class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess_tokens(sources, tokenizer, max_len)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data = update_cached_data()
        self.cached_data_dict = {}

        self.steps_left = {}
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
        with atomic_write(STEPS_LEFT_PATH, "w") as f:
            f.write(steps_left_json)

        # print(str(i) + " " + str(step))
        if (i, step) in self.cached_data_dict:
            # print("cached already")
            return self.cached_data_dict[(i, step)]

        data_index, step_cached = check_cache(self.raw_data[i]["id"], step)
        if data_index != -1 and step_cached:
            # print("caching")
            self.cached_data = update_cached_data()
            return tokenize_cached_data(self.cached_data[data_index], self.cached_data_dict, step, self.tokenizer, self.max_len, context = CONTEXT)
        
        # print("augmenting and caching")
        index = create_augmented_data(self.raw_data[i]["id"], self.raw_data[i]["task"], self.raw_data[i]["image"], self.raw_data[i]["coordinate"])
        self.cached_data = update_cached_data()

        return tokenize_cached_data(self.cached_data[index], self.cached_data_dict, step, self.tokenizer, self.max_len, context = CONTEXT)


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args, max_len,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = LazySupervisedDataset
    rank0_print("Loading data...")

    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer, max_len=max_len)

    if EVAL_DATA_PATH:
        eval_json = json.load(open(EVAL_DATA_PATH, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer, max_len=max_len)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)

def train():
    global local_rank
    
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()
    print("model_args:", model_args)
    print("data_args:", data_args)
    print("training_args:", training_args)
    print("lora_args:", lora_args)

    if getattr(training_args, 'deepspeed', None) and getattr(lora_args, 'q_lora', False):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    local_rank = training_args.local_rank

    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning(
                "FSDP or ZeRO3 are not incompatible with QLoRA."
            )

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    config.use_cache = False

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        device_map=device_map,
        trust_remote_code=True,
        quantization_config=GPTQConfig(
            bits=4, disable_exllama=True
        )
        if training_args.use_lora and lora_args.q_lora
        else None,
    )

    if not training_args.use_lora:
        if training_args.fix_vit and hasattr(model,'transformer') and hasattr(model.transformer,'visual'):
            model.transformer.visual.requires_grad_(False)
            if hasattr(model.transformer.visual,'attn_pool'):
                model.transformer.visual.attn_pool.requires_grad_(True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    tokenizer.pad_token_id = tokenizer.eod_id

    if training_args.use_lora:
        if lora_args.q_lora or "chat" in model_args.model_name_or_path.lower():
            modules_to_save = None
        else:
            modules_to_save = ["wte", "lm_head"]
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
            modules_to_save=modules_to_save  # This argument serves for adding new tokens.
        )
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

        model = get_peft_model(model, lora_config)

        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

    # Load data
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args, max_len=training_args.model_max_length
    )

    # Start trainner
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    trainer.train(resume_from_checkpoint=CHECKPOINT)
    trainer.save_state()

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir, bias=lora_args.lora_bias)


if __name__ == "__main__":
    train()