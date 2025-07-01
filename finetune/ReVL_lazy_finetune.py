import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig, GPTQConfig
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model
from trl import SFTConfig
from PIL import Image
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset
import wandb
from argparse import ArgumentParser
from trl import SFTTrainer
import json
import time

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="Qwen/Qwen2-VL-7B-Instruct",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    load_in_4bit: Optional[bool] = field(
        default=True,
        metadata={"help": "Load model in 4-bit precision"}
    )
    use_double_quant: Optional[bool] = field(
        default=True,
        metadata={"help": "Use double quantization in 4-bit mode"}
    )
    quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Options: fp4, nf4"}
    )
    compute_dtype: Optional[str] = field(
        default="bfloat16",
        metadata={"help": "Compute dtype for 4-bit quantization. Options: float16, bfloat16, float32"}
    )

@dataclass 
class LoraArguments:
    lora_alpha: Optional[int] = field(
        default=16,
        metadata={"help": "Alpha parameter for LoRA scaling"}
    )
    lora_dropout: Optional[float] = field(
        default=0.05,
        metadata={"help": "Dropout probability for LoRA layers"}
    )
    lora_r: Optional[int] = field(
        default=8,
        metadata={"help": "LoRA attention dimension"}
    )
    lora_bias: Optional[str] = field(
        default="none",
        metadata={"help": "Bias type for LoRA. Options: none, all, lora_only"}
    )
    lora_target_modules: Optional[str] = field(
        default="q_proj,v_proj",
        metadata={"help": "Comma-separated list of target modules to apply LoRA"}
    )

@dataclass
class TrainingArguments:
    output_dir: Optional[str] = field(
        default="qwen2-7b-instruct-trl-sft-ChartQA",
        metadata={"help": "Output directory for model and checkpoints"}
    )
    num_train_epochs: Optional[int] = field(
        default=3,
        metadata={"help": "Number of training epochs"}
    )
    per_device_train_batch_size: Optional[int] = field(
        default=4,
        metadata={"help": "Batch size per device during training"}
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=4,
        metadata={"help": "Batch size per device during evaluation"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=8,
        metadata={"help": "Number of updates steps to accumulate before backward pass"}
    )
    learning_rate: Optional[float] = field(
        default=2e-4,
        metadata={"help": "Initial learning rate"}
    )
    max_grad_norm: Optional[float] = field(
        default=0.3,
        metadata={"help": "Max gradient norm for clipping"}
    )
    warmup_ratio: Optional[float] = field(
        default=0.03,
        metadata={"help": "Ratio of steps for learning rate warmup"}
    )
    logging_steps: Optional[int] = field(
        default=10,
        metadata={"help": "Number of steps between logging updates"}
    )
    eval_steps: Optional[int] = field(
        default=10,
        metadata={"help": "Number of steps between evaluations"}
    )
    save_steps: Optional[int] = field(
        default=20,
        metadata={"help": "Number of steps between model saves"}
    )
    K: Optional[int] = field(
        default=2,
        metadata={"help": "Number of inference steps for ReVL"}
    )
    train_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to training data"}
    )
    train_data_size: Optional[int] = field(
        default=10000,
        metadata={"help": "Size of training data"}
    )
    eval_data_size: Optional[int] = field(
        default=1000,
        metadata={"help": "Size of evaluation data"}
    )
model_id = "Qwen/Qwen2-VL-7B-Instruct"

def create_text_to_point_conversation_from_datapoint(image, task, coordinates):
    # 2     |      1
    #       |  
    # --------------  
    #       |
    # 3     |      4
    conversation = []
    x, y = coordinates
    quadrant = 0
    images = [image]
    for _ in range(training_args.K):
        if x >= 0.5 and y <= 0.5:
            x, y = ((x - 0.5) * 2, y * 2)
            quadrant = 1
            images.append(images[-1].crop((images[-1].width // 2, 0, images[-1].width, images[-1].height // 2)))
        elif x <= 0.5 and y <= 0.5:
            x, y = (x * 2, y * 2)
            quadrant = 2
            images.append(images[-1].crop((0, 0, images[-1].width // 2, images[-1].height // 2)))
        elif x <= 0.5 and y >= 0.5:
            x, y = (x * 2, (y - 0.5) * 2)
            quadrant = 3
            images.append(images[-1].crop((0, images[-1].height // 2, images[-1].width // 2, images[-1].height)))
        elif x >= 0.5 and y >= 0.5:
            x, y = ((x - 0.5) * 2, (y - 0.5) * 2)
            quadrant = 4
            images.append(images[-1].crop((images[-1].width // 2, images[-1].height // 2, images[-1].width, images[-1].height)))
        else:
            raise ValueError(f"Invalid coordinates: {coordinates}")
            
        conversation.append({
            "role": "user",
            "content": [
                {
                    "type": "image" 
                },
                {
                    "type": "text",
                    "text": f"In this UI screenshot, what is the partition of the element corresponding to the command \"{task}\" (with quadrant number)?",
                }
            ],
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": f"{quadrant}"
                }
            ]
        })  

    # add the last image to the conversation
    conversation.append({
        "role": "user",
        "content": [
            {
                "type": "image" 
            },
            {
                "type": "text",
                "text": f"In this UI screenshot, what is the position of the element corresponding to the command \"{task}\" (with point)?",
            }
        ],
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": f"({x}, {y})"
            }   
        ]
    })

    return conversation, images

def create_text_to_bbox_conversation_from_datapoint(image, task, bbox):
    images = [image]
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image"
                },
                {
                    "type": "text",
                    "text": f"In this UI screenshot, what is the position of the element corresponding to the command \"{task}\" (with bbox)?",
                }
            ]
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": f"({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})"
                }
            ]
        }
    ]
    return conversation, images

def create_point_to_text_conversation_from_datapoint(image, element, coordinates):
    images = [image]
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image"
                },
                {
                    "type": "text",
                    "text": f"In this UI screenshot, what is the element corresponding to the point ({coordinates[0]}, {coordinates[1]}) on the screen?",
                }
            ]
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": f"{element}"
                }
            ]
        }
    ]
    return conversation, images

def create_bbox_to_text_conversation_from_datapoint(image, element, bbox):
    images = [image]
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image"
                },
                {
                    "type": "text",
                    "text": f"In this UI screenshot, what is the element corresponding to the bounding box ({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}) on the screen?",
                }
            ]
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": f"{element}"
                }
            ]
        }
    ]
    return conversation, images

def create_ui_summarization_conversation_from_datapoint(image, description):
    images = [image]
    conversation = [
        {
            "role": "user", 
            "content": [
                {
                    "type": "image"
                },
                {
                    "type": "text",
                    "text": f"Can you provide a detailed description of the interface screenshot shown?",
                }
            ]
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": f"{description}"
                }
            ]
        }
    ]
    return conversation, images

def create_widget_captioning_conversation_from_datapoint(image, coordinates, caption):
    images = [image]
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image"
                },
                {
                    "type": "text",
                    "text": f"From this UI screenshot, can you caption the widget corresponding to the point \"{coordinates}\" on the screen?",
                }
            ]
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": f"{caption}"
                }
            ]
        }
    ]
    return conversation, images

def create_general_conversation_from_datapoint(images, conversation):
    return conversation, images

def create_conversation_from_datapoint(images, text, coordinates, bbox, data_type, conversation = None):
    if data_type == "text_to_point":
        return create_text_to_point_conversation_from_datapoint(images[0], text, coordinates)
    elif data_type == "text_to_bbox":
        return create_text_to_bbox_conversation_from_datapoint(images[0], text, bbox)
    elif data_type == "point_to_text":
        return create_point_to_text_conversation_from_datapoint(images[0], text, coordinates)
    elif data_type == "bbox_to_text":
        return create_bbox_to_text_conversation_from_datapoint(images[0], text, bbox)
    elif data_type == "ui_summarization":
        return create_ui_summarization_conversation_from_datapoint(images[0], text)
    elif data_type == "widget_captioning":
        return create_widget_captioning_conversation_from_datapoint(images[0], coordinates, text)
    elif data_type == "general_llava":
        return create_general_conversation_from_datapoint(images, conversation)
    else:
        raise ValueError(f"Invalid type: {type}")

# Create a data collator to encode text and image pairs
def collate_fn(examples):
    conversations, pre_processed_image_inputs = zip(*[create_conversation_from_datapoint(example["images"], example["text"], example["coordinates"], example["bbox"], example["data_type"]) for example in examples])
    texts = [processor.apply_chat_template(conversation, tokenize=False) for conversation in conversations]  # Prepare texts for processing 
    image_inputs = [[process_vision_info(image) for image in images] for images in pre_processed_image_inputs]

    # Tokenize the texts and process the images
    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)  # Encode texts and images into tensors

    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()  # Clone input IDs for labels
    labels[labels == processor.tokenizer.pad_token_id] = -100  # Mask padding tokens in labels

    # Ignore the image token index in the loss computation (model specific)
    if isinstance(processor, Qwen2VLProcessor):  # Check if the processor is Qwen2VLProcessor
        image_tokens = [151652, 151653, 151655]  # Specific image token IDs for Qwen2VLProcessor
    else:
        image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]  # Convert image token to ID

    # Mask image token IDs in the labels
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100  # Mask image token IDs in labels

    batch["labels"] = labels  # Add labels to the batch

    return batch  # Return the prepared batch

if __name__ == "__main__":
    #     --model_name_or_path $MODEL \
    # --train_dataset $DATA \
    # --output_dir $OUT \
    # --epochs $EPOCHS \
    # --per_device_train_batch_size 4 \
    # --per_device_eval_batch_size 4 \
    # --gradient_accumulation_steps 8 \
    # --learning_rate 2e-4 \
    # --max_grad_norm 0.3 \
    # --warmup_ratio 0.03 \
    # --logging_steps 10 \
    # --eval_steps 10 \
    # --save_steps 20 \
    # --K $K \
    # --train_data_size $TRAIN_DATA_SIZE \
    # --eval_data_size $EVAL_DATA_SIZE
    parser = ArgumentParser(description="Fine-tune Qwen2VL model.")
    parser.add_argument("--train_dataset", type=str, required=True, help="Path to the training dataset JSON file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the fine-tuned model.")
    parser.add_argument("--epochs", type=int, required=True, default=3)
    parser.add_argument("--train_data_size", type=int, required=True, default=10000)
    parser.add_argument("--eval_data_size", type=int, required=True, default=1000)
    parser.add_argument("--per_device_train_batch_size", type=int, required=True, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, required=True, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, required=True, default=8)
    parser.add_argument("--learning_rate", type=float, required=True, default=2e-4)
    parser.add_argument("--max_grad_norm", type=float, required=True, default=0.3)
    parser.add_argument("--warmup_ratio", type=float, required=True, default=0.03)
    parser.add_argument("--logging_steps", type=int, required=True, default=10)
    parser.add_argument("--eval_steps", type=int, required=True, default=10)
    parser.add_argument("--save_steps", type=int, required=True, default=20)
    parser.add_argument("--K", type=int, required=True, default=3)
    parser.add_argument("--model_name_or_path", type=str, required=True, default="Qwen/Qwen2-VL-7B-Instruct")
    args, unknown = parser.parse_known_args()

    print('starting script')
    # Load training and evaluation datasets
    print('opening dataset')
    start_time = time.time()
    with open(args.train_dataset, "r") as f:
        train_dataset = json.load(f)
    train_dataset = train_dataset[:args.train_data_size]
    with open(args.train_dataset, "r") as f:
        eval_dataset = json.load(f)
    eval_dataset = eval_dataset[args.train_data_size:args.train_data_size + args.eval_data_size]
    print('done opening dataset')
    print("length", len(train_dataset))

    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model and tokenizer
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        # quantization_config=GPTQConfig(
        #     bits=4, disable_exllama=True, dataset="c4"
        # )
    )
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)

    # Configure LoRA
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=8,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )

    # Apply PEFT model adaptation
    peft_model = get_peft_model(model, peft_config)

    # Configure training arguments
    training_args = SFTConfig(
        output_dir=args.output_dir,  # Directory to save the model
        num_train_epochs=args.epochs,  # Number of training epochs
        per_device_train_batch_size=args.per_device_train_batch_size,  # Batch size for training
        per_device_eval_batch_size=args.per_device_eval_batch_size,  # Batch size for evaluation
        gradient_accumulation_steps=args.gradient_accumulation_steps,  # Steps to accumulate gradients
        gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
        # Optimizer and scheduler settings
        optim="adamw_torch_fused",  # Optimizer type
        learning_rate=args.learning_rate,  # Learning rate for training
        lr_scheduler_type="constant",  # Type of learning rate scheduler
        # Logging and evaluation
        logging_steps=args.logging_steps,  # Steps interval for logging
        eval_steps=args.eval_steps,  # Steps interval for evaluation
        eval_strategy="no",  # Strategy for evaluation
        save_strategy="no",  # Strategy for saving the model
        save_steps=args.save_steps,  # Steps interval for saving
        metric_for_best_model="eval_loss",  # Metric to evaluate the best model
        greater_is_better=False,  # Whether higher metric values are better
        load_best_model_at_end=True,  # Load the best model after training
        # # Mixed precision and gradient settings
        # bf16=True,  # Use bfloat16 precision
        # tf32=False,  # Use TensorFloat-32 precision (CHANGED TO FALSE)
        fp16=True,  
        max_grad_norm=args.max_grad_norm,  # Maximum norm for gradient clipping
        warmup_ratio=args.warmup_ratio,  # Ratio of total steps for warmup
        # Hub and reporting
        push_to_hub=False,  # Whether to push model to Hugging Face Hub
        report_to="wandb",  # Reporting tool for tracking metrics
        # Gradient checkpointing settings
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Options for gradient checkpointing
        # Dataset configuration
        dataset_text_field="",  # Text field in dataset
        dataset_kwargs={"skip_prepare_dataset": True},  # Additional dataset options
        #max_seq_length=1024  # Maximum sequence length for input
        # train_data_path=args.train_dataset,
        # train_data_size=args.train_data_size, 
        # eval_data_size=args.eval_data_size,
    )

    training_args.remove_unused_columns = False  # Keep unused columns in dataset

    wandb.init(
        project="qwen2-7b-instruct-trl-sft-ChartQA",  # change this
        name=args.output_dir,  # change this
        config=training_args,
    )


    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        data_collator=collate_fn,
        peft_config=peft_config,
        tokenizer=processor.tokenizer,
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)