import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model
from trl import SFTConfig
from trl import SFTTrainer
import time
import json
import wandb
import argparse
from PIL import Image
import sys

# SYSTEM_MESSAGE = """"""

MODEL_ID = "Qwen/Qwen2-VL-7B"

def format_data(sample):
    final = []

    img_path = sample["conversations"][0]["Image"]

    img = Image.open(img_path).convert("RGB")

    width = img.size[0]
    height = img.size[1]

    # final.append({
    #     "role": "system",
    #     "content": [
    #         {
    #             "type": "text",
    #             "text": SYSTEM_MESSAGE
    #         }            
    #     ]
    # })

    for conversation in sample["conversations"]:
        if conversation["from"] == "user":
            zoom = conversation["zoom"]
            new_x1 = int(zoom[0] * width)
            new_y1 = int(zoom[1] * height)
            new_x2 = int(zoom[2] * width)
            new_y2 = int(zoom[3] * height)

            cropped_img = img.crop((new_x1, new_y1, new_x2, new_y2))

            text = conversation["value"].split("\n")[1]

            final.append({
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": cropped_img,
                    },
                    {
                        "type": "text",
                        "text": text
                    }
                ]
            })
        else:
            final.append({
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": conversation["value"]
                    }
                ]
            })

    return final

if __name__ == "__main__":
    print('starting script')
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2VL model.")
    parser.add_argument("--train_dataset", type=str, required=True, help="Path to the training dataset JSON file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the fine-tuned model.")
    parser.add_argument("--epochs", type=int, required=True)
    args = parser.parse_args()

    # Load training and evaluation datasets
    print('opening dataset')
    start_time = time.time()
    with open(args.train_dataset, "r") as f:
        train_dataset = json.load(f)
    print('done opening dataset')
    print("length", len(train_dataset))
    # Format datasets
    start_time = time.time()
    train_dataset = [format_data(sample) for sample in train_dataset]
    end_time = time.time()
    print('done formatting dataset')
    print(f"Time taken to format dataset: {end_time - start_time} seconds")

    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        # bnb_4bit_compute_dtype=torch.bfloat16
        bnb_4bit_compute_dtype=torch.float16
    )

    # Load model and tokenizer
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",
        # torch_dtype=torch.bfloat16,
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    def process_chat_template(conversation, add_generation_prompt=False):
        """
        Manually constructs a chat prompt string according to Qwen2-VL specifications.
        
        Args:
            conversation (list): A list of message dictionaries. Each message must have:
                - "role": One of "system", "user", or "assistant".
                - "content": Either a plain string or a list of segments.
                Each segment is a dict with:
                    - "type": "text", "image", or "video"
                    - For text segments: a "text" key with the message string.
                    - For image segments: an "image" key (its value is ignored here, a placeholder is used).
                    - For video segments: similarly a "video" key.
            add_generation_prompt (bool): If True, appends a new assistant prompt (i.e. "<|im_start|>assistant\n") at the end.
                                        
        Returns:
            str: The full formatted prompt string.
        """
        
        images = []

        for msg in conversation:
            for i, content in enumerate(msg["content"]):
                if content["type"] == "image":
                    images.append(content["image"])
                    msg["content"][i] = {"type": "image"}


        # Define the special tokens per Qwen2-VL's chat format.
        im_start = "<|im_start|>"
        im_end = "<|im_end|>"
        # Placeholders for non-text segments.
        vision_placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
        video_placeholder = "<|vision_start|><|video_pad|><|vision_end|>"
        
        parts = []
        for message in conversation:
            role = message.get("role", "").strip().lower()
            # Begin the message block with role header.
            message_block = f"{im_start}{role}\n"
            content = message.get("content", "")
            
            # If content is a list of segments, process each segment.
            if isinstance(content, list):
                segment_texts = []
                for segment in content:
                    seg_type = segment.get("type", "text").lower()
                    if seg_type == "text":
                        text = segment.get("text", "")
                        segment_texts.append(text)
                    elif seg_type == "image":
                        segment_texts.append(vision_placeholder)
                    elif seg_type == "video":
                        segment_texts.append(video_placeholder)
                    else:
                        segment_texts.append(str(segment))
                message_block += "\n".join(segment_texts)
            else:
                message_block += str(content)
                
            # End the message block.
            message_block += f"\n{im_end}\n"
            parts.append(message_block)
            
        if add_generation_prompt:
            parts.append(f"{im_start}assistant\n")
            
        return "".join(parts), images

    

    # Create a data collator to encode te0xt and image pairs
    def collate_fn(examples):
        print(examples)
        # Get the texts and images, and apply the chat template
        texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]  # Prepare texts for processing
        texts = [processor(text=text, images=images, return_tensors="pt", padding=True) for text, images in texts]  # Process the chat template to get the final text format
        texts = [text[0] for text,_ in texts]  # Extract the formatted text strings
        images = [images for _, images in texts]  # Extract the images from the processed texts
        print('texts', texts)
        print('images', images)
        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)  # Encode texts and images into tensors
        # image_inputs = [process_vision_info(example)[0] for example in examples]  # Process the images to extract inputs

        # print(image_inputs[0])

        # Tokenize the texts and process the images
        # batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)  # Encode texts and images into tensors

        print(batch)

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # batch = {key: tensor.to(device) for key, tensor in batch.items()}

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()  # Clone input IDs for labels
        labels[labels == processor.tokenizer.pad_token_id] = -100  # Mask padding tokens in labels

        # Ignore the image token index in the loss computation (model specific)
        # if isinstance(processor, Qwen2VLProcessor):  # Check if the processor is Qwen2VLProcessor
            # image_tokens = [151652, 151653, 151655]  # Specific image token IDs for Qwen2VLProcessor
        # else:
        image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]  # Convert image token to ID
        
        print("image tokens", image_tokens)

        # Mask image token IDs in the labels
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100  # Mask image token IDs in labels

        batch["labels"] = labels  # Add labels to the batch

        batch["input_ids"] = batch["input_ids"].long()

        return batch  # Return the prepared batch

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
        per_device_train_batch_size=1,  # Batch size for training
        per_device_eval_batch_size=2,  # Batch size for evaluation
        gradient_accumulation_steps=32,  # Steps to accumulate gradients
        gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
        # Optimizer and scheduler settings
        optim="adamw_torch_fused",  # Optimizer type
        learning_rate=2e-4,  # Learning rate for training
        lr_scheduler_type="constant",  # Type of learning rate scheduler
        # Logging and evaluation
        logging_steps=10,  # Steps interval for logging
        # eval_steps=10,  # Steps interval for evaluation
        # eval_strategy="steps",  # Strategy for evaluation
        # save_strategy="steps",  # Strategy for saving the model
        # save_steps=20,  # Steps interval for saving
        # metric_for_best_model="eval_loss",  # Metric to evaluate the best model
        # greater_is_better=False,  # Whether higher metric values are better
        # load_best_model_at_end=True,  # Load the best model after training
        # Mixed precision and gradient settings
        # bf16=True,  # Use bfloat16 precision
        fp16=True,  
        # tf32=True,  # Use TensorFloat-32 precision
        max_grad_norm=0.3,  # Maximum norm for gradient clipping
        warmup_ratio=0.03,  # Ratio of total steps for warmup
        # Hub and reporting
        push_to_hub=False,  # Whether to push model to Hugging Face Hub
        report_to="wandb",  # Reporting tool for tracking metrics
        # Gradient checkpointing settings
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Options for gradient checkpointing
        # Dataset configuration
        # dataset_text_field="",  # Text field in dataset
        dataset_kwargs={"skip_prepare_dataset": True},  # Additional dataset options
        #max_seq_length=1024  # Maximum sequence length for input
    )

    training_args.remove_unused_columns = False  # Keep unused columns in dataset

    wandb.init(
        project="qwen2-7b-instruct-experiment",  # change this
        name="qwen2-7b-instruct-experiment",  # change this
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