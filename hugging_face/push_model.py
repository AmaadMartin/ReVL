from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import transformers
from peft import AutoPeftModelForCausalLM
torch.manual_seed(1234)

MODEL_DIRECTORY = "./output_qwen/{model_name}"
MODEL = "k_2_model"

if __name__ == '__main__':
    model = AutoPeftModelForCausalLM.from_pretrained(
    MODEL_DIRECTORY.format(model_name=MODEL),
    device_map="cuda",
    trust_remote_code=True,
    fp16=True).eval()

    model = model.merge_and_unload(progressbar=True)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_DIRECTORY.format(model_name=MODEL),
        cache_dir=None,
        model_max_length=2048,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    tokenizer.pad_token_id = tokenizer.eod_id

    model.push_to_hub(MODEL)
    tokenizer.push_to_hub(MODEL)

    
