import subprocess
import sys

try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "trl", "transformers", "accelerate", "peft", "datasets", "bitsandbytes", "einops", "torch", "torchvision", "torchaudio", "flash-attn"])
except subprocess.CalledProcessError as e:
    print(f"Error installing requirements: {e}")
    sys.exit(1)

import gc
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)
from trl import ORPOConfig, ORPOTrainer, setup_chat_format

# Model
base_model = "google/gemma-1.1-2b-it"
new_model = "EMO-2B"


# Set torch dtype and attention implementation
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    torch_dtype = torch.float16

# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    llm_int8_enable_fp32_cpu_offload=True,
    device_map={'':torch.cuda.current_device()}

)

# LoRA config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']


)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

# Load model
try:
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

model, tokenizer = setup_chat_format(model, tokenizer)
model = prepare_model_for_kbit_training(model)
# -------------------------------------------------------------------------------------
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

try:
    dataset = load_dataset("OEvortex/EmotionalIntelligence-10K", split = "train")
    dataset = dataset.map(formatting_prompts_func, batched = True,)
except Exception as e:
    print(f"Error loading dataset: {e}. Please ensure the dataset is available and accessible.")
    exit(1)
# -------------------------------------------------------------------------------------

from transformers import TrainingArguments

output_dir = "./HelpingAI-1B-v2"
per_device_train_batch_size = 3
gradient_accumulation_steps = 4
optim = "paged_adamw_32bit"
save_steps = 10
logging_steps = 10
learning_rate = 2e-4
max_grad_norm = 0.3
max_steps = 10
warmup_ratio = 0.03
lr_scheduler_type = "constant"

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=True,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
)

from trl import SFTTrainer

max_seq_length = 512

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    
)

try:
    trainer.train()
    trainer.save_model(new_model)
except Exception as e:
    print(f"Error during training or saving: {e}")
    exit(1)


tokenizer = AutoTokenizer.from_pretrained(base_model)
fp16_model = AutoModelForCausalLM.from_pretrained(
    base_model,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    offload_buffers=True
)
fp16_model, tokenizer = setup_chat_format(fp16_model, tokenizer)

# Merge adapter with base model
try:
    model = PeftModel.from_pretrained(fp16_model, new_model)
    model = model.merge_and_unload()
    model.push_to_hub(new_model, use_temp_dir=False, token="")
    tokenizer.push_to_hub(new_model, use_temp_dir=False, token="")
except Exception as e:
    print(f"Error merging, unloading, or pushing to hub: {e}")
    exit(1)
