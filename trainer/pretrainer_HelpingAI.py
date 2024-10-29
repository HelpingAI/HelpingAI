from configparser import ConfigParser
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from transformers import TrainingArguments, AutoTokenizer
from trl import SFTTrainer

from HelpingAI_.configuration_HelpingAI import HelpingAIConfig
from HelpingAI_.modeling_HelpingAI import HelpingAIForCausalLM
from HelpingAI_.tokenization_HelpingAI_fast import HelpingAITokenizerFast


@dataclass
class ModelTrainingArguments:
    model_name: str = field(default="LLM")
    vocab_size: int = field(default=50281)
    dataset_name: str = field(default="roneneldan/TinyStories")
    dataset_split: str = field(default="train")
    output_dir: str = field(default="M_outputs")
    per_device_train_batch_size: int = field(default=2)
    gradient_accumulation_steps: int = field(default=1)
    warmup_steps: int = field(default=2)
    max_steps: int = field(default=20000)
    learning_rate: float = field(default=1e-4)
    logging_steps: int = field(default=100)
    save_steps: int = field(default=20000)
    optim: str = field(default="paged_adamw_32bit")
    report_to: str = field(default="none")
    seed: int = field(default=42)
    max_seq_length: int = field(default=2048)
    dataset_num_proc: int = field(default=2)
    dataset_text_field: str = field(default="text")
    push_to_hub_token: Optional[str] = field(default=None)


def train_model(args: ModelTrainingArguments):
    # Model Configuration
    configuration = HelpingAIConfig(
        vocab_size=args.vocab_size,
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        head_dim=64,
        num_local_experts=4,
        num_experts_per_tok=1,
        intermediate_size=1024,
        hidden_act="silu",
        hidden_dropout=0.1,
        attention_dropout=0.1,
        classifier_dropout=0.1,
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        layer_norm_eps=1e-5,
        use_cache=True,
        bos_token_id=50278,
        eos_token_id=50279,
        num_key_value_heads=8,
        norm_eps=1e-05,
    )

    # Model
    model = HelpingAIForCausalLM(configuration)

    # Tokenizer
    tokenizer = HelpingAITokenizerFast.from_pretrained(r"C:\Users\koula\OneDrive\Desktop\model\tokenizer", local_files_only=False)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Wow! 😮 Our model has {model.num_parameters():,} parameters!")

    # Dataset
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    dataset = dataset.shuffle(seed=args.seed)

    print(f'Amazing! 🌟 We have {len(dataset)} prompts to work with!')
    print(f'Our dataset columns are: {dataset.column_names}')

    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field=args.dataset_text_field,
        max_seq_length=args.max_seq_length,
        dataset_num_proc=args.dataset_num_proc,
        args=TrainingArguments(
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            logging_steps=args.logging_steps,
            output_dir=args.output_dir,
            overwrite_output_dir=True,
            save_steps=args.save_steps,
            optim=args.optim,
            report_to=args.report_to,
        ),
    )

    # Training
    trainer.train()
    trainer.save_model(args.model_name)

    # Push to Hub (if token provided)
    if args.push_to_hub_token:
        try:
            model.push_to_hub(args.model_name, use_temp_dir=False, token=args.push_to_hub_token)
            tokenizer.push_to_hub(args.model_name, use_temp_dir=False, token=args.push_to_hub_token)
            print("Model and tokenizer pushed to the Hub successfully!")
        except Exception as e:
            print(f"Error pushing to the Hub: {e}")
    else:
        print("push_to_hub_token not provided. Model and tokenizer not pushed to the Hub.")


if __name__ == "__main__":
    train_model(ModelTrainingArguments())
