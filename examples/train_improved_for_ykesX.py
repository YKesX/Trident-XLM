#!/usr/bin/env python3
"""
IMPROVED TRAINING FOR @YKesX - Medium-scale quality training
Balances training time with quality for better Turkish outputs
"""

import json
import argparse
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model
import random

def load_balanced_jsonl(path, task_filter, max_examples=200):
    """Load balanced subset for quality training"""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            if r.get("task") == task_filter:
                rows.append(r)
                if len(rows) >= max_examples:
                    break
    
    random.seed(42)
    random.shuffle(rows)
    return Dataset.from_list(rows)

def tokenize_fn(tok, max_in, max_out):
    def fn(batch):
        # Tokenize input
        x = tok(batch["prompt"], truncation=True, max_length=max_in, padding=True, return_tensors=None)
        
        # Tokenize target
        with tok.as_target_tokenizer():
            y = tok(batch["target"], truncation=True, max_length=max_out, padding=True, return_tensors=None)
        
        x["labels"] = y["input_ids"]
        return x
    return fn

def train_improved_model():
    """Train improved model with balanced parameters"""
    print("🚀 IMPROVED TRAINING FOR @YKesX")
    print("Training with 200 examples, 4 epochs for quality")
    
    tok = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

    # Improved LoRA config
    lora = LoraConfig(
        r=24, lora_alpha=24, lora_dropout=0.08,  # Balanced rank
        target_modules=["q","k","v","o","wi_0","wi_1","wo"],
        bias="none", task_type="SEQ_2_SEQ_LM"
    )
    model = get_peft_model(model, lora)

    # Load balanced dataset
    train = load_balanced_jsonl("training_data/train.jsonl", "one_liner", max_examples=200)
    val = load_balanced_jsonl("training_data/val.jsonl", "one_liner", max_examples=40)

    tf = tokenize_fn(tok, max_in=400, max_out=100)  # Balanced sequence lengths
    train = train.map(tf, batched=True, remove_columns=train.column_names)
    val = val.map(tf, batched=True, remove_columns=val.column_names)

    # Quality-focused training settings
    args_tr = TrainingArguments(
        output_dir="flan_improved",
        learning_rate=8e-5,  # Balanced learning rate
        weight_decay=0.01,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=4,  # 4 epochs for better quality
        eval_strategy="epoch", 
        save_strategy="epoch",
        logging_steps=10,
        fp16=False,
        remove_unused_columns=False,
        warmup_steps=20,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=0,
        gradient_accumulation_steps=1
    )
    
    collator = DataCollatorForSeq2Seq(tok, model=model)
    trainer = Trainer(model=model, args=args_tr, data_collator=collator, train_dataset=train, eval_dataset=val)
    
    print("Starting improved training...")
    trainer.train()
    trainer.save_model("flan_improved")
    tok.save_pretrained("flan_improved")
    print("✅ Improved Flan-T5 model saved to flan_improved/")
    
    return "flan_improved"

if __name__ == "__main__":
    model_path = train_improved_model()
    print(f"✅ Improved model ready: {model_path}")