#!/usr/bin/env python3
"""
Quick training script for fast model validation - addressing @YKesX feedback
Trains with fewer examples and epochs to get working models quickly
"""

import json
import argparse
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model
import random

def load_subset_jsonl(path, task_filter, max_examples=100):
    """Load a subset of examples for quick training"""
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

def quick_train_flan_t5():
    """Quick Flan-T5 training for immediate testing"""
    print("🚀 QUICK FLAN-T5 TRAINING FOR @YKesX")
    print("Training with 80 examples, 2 epochs for fast validation")
    
    tok = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

    # Smaller LoRA config for speed
    lora = LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q","k","v","o","wi_0","wi_1","wo"],
        bias="none", task_type="SEQ_2_SEQ_LM"
    )
    model = get_peft_model(model, lora)

    # Load subset of data
    train = load_subset_jsonl("training_data/train.jsonl", "one_liner", max_examples=80)
    val = load_subset_jsonl("training_data/val.jsonl", "one_liner", max_examples=20)

    tf = tokenize_fn(tok, max_in=256, max_out=80)  # Shorter sequences for speed
    train = train.map(tf, batched=True, remove_columns=train.column_names)
    val = val.map(tf, batched=True, remove_columns=val.column_names)

    # Fast training settings
    args_tr = TrainingArguments(
        output_dir="flan_quick",
        learning_rate=1e-4,  # Higher LR for faster convergence
        weight_decay=0.01,
        per_device_train_batch_size=4,  # Larger batch
        per_device_eval_batch_size=4,
        num_train_epochs=2,  # Just 2 epochs for speed
        eval_strategy="epoch", 
        save_strategy="epoch",
        logging_steps=5,
        fp16=False,
        remove_unused_columns=False,
        warmup_steps=10,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=0
    )
    
    collator = DataCollatorForSeq2Seq(tok, model=model)
    trainer = Trainer(model=model, args=args_tr, data_collator=collator, train_dataset=train, eval_dataset=val)
    
    print("Starting training...")
    trainer.train()
    trainer.save_model("flan_quick")
    tok.save_pretrained("flan_quick")
    print("✅ Quick Flan-T5 model saved to flan_quick/")
    
    return "flan_quick"

if __name__ == "__main__":
    model_path = quick_train_flan_t5()
    print(f"✅ Model ready for testing at: {model_path}")