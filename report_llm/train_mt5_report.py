#!/usr/bin/env python
import argparse, json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model

def load_jsonl(path, task_filter):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            if r.get("task") == task_filter:
                rows.append(r)
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

def main(args):
    tok = AutoTokenizer.from_pretrained(args.base)  # Use default tokenizer settings
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base)

    # LoRA config - optimized for Turkish multilingual learning
    lora = LoraConfig(
        r=32, lora_alpha=32, lora_dropout=0.1,  # Higher rank for Turkish expressivity
        target_modules=["q","k","v","o","wi_0","wi_1","wo"],
        bias="none", task_type="SEQ_2_SEQ_LM"
    )
    model = get_peft_model(model, lora)

    train = load_jsonl(args.train, "report")
    val   = load_jsonl(args.val,   "report")

    tf = tokenize_fn(tok, max_in=512, max_out=192)
    train = train.map(tf, batched=True, remove_columns=train.column_names)
    val   = val.map(tf,   batched=True, remove_columns=val.column_names)

    args_tr = TrainingArguments(
        output_dir=args.out,
        learning_rate=3e-5,  # Lower learning rate for multilingual model
        weight_decay=0.01,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=6,  # Fewer epochs but better quality
        eval_strategy="epoch", 
        save_strategy="epoch",
        logging_steps=50,
        fp16=False,
        remove_unused_columns=False,
        warmup_steps=150,  # More warmup for multilingual
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=0,
        gradient_accumulation_steps=2,
        save_steps=200,
        eval_steps=200
    )
    collator = DataCollatorForSeq2Seq(tok, model=model)
    trainer = Trainer(model=model, args=args_tr, data_collator=collator, train_dataset=train, eval_dataset=val)
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print("Saved to", args.out)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="google/mt5-small")
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--out", required=True)
    main(p.parse_args())
