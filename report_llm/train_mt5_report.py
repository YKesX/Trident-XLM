#!/usr/bin/env python
import argparse, json, os, sys, types, importlib.machinery
from datasets import Dataset
# Harden environment to torch-only to avoid TF/NumPy issues
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("USE_TORCH", "1")
if "tensorflow" not in sys.modules:
    _tf = types.ModuleType("tensorflow")
    _tf.__spec__ = importlib.machinery.ModuleSpec("tensorflow", loader=None)
    class _DummyTensor:
        pass
    _tf.Tensor = _DummyTensor
    sys.modules["tensorflow"] = _tf
if "ml_dtypes" not in sys.modules:
    _mld = types.ModuleType("ml_dtypes")
    _mld.__spec__ = importlib.machinery.ModuleSpec("ml_dtypes", loader=None)
    sys.modules["ml_dtypes"] = _mld
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model

def load_jsonl(path, task_filter, limit: int | None = None):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            if r.get("task") == task_filter:
                rows.append(r)
            # Only apply limit if it's specified and > 0
            if limit is not None and limit > 0 and len(rows) >= limit:
                break
    
    if len(rows) == 0:
        raise ValueError(f"No samples found for task '{task_filter}' in {path}. "
                        f"Check that the file exists and contains data with task='{task_filter}'")
    
    print(f"Loaded {len(rows)} samples for task '{task_filter}' from {path}")
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

    # Parse limit arguments - treat 0 as None (no limit)
    limit_train = int(args.limit_train) if args.limit_train and int(args.limit_train) > 0 else None
    limit_val = int(args.limit_val) if args.limit_val and int(args.limit_val) > 0 else None
    
    train = load_jsonl(args.train, "report", limit=limit_train)
    val   = load_jsonl(args.val,   "report", limit=limit_val)

    # Validate dataset sizes before proceeding
    if len(train) == 0:
        raise ValueError(f"Training dataset is empty after loading from {args.train}")
    if len(val) == 0:
        raise ValueError(f"Validation dataset is empty after loading from {args.val}")
    
    print(f"Training with {len(train)} training samples and {len(val)} validation samples")

    tf = tokenize_fn(tok, max_in=512, max_out=192)
    train = train.map(tf, batched=True, remove_columns=train.column_names)
    val   = val.map(tf,   batched=True, remove_columns=val.column_names)
    
    # Final validation after tokenization
    if len(train) == 0:
        raise ValueError("Training dataset is empty after tokenization")
    if len(val) == 0:
        raise ValueError("Validation dataset is empty after tokenization")

    args_tr = TrainingArguments(
        output_dir=args.out,
        learning_rate=3e-5,  # Lower learning rate for multilingual model
        weight_decay=0.01,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=float(args.epochs),  # configurable
        max_steps=int(args.max_steps) if args.max_steps else -1,
        eval_strategy="epoch", 
        save_strategy="epoch",
        logging_steps=50,
        report_to=["none"],  # avoid tensorboard/tf deps
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
    p.add_argument("--epochs", default="6")
    p.add_argument("--max-steps", dest="max_steps", default="0")
    p.add_argument("--limit-train", dest="limit_train", default="0")
    p.add_argument("--limit-val", dest="limit_val", default="0")
    main(p.parse_args())
