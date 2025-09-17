#!/usr/bin/env python3
"""
Mock training test to validate the training pipeline without doing full training.
This loads models, processes data, and validates the pipeline end-to-end.
"""
import os
import sys
import json
import tempfile
import argparse

# Add the report_llm package to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_flan_pipeline(train_path, val_path, temp_dir):
    """Test the one-liner training pipeline without full training."""
    print("🧪 Testing Flan-T5 one-liner training pipeline...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        
        # Load a small model for testing
        print("   Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
        
        # Create LoRA config
        lora = LoraConfig(
            r=16, lora_alpha=16, lora_dropout=0.05,
            target_modules=["q","k","v","o","wi_0","wi_1","wo"],
            bias="none", task_type="SEQ_2_SEQ_LM"
        )
        model = get_peft_model(model, lora)
        print(f"   ✅ Model loaded with LoRA (trainable params: {model.num_parameters()})")
        
        # Load and test data processing
        def load_jsonl(path, task_filter):
            rows = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    r = json.loads(line)
                    if r.get("task") == task_filter:
                        rows.append(r)
            return Dataset.from_list(rows)
        
        train_data = load_jsonl(train_path, "one_liner") 
        val_data = load_jsonl(val_path, "one_liner")
        print(f"   ✅ Loaded {len(train_data)} train, {len(val_data)} val samples")
        
        # Test tokenization
        def tokenize_fn(batch):
            x = tokenizer(batch["prompt"], truncation=True, max_length=192)
            y = tokenizer(batch["target"], truncation=True, max_length=64)
            x["labels"] = y["input_ids"]
            return x
        
        train_data = train_data.map(tokenize_fn, batched=True, remove_columns=train_data.column_names)
        print(f"   ✅ Tokenization successful")
        
        # Test a quick inference
        sample = train_data[0]
        input_ids = sample["input_ids"][:50]  # truncate for quick test
        
        with model.eval():
            outputs = model.generate(
                input_ids=tokenizer.convert_tokens_to_ids(tokenizer.convert_ids_to_tokens(input_ids)),
                max_new_tokens=10,
                temperature=0.0
            )
        print(f"   ✅ Generation test successful")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_mt5_pipeline(train_path, val_path, temp_dir):
    """Test the report training pipeline without full training."""
    print("🧪 Testing mT5 report training pipeline...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        
        # Load a small model for testing
        print("   Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")
        
        # Create LoRA config
        lora = LoraConfig(
            r=16, lora_alpha=16, lora_dropout=0.05,
            target_modules=["q","k","v","o","wi_0","wi_1","wo"],
            bias="none", task_type="SEQ_2_SEQ_LM"
        )
        model = get_peft_model(model, lora)
        print(f"   ✅ Model loaded with LoRA (trainable params: {model.num_parameters()})")
        
        # Load and test data processing
        def load_jsonl(path, task_filter):
            rows = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    r = json.loads(line)
                    if r.get("task") == task_filter:
                        rows.append(r)
            return Dataset.from_list(rows)
        
        train_data = load_jsonl(train_path, "report") 
        val_data = load_jsonl(val_path, "report")
        print(f"   ✅ Loaded {len(train_data)} train, {len(val_data)} val samples")
        
        # Test tokenization  
        def tokenize_fn(batch):
            x = tokenizer(batch["prompt"], truncation=True, max_length=512)
            y = tokenizer(batch["target"], truncation=True, max_length=192)
            x["labels"] = y["input_ids"]
            return x
        
        train_data = train_data.map(tokenize_fn, batched=True, remove_columns=train_data.column_names)
        print(f"   ✅ Tokenization successful")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Test training pipelines without full training')
    parser.add_argument('--train', default='report_llm/data/train.jsonl',
                       help='Path to training data')
    parser.add_argument('--val', default='report_llm/data/val.jsonl',
                       help='Path to validation data')
    args = parser.parse_args()
    
    print("🚀 Testing Trident-XLM Training Pipelines")
    print("=" * 50)
    
    if not os.path.exists(args.train) or not os.path.exists(args.val):
        print(f"❌ Data files not found: {args.train}, {args.val}")
        return 1
    
    temp_dir = tempfile.mkdtemp()
    print(f"Using temp directory: {temp_dir}")
    
    success = True
    
    # Test both pipelines
    success &= test_flan_pipeline(args.train, args.val, temp_dir)
    success &= test_mt5_pipeline(args.train, args.val, temp_dir)
    
    if success:
        print("\n🎉 All training pipeline tests passed!")
        print("\nNote: These are validation tests only. For actual training:")
        print("   python -m report_llm.train_flan_one_liner --train train.jsonl --val val.jsonl --out exports/flan_model")
        print("   python -m report_llm.train_mt5_report --train train.jsonl --val val.jsonl --out exports/mt5_model")
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())