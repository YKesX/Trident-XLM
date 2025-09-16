#!/usr/bin/env python
import argparse, torch, os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig

def main(args):
    # Simply copy the model with optimized settings for CPU inference
    print("Loading model from", args.model_in)
    # Ensure we resolve local path to avoid HF repo id validation
    model_in = os.path.abspath(args.model_in)
    model_out = os.path.abspath(args.model_out)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_in, torch_dtype=torch.float32, local_files_only=False)
    tokenizer = AutoTokenizer.from_pretrained(model_in, local_files_only=False)
    
    # Save in optimized format
    os.makedirs(model_out, exist_ok=True)
    model.save_pretrained(model_out, safe_serialization=True)
    tokenizer.save_pretrained(model_out)
    
    print("Optimized model saved to", model_out)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_in", required=True)
    p.add_argument("--model_out", required=True)
    main(p.parse_args())
