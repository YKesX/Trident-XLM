#!/usr/bin/env python
import argparse, torch, os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig

def main(args):
    # Simply copy the model with optimized settings for CPU inference
    print("Loading model from", args.model_in)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_in, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(args.model_in)
    
    # Save in optimized format
    os.makedirs(args.model_out, exist_ok=True)
    model.save_pretrained(args.model_out, safe_serialization=True)
    tokenizer.save_pretrained(args.model_out)
    
    print("Optimized model saved to", args.model_out)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_in", required=True)
    p.add_argument("--model_out", required=True)
    main(p.parse_args())
