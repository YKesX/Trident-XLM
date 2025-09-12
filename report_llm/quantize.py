#!/usr/bin/env python
import argparse, torch
from transformers import AutoModelForSeq2SeqLM

def main(args):
    m = AutoModelForSeq2SeqLM.from_pretrained(args.model_in)
    q = torch.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8)
    q.save_pretrained(args.model_out)
    print("Quantized model saved to", args.model_out)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_in", required=True)
    p.add_argument("--model_out", required=True)
    main(p.parse_args())
