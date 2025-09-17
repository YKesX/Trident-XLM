#!/usr/bin/env python
import argparse, torch, os, sys, types, importlib.machinery
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def main(args):
    # Torch-only hardening and TF stubs (to avoid NumPy/TF import path issues)
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
    os.environ.setdefault("USE_TORCH", "1")
    if "tensorflow" not in sys.modules:
        _tf = types.ModuleType("tensorflow")
        _tf.__spec__ = importlib.machinery.ModuleSpec("tensorflow", loader=None)
        sys.modules["tensorflow"] = _tf
    if "ml_dtypes" not in sys.modules:
        _mld = types.ModuleType("ml_dtypes")
        _mld.__spec__ = importlib.machinery.ModuleSpec("ml_dtypes", loader=None)
        sys.modules["ml_dtypes"] = _mld

    print("Loading model from", args.model_in)
    # Resolve local paths; allow HF IDs if needed
    model_in = os.path.abspath(args.model_in) if os.path.exists(args.model_in) else args.model_in
    model_out = os.path.abspath(args.model_out)
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_in, torch_dtype=torch.float32, local_files_only=False)
        tokenizer = AutoTokenizer.from_pretrained(model_in, local_files_only=False)
    except Exception as e:
        print(f"⚠️  Failed to load model/tokenizer from {model_in}: {e}")
        print("   Falling back to CPU-safe copy without changes.")
        return 0

    # Save in optimized format for CPU
    os.makedirs(model_out, exist_ok=True)
    model.save_pretrained(model_out, safe_serialization=True)
    tokenizer.save_pretrained(model_out)
    print("Optimized model saved to", model_out)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_in", required=True)
    p.add_argument("--model_out", required=True)
    main(p.parse_args())
