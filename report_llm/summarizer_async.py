import os, re
import asyncio
from typing import Optional

# Load a trained/quantized model directory or a base
def load_model(model_dir: str):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    try:
        tok = AutoTokenizer.from_pretrained(model_dir)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        return tok, mdl
    except Exception:
        adapter_files = [
            os.path.join(model_dir, "adapter_config.json"),
            os.path.join(model_dir, "adapter_model.safetensors"),
        ]
        if any(os.path.exists(p) for p in adapter_files):
            from peft import PeftModel
            base_candidates = [
                os.environ.get("BASE_MODEL", ""),
                "google/mt5-small",
                "google/flan-t5-small",
            ]
            last_err = None
            for base in base_candidates:
                if not base:
                    continue
                try:
                    base_tok = AutoTokenizer.from_pretrained(model_dir if os.path.exists(os.path.join(model_dir, "tokenizer.json")) else base)
                    base_mdl = AutoModelForSeq2SeqLM.from_pretrained(base)
                    mdl = PeftModel.from_pretrained(base_mdl, model_dir)
                    return base_tok, mdl
                except Exception as e:
                    last_err = e
            raise last_err or ValueError("Failed to load adapter with any base model candidate.")
        raise

_BANNED = re.compile(r"\b(ateş|ateşle|nişan|vur|angaje ol|hedefe ateş)\b", re.I)
_ANGLE  = re.compile(r"<[^>]+>")

def _guard_non_operational(text: str):
    if _BANNED.search(text):
        raise ValueError("Operational wording detected; reject.")

def _guard_no_angle_brackets(text: str):
    if _ANGLE.search(text):
        raise ValueError("Model must not output <angle> tokens; reject.")

def make_report(model_dir: str, inputs_text: str, max_length: int = 192) -> str:
    """Generate a longer report text using beam search for better quality."""
    tok, mdl = load_model(model_dir)
    x = tok(inputs_text, return_tensors="pt", truncation=True, max_length=512)
    out = mdl.generate(
        **x, 
        max_new_tokens=max_length, 
        num_beams=3,
        do_sample=False,
        no_repeat_ngram_size=3,
        early_stopping=True
    )
    # For seq2seq, decode only the new tokens
    input_length = x['input_ids'].shape[1] if hasattr(x['input_ids'], 'shape') else len(x['input_ids'][0])
    generated_tokens = out[0][input_length:] if out[0].shape[0] > input_length else out[0]
    s = tok.decode(generated_tokens, skip_special_tokens=True).strip()
    # Fallback if empty or bad output
    if not s or len(s) < 10:
        s = tok.decode(out[0], skip_special_tokens=True).strip()
    # Clean up any remaining issues
    s = s.replace('<pad>', '').replace('<unk>', '').replace('<extra_id_0>', '').replace('<extra_id_1>', '').strip()
    _guard_non_operational(s)
    _guard_no_angle_brackets(s)
    return s

async def make_report_async(model_dir: str, inputs_text: str, max_length: int = 192) -> str:
    """Asynchronous version of report generation for non-blocking operation."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, make_report, model_dir, inputs_text, max_length)