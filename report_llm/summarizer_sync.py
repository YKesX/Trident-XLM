from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re

# Load a trained/quantized model directory or a base
def load_model(model_dir: str):
    tok = AutoTokenizer.from_pretrained(model_dir)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    return tok, mdl

_BANNED = re.compile(r"\b(ateş|ateşle|nişan|vur|angaje ol|hedefe ateş)\b", re.I)
_ANGLE  = re.compile(r"<[^>]+>")

def _guard_non_operational(text: str):
    if _BANNED.search(text):
        raise ValueError("Operational wording detected; reject.")

def _guard_no_angle_brackets(text: str):
    if _ANGLE.search(text):
        raise ValueError("Model must not output <angle> tokens; reject.")

def make_one_liner(model_dir: str, inputs_text: str) -> str:
    tok, mdl = load_model(model_dir)
    x = tok(inputs_text, return_tensors="pt", truncation=True, max_length=192)
    out = mdl.generate(**x, max_new_tokens=32, do_sample=False, no_repeat_ngram_size=3)
    # For seq2seq, the output doesn't include the input, so decode from token 0
    s = tok.decode(out[0], skip_special_tokens=True).strip()
    _guard_non_operational(s); _guard_no_angle_brackets(s)
    return s
