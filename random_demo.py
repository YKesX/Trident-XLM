#!/usr/bin/env python3
"""
Randomized telemetry demo for Trident-XLM.
- Generates plausible random TelemetryNLIn inputs (Turkish-friendly)
- Builds prompts with the prompt builder
- Tries to run both one-liner (Flan-T5) and report (mT5)
- Falls back to tiny templated Turkish text if models are missing
"""
import os, random, json
import sys, types
import importlib.machinery
from dataclasses import asdict
from typing import List
from report_llm.types import TelemetryNLIn, Contribution
from report_llm.prompt_builder import build_inputs_for_llm

# Optional model imports guarded
def try_models():
    try:
        from report_llm.summarizer_sync import make_one_liner
        from report_llm.summarizer_async import make_report
        return make_one_liner, make_report
    except Exception:
        return None, None

# Harden environment: force Transformers into torch-only paths and stub TensorFlow
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("USE_TORCH", "1")
# Stub tensorflow to bypass incompatible NumPy/TensorFlow imports pulled by optional modules
if "tensorflow" not in sys.modules:
    _tf = types.ModuleType("tensorflow")
    _tf.__spec__ = importlib.machinery.ModuleSpec("tensorflow", loader=None)
    sys.modules["tensorflow"] = _tf
if "ml_dtypes" not in sys.modules:
    _mld = types.ModuleType("ml_dtypes")
    _mld.__spec__ = importlib.machinery.ModuleSpec("ml_dtypes", loader=None)
    sys.modules["ml_dtypes"] = _mld

SENSOR_MODES = ["RGB", "IR", "RGB+IR", "RADAR+EO"]
APPROACH = ["yaklaşan", "uzaklaşan", "sabit", "manzara"]
ATTN_SPOTS = ["burun", "gövde", "orta-bölüm", "arka", "sol-kanat", "sağ-kanat", "iz"]

POS_CONTR = [
    ("Radar Kesit Alanı (RCS)", "RADAR (Ka-band)", 15, 45, "güçlü yansıma"),
    ("Termal İmza Analizi", "EO (LWIR)", 10, 40, "yüksek sıcaklık stabil"),
    ("Doppler Tehdit", "RADAR (Doppler)", 8, 30, "yaklaşma vektörü tutarlı"),
]
NEG_CONTR = [
    ("SpoofShield Düzeltmesi", "Guard", -2, -0.3, "tutarlı sensörler"),
    ("Atmosferik Parıldama", "EO", -3, -0.5, "uzun menzil etkisi"),
    ("Parazit/Karıştırma", "RADAR", -3, -0.7, "düşük arka plan"),
]

def rand_contributions() -> List[Contribution]:
    items: List[Contribution] = []
    for name, mod, lo, hi, note in POS_CONTR:
        if random.random() < 0.8:
            val = round(random.uniform(lo, hi), 2)
            items.append(Contribution(name, mod, "pos", val, note))
    for name, mod, lo, hi, note in NEG_CONTR:
        if random.random() < 0.7:
            val = round(random.uniform(hi, lo), 2)  # hi is negative
            items.append(Contribution(name, mod, "neg", val, note))
    return items

def rand_telem() -> TelemetryNLIn:
    p_hit_calib = round(random.uniform(0.6, 0.99), 2)
    p_kill_calib = round(random.uniform(0.5, p_hit_calib), 2)
    p_hit_masked = max(0.0, min(1.0, round(p_hit_calib - random.uniform(0.0, 0.05), 2)))
    p_kill_masked = max(0.0, min(1.0, round(p_kill_calib - random.uniform(0.0, 0.05), 2)))
    spoof_risk = round(random.uniform(0.0, 0.2), 2)
    flags = {
        "mask_applied": random.random() < 0.5,
        "roi_coverage": round(random.uniform(0.6, 0.95), 2),
        "blur": round(random.uniform(0.0, 0.2), 2),
    }
    exp = {
        "attn_hotspots": random.sample(ATTN_SPOTS, k=random.randint(1, 3)),
        "gradcam_roi": random.choice(["merkez", "sol", "sağ", "üst", "alt"]),
        "shap_top_feats": random.sample(["closing_rate", "RCS_db", "LWIR_peak", "range_delta"], k=2),
    }
    meta = {
        "sensor_mode": random.choice(SENSOR_MODES),
        "approach": random.choice(APPROACH),
        "speed_kph": round(random.uniform(10, 60), 1),
    }
    return TelemetryNLIn(
        p_hit_calib=p_hit_calib,
        p_kill_calib=p_kill_calib,
        p_hit_masked=p_hit_masked,
        p_kill_masked=p_kill_masked,
        spoof_risk=spoof_risk,
        flags=flags,
        exp=exp,
        meta=meta,
        contributions=rand_contributions(),
    )

# Fallback tiny Turkish template if models unavailable
def fallback_one_liner(t: TelemetryNLIn) -> str:
    return (
        f"Kalibre güven skorları (vuruş={t.p_hit_calib:.2f}, imha={t.p_kill_calib:.2f}); "
        f"maske sonrası (vuruş={t.p_hit_masked:.2f}, imha={t.p_kill_masked:.2f}). "
        f"Sahtecilik riski {t.spoof_risk:.2f}."
    )

def fallback_report(t: TelemetryNLIn) -> str:
    pos = [c for c in t.contributions if c.sign == "pos"]
    neg = [c for c in t.contributions if c.sign == "neg"]
    s = [
        "3.0. Birincil Gerekçe",
        "Sistem, çoklu sensör kaynaklarından gelen tutarlı sinyalleri değerlendirerek yüksek güven seviyesine ulaşmıştır.",
        "4.0. Karara Etki Eden Faktörler",
        "Pozitif katkılar: " + ", ".join(f"{c.name} (+{c.value_pct:.2f}%, {c.modality})" for c in pos) if pos else "Pozitif katkı yok.",
        "Negatif katkılar: " + ", ".join(f"{c.name} ({c.value_pct:.2f}%, {c.modality})" for c in neg) if neg else "Negatif katkı yok.",
    ]
    return "\n".join(s)

def _poor(txt: str) -> bool:
    if not txt:
        return True
    t = txt.strip()
    return (len(t) < 40) or (t.count(" ") < 5)

def main():
    random.seed(7)
    telem = rand_telem()
    prompt = build_inputs_for_llm(telem, style="resmi")

    # Try exports if present; else fall back to env or quick folders
    flan_default = "report_llm/exports/flan_one_liner" if os.path.exists("report_llm/exports/flan_one_liner") else "flan_quick_int8"
    mt5_default = "report_llm/exports/mt5_report" if os.path.exists("report_llm/exports/mt5_report") else "flan_quick_int8"
    flan_path = os.environ.get("FLAN_MODEL", flan_default)
    mt5_path = os.environ.get("MT5_MODEL", mt5_default)

    make_one_liner, make_report = try_models()

    print("🇹🇷 RANDOMIZED TRIDENT-XLM DEMO")
    print("="*60)
    print("Prompt (ilk 240):\n" + prompt[:240] + "...\n")

    # One-liner
    one = None
    if make_one_liner:
        try:
            one = make_one_liner(flan_path, prompt)
        except Exception as e:
            print(f"⚠️  One-liner modeli kullanılamadı: {e}")
    if (not one) or _poor(one):
        one = fallback_one_liner(telem)
    print("ONE-LINER:\n" + one + "\n")

    # Report
    rep = None
    if make_report:
        try:
            rep = make_report(mt5_path, prompt, max_length=160)
        except Exception as e:
            print(f"⚠️  Rapor modeli kullanılamadı: {e}")
    if (not rep) or _poor(rep):
        rep = fallback_report(telem)
    print("RAPOR:\n" + rep)

    # Save JSON result
    out = {
    "inputs": asdict(telem),
        "prompt": prompt,
        "outputs": {"one_liner": one, "report": rep},
    }
    os.makedirs("outputs", exist_ok=True)
    with open(os.path.join("outputs", "random_demo_output.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n✅ Kaydedildi: outputs/random_demo_output.json")

if __name__ == "__main__":
    main()
