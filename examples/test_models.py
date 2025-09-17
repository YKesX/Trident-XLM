#!/usr/bin/env python
"""Quick test of trained models"""

import sys, os
sys.path.append(os.path.dirname(__file__))

from report_llm.types import TelemetryNLIn, Contribution
from report_llm.prompt_builder import build_inputs_for_llm
from report_llm.summarizer_sync import load_model

# Create a test sample
test_telemetry = TelemetryNLIn(
    p_hit_calib=0.78,
    p_kill_calib=0.71,
    p_hit_masked=0.75,
    p_kill_masked=0.68,
    spoof_risk=0.18,
    flags={'mask_applied': True, 'roi_coverage': 0.82, 'blur': False},
    exp={'attn_hotspots': ['gövde', 'motor'], 'gradcam_roi': 'merkez'},
    meta={'sensor_mode': 'IR+RADAR', 'approach': 'yan', 'speed_kph': 55},
    contributions=[
        Contribution(name="Motor Termal", modality="EO (LWIR)", sign="pos", value_pct=28.5, note="güçlü ısı"),
        Contribution(name="Gövde RCS", modality="RADAR", sign="pos", value_pct=24.2, note="iyi yansıma"),
        Contribution(name="Atmospheric Fade", modality="META", sign="neg", value_pct=-12.8, note="nem etkisi"),
    ]
)

def main():
    print("Building prompt...")
    prompt = build_inputs_for_llm(test_telemetry, style="resmi")
    print("Full prompt:")
    print(prompt)
    print("\n" + "="*50 + "\n")
    
    print("Testing one-liner model...")
    try:
        tok, mdl = load_model("report_llm/exports/flan_one_liner_int8")
        x = tok(prompt, return_tensors="pt", truncation=True, max_length=192)
        print(f"Input tokens shape: {x['input_ids'].shape}")
        out = mdl.generate(**x, max_new_tokens=32, do_sample=False, no_repeat_ngram_size=3)
        print(f"Output tokens shape: {out.shape}")
        s = tok.decode(out[0], skip_special_tokens=True).strip()
        print(f"Raw output: '{s}'")
        
        # Check if output contains angle brackets
        if "<" in s:
            print("⚠️  Output contains angle brackets")
        else:
            print("✓ No angle brackets detected")
            
    except Exception as e:
        print(f"One-liner error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()