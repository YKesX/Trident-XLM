from .types import TelemetryNLIn, StylePreset, Contribution

def build_inputs_for_llm(t: TelemetryNLIn, style: StylePreset = "resmi") -> str:
    """Build comprehensive Turkish prompt from telemetry data."""
    
    # Basic probabilities
    prompt = f"Kalibre olasılıklar p_hit={t.p_hit_calib:.2f} p_kill={t.p_kill_calib:.2f}. "
    prompt += f"Maskeli değerler p_hit={t.p_hit_masked:.2f} p_kill={t.p_kill_masked:.2f}. "
    prompt += f"Sahtecilik riski {t.spoof_risk:.2f}. "
    
    # Flags
    if t.flags:
        if t.flags.get('mask_applied'):
            prompt += "Maskeleme uygulandı. "
        if 'roi_coverage' in t.flags:
            prompt += f"ROI kapsama {t.flags['roi_coverage']:.2f}. "
        if 'blur' in t.flags:
            prompt += f"Bulanıklık {t.flags['blur']:.2f}. "
    
    # Explainability
    if t.exp:
        if 'attn_hotspots' in t.exp and t.exp['attn_hotspots']:
            hotspots = ", ".join(t.exp['attn_hotspots'])
            prompt += f"Dikkat odakları: {hotspots}. "
        if 'gradcam_roi' in t.exp:
            prompt += f"Grad-CAM bölgesi {t.exp['gradcam_roi']}. "
        if 'shap_top_feats' in t.exp and t.exp['shap_top_feats']:
            feats = ", ".join(t.exp['shap_top_feats'])
            prompt += f"SHAP öznitelikler: {feats}. "
    
    # Contributions
    if t.contributions:
        pos_contribs = [c for c in t.contributions if c.sign == "pos"]
        neg_contribs = [c for c in t.contributions if c.sign == "neg"]
        
        if pos_contribs:
            prompt += "Pozitif katkılar: "
            for c in pos_contribs:
                prompt += f"{c.name} +{c.value_pct:.1f}% ({c.modality}); "
        
        if neg_contribs:
            prompt += "Negatif katkılar: "
            for c in neg_contribs:
                prompt += f"{c.name} {c.value_pct:.1f}% ({c.modality}); "
    
    # Metadata
    if t.meta:
        if 'sensor_mode' in t.meta:
            prompt += f"Sensör modu {t.meta['sensor_mode']}. "
        if 'approach' in t.meta:
            prompt += f"Yaklaşım {t.meta['approach']}. "
        if 'speed_kph' in t.meta:
            prompt += f"Hız {t.meta['speed_kph']:.1f} km/h. "
    
    return prompt.strip()