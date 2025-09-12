from .types import TelemetryNLIn, StylePreset, Contribution

def build_inputs_for_llm(t: TelemetryNLIn, style: StylePreset = "resmi") -> str:
    """Simple prompt for small models - focus on core telemetry values only."""
    
    # For small models, use very simple format
    return f"Güvenilirlik: {t.p_hit_calib:.2f}, Etkili: {t.p_kill_calib:.2f}, Spoof: {t.spoof_risk:.2f}. Türkçe açıklama yap:"