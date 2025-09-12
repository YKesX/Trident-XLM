from .types import TelemetryNLIn, StylePreset, Contribution

def build_inputs_for_llm(t: TelemetryNLIn, style: StylePreset = "resmi") -> str:
    """Ultra-simple prompt - just numbers to Turkish."""
    
    # Ultra-minimal format 
    return f"{t.p_hit_calib:.2f} {t.p_kill_calib:.2f} {t.spoof_risk:.2f}"