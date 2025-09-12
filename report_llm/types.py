from dataclasses import dataclass
from typing import List, Literal, Dict, Any

StylePreset = Literal["resmi", "madde", "anlatımcı"]

@dataclass
class Contribution:
    """A single scored contribution item to overall confidence.
    value_pct is a signed percentage *without* the % symbol (e.g., +39.40 or -0.75).
    sign is derived from the sign of value_pct but kept for clarity.
    """
    name: str                 # e.g., "Radar Kesit Alanı (RCS)"
    modality: str             # e.g., "RADAR (Ka-band)" or "EO (LWIR)"
    sign: Literal["pos","neg"]
    value_pct: float          # +39.40 or -0.75 (no % symbol)
    note: str                 # short rationale

@dataclass
class TelemetryNLIn:
    """Telemetry and explainability summary provided by the TRIDENT pipeline.
    This is the ONLY input used to condition the language models.
    No angle-bracket placeholders or wrapper-level fields appear here.
    """
    # Core probabilities (already calibrated/masked by the pipeline)
    p_hit_calib: float
    p_kill_calib: float
    p_hit_masked: float
    p_kill_masked: float
    spoof_risk: float

    # Guard & quality flags (free-form, but JSON-serializable)
    flags: Dict[str, Any]     # e.g., {"mask_applied": True, "roi_coverage": 0.83, ...}

    # Explainability digests
    exp: Dict[str, Any]       # e.g., {"attn_hotspots":["nose","mid-body"], "gradcam_roi":"center-left", ...}

    # Kinematics / meta summaries safe for NL use
    meta: Dict[str, Any]      # e.g., {"sensor_mode":"RGB+IR","approach":"inbound","speed_kph":45}

    # Decomposed score contributions
    contributions: List[Contribution]  # both positive and negative with sign
