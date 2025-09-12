#!/usr/bin/env python
"""
Create a high-quality silver dataset with proper Turkish examples for training.
"""
import json
import random
from typing import List, Dict, Any

def create_silver_examples() -> List[Dict[str, Any]]:
    """Create high-quality Turkish training examples with realistic telemetry data."""
    
    examples = []
    
    # Example 1: High confidence target detection
    examples.append({
        "inputs": {
            "p_hit_calib": 0.87,
            "p_kill_calib": 0.82,
            "p_hit_masked": 0.84,
            "p_kill_masked": 0.79,
            "spoof_risk": 0.08,
            "flags": {
                "mask_applied": True,
                "roi_coverage": 0.92,
                "blur": False,
                "saturation": False,
                "temporal_consistency": True
            },
            "exp": {
                "attn_hotspots": ["motor", "gövde", "kanat"],
                "gradcam_roi": "merkez",
                "shap_top_feats": ["RCS", "termal", "Doppler"]
            },
            "meta": {
                "sensor_mode": "RADAR+IR",
                "approach": "frontal",
                "speed_kph": 120,
                "off_axis_deg": 5,
                "clip_len": 3.1,
                "fps": 30
            },
            "contributions": [
                {
                    "name": "Radar Kesit Alanı",
                    "modality": "RADAR (Ka-band)",
                    "sign": "pos",
                    "value_pct": 42.3,
                    "note": "güçlü metal yansıma"
                },
                {
                    "name": "Motor Termal İmza",
                    "modality": "EO (LWIR)",
                    "sign": "pos",
                    "value_pct": 28.7,
                    "note": "yüksek sıcaklık"
                },
                {
                    "name": "Doppler Kayması",
                    "modality": "RADAR (Ka-band)",
                    "sign": "pos",
                    "value_pct": 15.2,
                    "note": "tutarlı hareket"
                },
                {
                    "name": "Atmosferik Gürültü",
                    "modality": "EO (LWIR)",
                    "sign": "neg",
                    "value_pct": -6.4,
                    "note": "nem faktörü"
                }
            ]
        },
        "targets": {
            "one_liner": "Yüksek güvenilirlik seviyesinde hedef tespit edildi, radar ve termal sinyaller güçlü.",
            "report": "Sistem yüksek güvenilirlikle hedef tespiti gerçekleştirmiştir. Radar kesit alanı ölçümleri güçlü metal yansıma karakteristiği gösterirken, motor termal imzası belirgin sıcaklık profili sunmuştur. Doppler frekans analizi tutarlı hareket verisi sağlamıştır. Atmosferik nem faktörü nedeniyle minimal seviyede gürültü artışı gözlenmiştir ancak bu durum genel tespit kalitesini olumsuz etkilememiştir."
        }
    })
    
    # Example 2: Medium confidence with atmospheric interference
    examples.append({
        "inputs": {
            "p_hit_calib": 0.68,
            "p_kill_calib": 0.61,
            "p_hit_masked": 0.65,
            "p_kill_masked": 0.58,
            "spoof_risk": 0.23,
            "flags": {
                "mask_applied": True,
                "roi_coverage": 0.71,
                "blur": True,
                "saturation": False,
                "temporal_consistency": False
            },
            "exp": {
                "attn_hotspots": ["gövde", "motor"],
                "gradcam_roi": "sağ-üst",
                "shap_top_feats": ["termal", "RCS", "geometri"]
            },
            "meta": {
                "sensor_mode": "IR+RGB",
                "approach": "yan",
                "speed_kph": 85,
                "off_axis_deg": 35,
                "clip_len": 2.8,
                "fps": 25
            },
            "contributions": [
                {
                    "name": "Termal İmza",
                    "modality": "EO (LWIR)",
                    "sign": "pos",
                    "value_pct": 31.5,
                    "note": "kısmi sıcaklık"
                },
                {
                    "name": "Geometrik Özellik",
                    "modality": "EO (RGB)",
                    "sign": "pos",
                    "value_pct": 24.8,
                    "note": "şekil tanıma"
                },
                {
                    "name": "Atmosferik Bulanıklık",
                    "modality": "EO (RGB)",
                    "sign": "neg",
                    "value_pct": -18.2,
                    "note": "görüş mesafesi azalması"
                },
                {
                    "name": "Temporal Tutarsızlık",
                    "modality": "META",
                    "sign": "neg",
                    "value_pct": -12.7,
                    "note": "kareler arası sapma"
                }
            ]
        },
        "targets": {
            "one_liner": "Orta seviye güvenilirlikle tespit, atmosferik koşullar etkili.",
            "report": "Sistem orta seviyede güvenilirlikle hedef tespiti gerçekleştirmiştir. Termal görüntüleme kısmi sıcaklık imzası tespit ederken geometrik özellik analizi şekil tanıma verisi sağlamıştır. Atmosferik bulanıklık görüş mesafesinde azalmaya neden olmuş ve temporal tutarsızlık kareler arası sapma oluşturmuştur. Bu faktörler tespit kalitesini olumsuz etkilemekle birlikte sistem güvenilir sonuç üretmeyi başarmıştır."
        }
    })
    
    # Example 3: Low confidence with spoofing risk
    examples.append({
        "inputs": {
            "p_hit_calib": 0.45,
            "p_kill_calib": 0.38,
            "p_hit_masked": 0.42,
            "p_kill_masked": 0.35,
            "spoof_risk": 0.67,
            "flags": {
                "mask_applied": False,
                "roi_coverage": 0.54,
                "blur": True,
                "saturation": True,
                "temporal_consistency": False
            },
            "exp": {
                "attn_hotspots": ["kenar", "gölge"],
                "gradcam_roi": "sol-alt",
                "shap_top_feats": ["gürültü", "kontrast", "hareket"]
            },
            "meta": {
                "sensor_mode": "RGB",
                "approach": "uzak",
                "speed_kph": 200,
                "off_axis_deg": 78,
                "clip_len": 1.9,
                "fps": 15
            },
            "contributions": [
                {
                    "name": "Hareket Bulanıklığı",
                    "modality": "EO (RGB)",
                    "sign": "neg",
                    "value_pct": -35.4,
                    "note": "yüksek hız etkisi"
                },
                {
                    "name": "Düşük Kontrast",
                    "modality": "EO (RGB)",
                    "sign": "neg",
                    "value_pct": -28.9,
                    "note": "ışık koşulları"
                },
                {
                    "name": "Gürültü Seviyesi",
                    "modality": "EO (RGB)",
                    "sign": "neg",
                    "value_pct": -22.1,
                    "note": "sensör limitasyonu"
                },
                {
                    "name": "Kısmi Geometri",
                    "modality": "EO (RGB)",
                    "sign": "pos",
                    "value_pct": 8.3,
                    "note": "sınırlı şekil verisi"
                }
            ]
        },
        "targets": {
            "one_liner": "Düşük güvenilirlik seviyesi, görüntü kalitesi yetersiz, spoof riski yüksek.",
            "report": "Sistem düşük güvenilirlik seviyesinde tespit gerçekleştirmiştir. Yüksek hız nedeniyle hareket bulanıklığı oluşmuş, ışık koşulları düşük kontrast yaratmış ve sensör limitasyonları gürültü seviyesini artırmıştır. Kısmi geometrik veriler sınırlı şekil bilgisi sağlamıştır. Spoof riski yüksek seviyede tespit edilmiş olup sistemin güvenilirliği sorgulanmalıdır."
        }
    })
    
    # Generate more variations
    base_examples = examples.copy()
    for base in base_examples:
        for _ in range(3):  # Create 3 variations of each base example
            variation = create_variation(base)
            examples.append(variation)
    
    return examples

def create_variation(base_example: Dict[str, Any]) -> Dict[str, Any]:
    """Create a variation of the base example with different values but similar structure."""
    variation = json.loads(json.dumps(base_example))  # Deep copy
    
    # Vary the probabilities slightly
    variation["inputs"]["p_hit_calib"] = max(0.1, min(0.95, base_example["inputs"]["p_hit_calib"] + random.uniform(-0.15, 0.15)))
    variation["inputs"]["p_kill_calib"] = max(0.1, min(0.9, base_example["inputs"]["p_kill_calib"] + random.uniform(-0.15, 0.15)))
    variation["inputs"]["p_hit_masked"] = max(0.1, min(0.95, base_example["inputs"]["p_hit_masked"] + random.uniform(-0.1, 0.1)))
    variation["inputs"]["p_kill_masked"] = max(0.1, min(0.9, base_example["inputs"]["p_kill_masked"] + random.uniform(-0.1, 0.1)))
    variation["inputs"]["spoof_risk"] = max(0.01, min(0.95, base_example["inputs"]["spoof_risk"] + random.uniform(-0.2, 0.2)))
    
    # Vary meta values
    variation["inputs"]["meta"]["speed_kph"] = max(10, min(300, base_example["inputs"]["meta"]["speed_kph"] + random.randint(-30, 30)))
    variation["inputs"]["meta"]["off_axis_deg"] = max(0, min(90, base_example["inputs"]["meta"]["off_axis_deg"] + random.randint(-15, 15)))
    
    # Vary contributions
    for contrib in variation["inputs"]["contributions"]:
        contrib["value_pct"] = contrib["value_pct"] + random.uniform(-5, 5)
    
    # Update targets based on new confidence levels
    confidence_level = variation["inputs"]["p_hit_calib"]
    if confidence_level > 0.8:
        confidence_text = "yüksek güvenilirlik"
        quality_desc = "güçlü"
    elif confidence_level > 0.6:
        confidence_text = "orta seviye güvenilirlik"
        quality_desc = "makul"
    else:
        confidence_text = "düşük güvenilirlik"
        quality_desc = "sınırlı"
    
    # Generate varied one-liner
    one_liner_templates = [
        f"{confidence_text.capitalize()} seviyesinde hedef tespit edildi, sinyaller {quality_desc}.",
        f"Hedef tespiti {confidence_text} ile gerçekleştirildi.",
        f"Sistem {confidence_text} gösterdi, {quality_desc} sinyal kalitesi."
    ]
    variation["targets"]["one_liner"] = random.choice(one_liner_templates)
    
    return variation

def main():
    """Generate the silver dataset."""
    random.seed(42)  # For reproducible results
    
    examples = create_silver_examples()
    
    # Save to silver dataset file
    output_path = "/home/runner/work/Trident-XLM/Trident-XLM/report_llm/trident_report_llm_silver.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    print(f"Created silver dataset with {len(examples)} examples")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    main()