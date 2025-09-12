#!/usr/bin/env python
"""
Create simple, focused training data for better Turkish model outputs.
"""
import json
import random
from typing import List, Dict, Any

def create_simple_training_examples() -> List[Dict[str, Any]]:
    """Create simple Turkish training examples with short prompts and clear targets."""
    
    examples = []
    
    # Simple training patterns
    patterns = [
        # High confidence examples
        {
            "p_hit": 0.85, "p_kill": 0.80, "spoof": 0.10,
            "one_liner": "Yüksek güvenilirlik seviyesinde hedef tespit edildi.",
            "report": "Sistem yüksek güvenilirlikle hedef tespiti gerçekleştirmiştir. Radar ve termal sinyaller güçlü karakteristik göstermiştir. Tespit kalitesi mükemmel seviyededir."
        },
        {
            "p_hit": 0.88, "p_kill": 0.83, "spoof": 0.08,
            "one_liner": "Çok güçlü hedef tespiti, sinyaller net.",
            "report": "Tespit sistemi çok güçlü performans sergilemiştir. Radar kesit alanı ve termal imza belirgin şekilde tespit edilmiştir. Güvenilirlik çok yüksek seviyededir."
        },
        {
            "p_hit": 0.82, "p_kill": 0.78, "spoof": 0.12,
            "one_liner": "İyi seviyede tespit başarısı kaydedildi.",
            "report": "Hedef tespiti iyi seviyede gerçekleştirilmiştir. Sensor verileri tatmin edici kalitede olup sistem güvenilir sonuç üretmiştir."
        },
        
        # Medium confidence examples  
        {
            "p_hit": 0.65, "p_kill": 0.58, "spoof": 0.25,
            "one_liner": "Orta seviye güvenilirlik, koşullar zor.",
            "report": "Sistem orta seviyede güvenilirlikle tespit gerçekleştirmiştir. Atmosferik koşullar ve gürültü faktörleri etkili olmuştur ancak makul sonuç elde edilmiştir."
        },
        {
            "p_hit": 0.68, "p_kill": 0.62, "spoof": 0.30,
            "one_liner": "Makul tespit kalitesi, sınırlı veriler.",
            "report": "Tespit kalitesi makul seviyededir. Sensor verilerinde kısıtlılıklar bulunmakta ancak sistem güvenilir analiz sağlamıştır."
        },
        {
            "p_hit": 0.70, "p_kill": 0.64, "spoof": 0.22,
            "one_liner": "Yeterli güvenilirlik, kabul edilebilir.",
            "report": "Hedef tespiti yeterli güvenilirlik seviyesinde tamamlanmıştır. Genel performans kabul edilebilir düzeydedir."
        },
        
        # Low confidence examples
        {
            "p_hit": 0.45, "p_kill": 0.38, "spoof": 0.55,
            "one_liner": "Düşük güvenilirlik, zor koşullar.",
            "report": "Sistem düşük güvenilirlik seviyesinde tespit gerçekleştirmiştir. Zorlu çevre koşulları ve yüksek gürültü seviyesi performansı olumsuz etkilemiştir."
        },
        {
            "p_hit": 0.42, "p_kill": 0.35, "spoof": 0.62,
            "one_liner": "Zayıf sinyal kalitesi, belirsizlik var.",
            "report": "Sinyal kalitesi zayıf olup belirsizlik düzeyi yüksektir. Tespit güvenilirliği sınırlıdır ve ek doğrulama gerekebilir."
        },
        {
            "p_hit": 0.38, "p_kill": 0.32, "spoof": 0.68,
            "one_liner": "Tespit zorluğu, spoof riski yüksek.",
            "report": "Hedef tespitinde zorluk yaşanmıştır. Yüksek spoof riski tespit edilmiş olup sonuçların dikkatli değerlendirilmesi gerekmektedir."
        }
    ]
    
    # Generate training examples from patterns
    for pattern in patterns:
        # Create simple prompt format
        prompt = f"Tespit güvenilirliği: {pattern['p_hit']:.2f}, Etkili güvenilirlik: {pattern['p_kill']:.2f}, Spoof riski: {pattern['spoof']:.2f}."
        
        # Add one-liner task
        examples.append({
            "inputs": {
                "p_hit_calib": pattern["p_hit"],
                "p_kill_calib": pattern["p_kill"], 
                "p_hit_masked": pattern["p_hit"] - 0.02,
                "p_kill_masked": pattern["p_kill"] - 0.02,
                "spoof_risk": pattern["spoof"],
                "flags": {"mask_applied": True},
                "exp": {"attn_hotspots": ["hedef"]},
                "meta": {"sensor_mode": "RADAR"},
                "contributions": []
            },
            "targets": {
                "one_liner": pattern["one_liner"],
                "report": pattern["report"]
            }
        })
        
        # Create variations
        for i in range(2):
            var_p_hit = max(0.1, min(0.95, pattern["p_hit"] + random.uniform(-0.1, 0.1)))
            var_p_kill = max(0.1, min(0.9, pattern["p_kill"] + random.uniform(-0.1, 0.1)))
            var_spoof = max(0.01, min(0.95, pattern["spoof"] + random.uniform(-0.15, 0.15)))
            
            # Adjust targets based on confidence
            if var_p_hit > 0.8:
                one_liner = random.choice([
                    "Yüksek güvenilirlikle tespit edildi.",
                    "Güçlü sinyal kalitesi, net tespit.",
                    "Mükemmel tespit performansı."
                ])
                report = "Sistem yüksek performans göstermiştir. Tespit kalitesi mükemmel seviyededir ve güvenilirlik çok yüksektir."
            elif var_p_hit > 0.6:
                one_liner = random.choice([
                    "Orta seviye güvenilirlik sağlandı.",
                    "Kabul edilebilir tespit kalitesi.",
                    "Yeterli güvenilirlik seviyesi."
                ])
                report = "Tespit kalitesi orta seviyededir. Sistem makul güvenilirlikle sonuç üretmiştir."
            else:
                one_liner = random.choice([
                    "Düşük güvenilirlik, zor koşullar.",
                    "Sınırlı tespit kalitesi.",
                    "Belirsizlik seviyesi yüksek."
                ])
                report = "Tespit güvenilirliği sınırlıdır. Zorlu koşullar ve yüksek belirsizlik seviyesi gözlemlenmiştir."
            
            examples.append({
                "inputs": {
                    "p_hit_calib": var_p_hit,
                    "p_kill_calib": var_p_kill,
                    "p_hit_masked": var_p_hit - 0.02,
                    "p_kill_masked": var_p_kill - 0.02, 
                    "spoof_risk": var_spoof,
                    "flags": {"mask_applied": True},
                    "exp": {"attn_hotspots": ["hedef"]},
                    "meta": {"sensor_mode": "RADAR"},
                    "contributions": []
                },
                "targets": {
                    "one_liner": one_liner,
                    "report": report
                }
            })
    
    return examples

def main():
    """Generate the simple training dataset."""
    random.seed(42)
    
    examples = create_simple_training_examples()
    
    # Save to dataset file
    output_path = "/home/runner/work/Trident-XLM/Trident-XLM/report_llm/data/telemetry.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    print(f"Created simple training dataset with {len(examples)} examples")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    main()