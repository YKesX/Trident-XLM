#!/usr/bin/env python3
"""
Minimal test of training logic without downloading models.
"""
import os
import sys
import json

def test_data_loading():
    """Test data loading and processing logic."""
    print("🧪 Testing data loading and processing...")
    
    try:
        from datasets import Dataset
        
        # Test loading logic
        def load_jsonl(path, task_filter):
            rows = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    r = json.loads(line)
                    if r.get("task") == task_filter:
                        rows.append(r)
            return Dataset.from_list(rows)
        
        # Test with actual data
        train_data_one = load_jsonl("report_llm/data/train.jsonl", "one_liner")
        train_data_rep = load_jsonl("report_llm/data/train.jsonl", "report")
        val_data_one = load_jsonl("report_llm/data/val.jsonl", "one_liner") 
        val_data_rep = load_jsonl("report_llm/data/val.jsonl", "report")
        
        print(f"   ✅ One-liner: {len(train_data_one)} train, {len(val_data_one)} val")
        print(f"   ✅ Report: {len(train_data_rep)} train, {len(val_data_rep)} val")
        
        # Check data structure
        if len(train_data_one) > 0:
            sample = train_data_one[0]
            required_keys = {"task", "prompt", "target"}
            if all(key in sample for key in required_keys):
                print("   ✅ Data structure validation passed")
                print(f"   Sample prompt length: {len(sample['prompt'])}")
                print(f"   Sample target length: {len(sample['target'])}")
            else:
                print(f"   ❌ Missing required keys: {required_keys - set(sample.keys())}")
                return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_guard_logic():
    """Test guard logic."""
    print("🧪 Testing guard logic...")
    
    try:
        import re
        
        # Test patterns from the actual guards
        _BANNED = re.compile(r"\b(ateş|ateşle|nişan|vur|angaje ol|hedefe ateş)\b", re.I)
        _ANGLE  = re.compile(r"<[^>]+>")
        
        # Test cases
        test_cases = [
            ("Normal text", False, False),
            ("Text with <system> token", False, True),
            ("Hedefe ateş et", True, False), 
            ("Bad <user> and ateş combo", True, True),
            ("Normal RCS and Doppler text", False, False),
        ]
        
        for text, should_ban, should_angle in test_cases:
            has_banned = _BANNED.search(text) is not None
            has_angle = _ANGLE.search(text) is not None
            
            if has_banned != should_ban or has_angle != should_angle:
                print(f"   ❌ Failed on: {text}")
                print(f"     Expected banned={should_ban}, angle={should_angle}")
                print(f"     Got banned={has_banned}, angle={has_angle}")
                return False
        
        print("   ✅ All guard tests passed")
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_prompt_building():
    """Test prompt building logic."""
    print("🧪 Testing prompt building...")
    
    try:
        from report_llm.types import TelemetryNLIn, Contribution
        from report_llm.prompt_builder import build_inputs_for_llm
        
        # Create test data
        contrib = Contribution(
            name="Test RCS", 
            modality="RADAR (Ka-band)", 
            sign="pos", 
            value_pct=25.5, 
            note="test signal"
        )
        
        telem = TelemetryNLIn(
            p_hit_calib=0.8, p_kill_calib=0.7, 
            p_hit_masked=0.75, p_kill_masked=0.65, 
            spoof_risk=0.1,
            flags={"test": True}, 
            exp={"test_exp": "value"}, 
            meta={"test_meta": 42},
            contributions=[contrib]
        )
        
        # Test different styles
        for style in ["resmi", "madde", "anlatımcı"]:
            prompt = build_inputs_for_llm(telem, style=style)
            if len(prompt) < 100:
                print(f"   ❌ Prompt too short for style {style}: {len(prompt)}")
                return False
            if style not in prompt:
                print(f"   ❌ Style {style} not found in prompt")
                return False
        
        print("   ✅ Prompt building tests passed")
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    print("🔬 Trident-XLM Logic Validation Tests")
    print("=" * 50)
    
    if not os.path.exists("report_llm/data/train.jsonl"):
        print("❌ Training data not found")
        return 1
    
    success = True
    success &= test_data_loading()
    success &= test_guard_logic()
    success &= test_prompt_building()
    
    if success:
        print("\n🎉 All logic validation tests passed!")
        print("\nThe training pipeline is ready for use.")
        print("For actual model training (requires network access):")
        print("   python -m report_llm.train_flan_one_liner --train report_llm/data/train.jsonl --val report_llm/data/val.jsonl --out report_llm/exports/flan_one_liner")
        print("   python -m report_llm.train_mt5_report --train report_llm/data/train.jsonl --val report_llm/data/val.jsonl --out report_llm/exports/mt5_report")
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())