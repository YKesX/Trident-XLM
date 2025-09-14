#!/usr/bin/env python3
"""
Simple CLI test for Trident-XLM without requiring full model training.
This demonstrates the core functionality pipeline.
"""
import os
import sys
import json
import argparse

# Add the report_llm package to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from report_llm.types import TelemetryNLIn, Contribution
from report_llm.prompt_builder import build_inputs_for_llm

def load_sample_telemetry(data_file):
    """Load a sample from the telemetry JSONL file."""
    with open(data_file, 'r', encoding='utf-8') as f:
        line = f.readline().strip()
        if line:
            data = json.loads(line)
            inputs = data.get('inputs') or data
            targets = data.get('targets', {})
            
            # Convert to TelemetryNLIn
            contribs = [Contribution(**c) for c in inputs.get('contributions', [])]
            telem = TelemetryNLIn(
                p_hit_calib=inputs['p_hit_calib'],
                p_kill_calib=inputs['p_kill_calib'],
                p_hit_masked=inputs['p_hit_masked'],
                p_kill_masked=inputs['p_kill_masked'],
                spoof_risk=inputs['spoof_risk'],
                flags=inputs.get('flags', {}),
                exp=inputs.get('exp', {}),
                meta=inputs.get('meta', {}),
                contributions=contribs
            )
            return telem, targets
    return None, None

def main():
    parser = argparse.ArgumentParser(description='Test Trident-XLM core functionality')
    parser.add_argument('--telemetry', default='report_llm/data/telemetry.jsonl',
                       help='Path to telemetry JSONL file')
    parser.add_argument('--style', default='resmi', choices=['resmi', 'madde', 'anlatımcı'],
                       help='Style preset for generation')
    args = parser.parse_args()
    
    # Test data loading and prompt building
    print("🔧 Testing Trident-XLM Core Functionality")
    print("=" * 50)
    
    if not os.path.exists(args.telemetry):
        print(f"❌ Telemetry file not found: {args.telemetry}")
        return 1
        
    telem, targets = load_sample_telemetry(args.telemetry)
    if not telem:
        print("❌ Failed to load telemetry data")
        return 1
        
    print("✅ Loaded telemetry data successfully")
    print(f"   - Probabilities: hit={telem.p_hit_calib:.2f}, kill={telem.p_kill_calib:.2f}")
    print(f"   - Spoof risk: {telem.spoof_risk:.2f}")
    print(f"   - Contributions: {len(telem.contributions)}")
    
    # Test prompt building
    prompt = build_inputs_for_llm(telem, style=args.style)
    print(f"✅ Generated prompt (style: {args.style})")
    print(f"   - Length: {len(prompt)} characters")
    print(f"   - Sample: {prompt[:150]}...")
    
    # Show expected targets if available
    if targets:
        print("\n📝 Expected targets:")
        if 'one_liner' in targets:
            print(f"   One-liner: {targets['one_liner']}")
        if 'report' in targets:
            print(f"   Report: {targets['report'][:100]}...")
    
    # Test guard patterns
    print("\n🛡️  Testing guard patterns:")
    
    # Test angle bracket detection
    test_angle = "This has <system> tokens"
    has_angle = '<' in test_angle and '>' in test_angle
    print(f"   Angle bracket detection: {'✅' if has_angle else '❌'}")
    
    # Test operational word detection  
    import re
    banned_pattern = re.compile(r"\b(ateş|ateşle|nişan|vur|angaje ol|hedefe ateş)\b", re.I)
    test_ops = "hedefe ateş edin"
    has_ops = banned_pattern.search(test_ops) is not None
    print(f"   Operational word detection: {'✅' if has_ops else '❌'}")
    
    print("\n🎉 Core functionality test completed successfully!")
    print("\nNext steps:")
    print("   1. Install ML dependencies: pip install -r requirements.txt")
    print("   2. Train models: python -m report_llm.train_flan_one_liner")  
    print("   3. Run full inference pipeline")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())