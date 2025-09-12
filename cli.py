#!/usr/bin/env python3
"""
Command-line interface for Trident-XLM explainability LLM.
Provides easy access to training, inference, and testing functionality.
"""
import os
import sys
import json
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from report_llm.types import TelemetryNLIn, Contribution
from report_llm.prompt_builder import build_inputs_for_llm

def load_telemetry_sample(telemetry_file: str):
    """Load a telemetry sample from JSON/JSONL file."""
    with open(telemetry_file, 'r', encoding='utf-8') as f:
        if telemetry_file.endswith('.jsonl'):
            # Take first line of JSONL
            line = f.readline().strip()
            if line:
                data = json.loads(line)
        else:
            # Assume JSON
            data = json.load(f)
    
    # Extract inputs
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

def cmd_generate_prompt(args):
    """Generate prompt from telemetry data."""
    print(f"📝 Generating prompt from: {args.telemetry}")
    
    telem, targets = load_telemetry_sample(args.telemetry)
    prompt = build_inputs_for_llm(telem, style=args.style)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(prompt)
        print(f"✅ Prompt saved to: {args.output}")
    else:
        print("\n" + "="*50)
        print(prompt)
        print("="*50)
    
    if targets:
        print(f"\n📋 Expected targets available:")
        for key, value in targets.items():
            print(f"  {key}: {value[:100]}...")

def cmd_inference(args):
    """Run inference with trained models."""
    print(f"🤖 Running inference...")
    
    if not os.path.exists(args.model_sync) or not os.path.exists(args.model_async):
        print(f"❌ Model directories not found:")
        print(f"   Sync: {args.model_sync}")
        print(f"   Async: {args.model_async}")
        print(f"   Train models first with:")
        print(f"   python -m report_llm.train_flan_one_liner --train data/train.jsonl --val data/val.jsonl --out {args.model_sync}")
        print(f"   python -m report_llm.train_mt5_report --train data/train.jsonl --val data/val.jsonl --out {args.model_async}")
        return
    
    try:
        from report_llm.summarizer_sync import make_one_liner
        from report_llm.summarizer_async import make_report
        
        telem, targets = load_telemetry_sample(args.telemetry)
        prompt = build_inputs_for_llm(telem, style=args.style)
        
        print("🔄 Generating one-liner...")
        one_liner = make_one_liner(args.model_sync, prompt)
        print(f"✅ One-liner: {one_liner}")
        
        print("🔄 Generating report...")
        report = make_report(args.model_async, prompt)
        print(f"✅ Report: {report}")
        
        if args.output:
            result = {
                "input_telemetry": telem.__dict__,
                "prompt": prompt,
                "outputs": {
                    "one_liner": one_liner,
                    "report": report
                }
            }
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"✅ Results saved to: {args.output}")
            
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Install with: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Inference failed: {e}")

def cmd_train(args):
    """Run training pipeline."""
    print(f"🏋️ Training models...")
    
    if not os.path.exists(args.train) or not os.path.exists(args.val):
        print(f"❌ Training data not found: {args.train}, {args.val}")
        print("Generate data first with:")
        print("python -m report_llm.build_dataset --telemetry data/telemetry.jsonl --out_dir data/")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    flan_out = os.path.join(args.output_dir, "flan_one_liner")
    mt5_out = os.path.join(args.output_dir, "mt5_report")
    
    print(f"📚 Training one-liner model: {flan_out}")
    os.system(f"python -m report_llm.train_flan_one_liner --train {args.train} --val {args.val} --out {flan_out}")
    
    print(f"📚 Training report model: {mt5_out}")
    os.system(f"python -m report_llm.train_mt5_report --train {args.train} --val {args.val} --out {mt5_out}")
    
    if args.quantize:
        print("🔧 Quantizing models...")
        flan_q8 = f"{flan_out}_int8"
        mt5_q8 = f"{mt5_out}_int8"
        
        os.system(f"python -m report_llm.quantize --model_in {flan_out} --model_out {flan_q8}")
        os.system(f"python -m report_llm.quantize --model_in {mt5_out} --model_out {mt5_q8}")
        
        print(f"✅ Quantized models ready: {flan_q8}, {mt5_q8}")

def cmd_test(args):
    """Run tests."""
    print("🧪 Running tests...")
    
    # Run the comprehensive test suite
    test_file = os.path.join(os.path.dirname(__file__), "report_llm", "tests", "test_comprehensive.py")
    if os.path.exists(test_file):
        os.system(f"python {test_file}")
    else:
        print("❌ Test file not found")
    
    # Run basic functionality tests
    if os.path.exists("test_core.py"):
        print("\n🔧 Running core functionality test...")
        os.system("python test_core.py")
    
    if os.path.exists("test_logic.py"):
        print("\n🔬 Running logic validation test...")
        os.system("python test_logic.py")

def main():
    parser = argparse.ArgumentParser(
        description="Trident-XLM: Turkish explainability LLM for TRIDENT-Net",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate prompt from telemetry
  python cli.py prompt --telemetry data/telemetry.jsonl --style resmi
  
  # Train models
  python cli.py train --train data/train.jsonl --val data/val.jsonl --output-dir exports/ --quantize
  
  # Run inference
  python cli.py inference --telemetry data/telemetry.jsonl --model-sync exports/flan_one_liner_int8 --model-async exports/mt5_report_int8
  
  # Run tests
  python cli.py test
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Prompt generation command
    prompt_parser = subparsers.add_parser('prompt', help='Generate prompt from telemetry')
    prompt_parser.add_argument('--telemetry', required=True, help='Telemetry JSON/JSONL file')
    prompt_parser.add_argument('--style', default='resmi', choices=['resmi', 'madde', 'anlatımcı'], help='Style preset')
    prompt_parser.add_argument('--output', help='Output file for prompt')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('--train', default='report_llm/data/train.jsonl', help='Training data')
    train_parser.add_argument('--val', default='report_llm/data/val.jsonl', help='Validation data')
    train_parser.add_argument('--output-dir', default='report_llm/exports', help='Output directory')
    train_parser.add_argument('--quantize', action='store_true', help='Quantize models after training')
    
    # Inference command
    inference_parser = subparsers.add_parser('inference', help='Run inference')
    inference_parser.add_argument('--telemetry', required=True, help='Telemetry JSON/JSONL file')
    inference_parser.add_argument('--model-sync', default='report_llm/exports/flan_one_liner_int8', help='One-liner model path')
    inference_parser.add_argument('--model-async', default='report_llm/exports/mt5_report_int8', help='Report model path')
    inference_parser.add_argument('--style', default='resmi', choices=['resmi', 'madde', 'anlatımcı'], help='Style preset')
    inference_parser.add_argument('--output', help='Output JSON file for results')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run tests')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Route to appropriate command
    if args.command == 'prompt':
        cmd_generate_prompt(args)
    elif args.command == 'train':
        cmd_train(args)
    elif args.command == 'inference':
        cmd_inference(args)
    elif args.command == 'test':
        cmd_test(args)

if __name__ == '__main__':
    main()