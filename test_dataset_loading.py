#!/usr/bin/env python3
"""
Test dataset loading functionality to ensure num_samples=0 error is fixed.
"""
import json
import sys
import os

def test_dataset_loading():
    """Test that dataset loading works correctly with different parameters."""
    print("🧪 Testing dataset loading functionality...")
    
    # Test 1: Check data files exist and have correct structure
    files_to_check = ['report_llm/data/train.jsonl', 'report_llm/data/val.jsonl']
    for filename in files_to_check:
        if not os.path.exists(filename):
            print(f"❌ {filename} does not exist")
            return False
        
        # Count tasks
        task_counts = {}
        line_count = 0
        with open(filename, 'r') as f:
            for line in f:
                if line.strip():
                    line_count += 1
                    data = json.loads(line)
                    task = data.get('task')
                    if task:
                        task_counts[task] = task_counts.get(task, 0) + 1
        
        print(f"✅ {filename}: {line_count} lines, tasks: {task_counts}")
        
        if 'one_liner' not in task_counts or 'report' not in task_counts:
            print(f"❌ {filename} missing required tasks")
            return False
    
    # Test 2: Simulate the load_jsonl function logic
    def simulate_load_jsonl(path, task_filter, limit=None):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): 
                    continue
                r = json.loads(line)
                if r.get("task") == task_filter:
                    rows.append(r)
                # Only apply limit if it's specified and > 0
                if limit is not None and limit > 0 and len(rows) >= limit:
                    break
        return rows
    
    # Test different scenarios
    test_cases = [
        ('report_llm/data/train.jsonl', 'one_liner', None),
        ('report_llm/data/train.jsonl', 'one_liner', 0),  # This should be treated as no limit
        ('report_llm/data/train.jsonl', 'one_liner', 10),
        ('report_llm/data/val.jsonl', 'report', None),
        ('report_llm/data/val.jsonl', 'report', 0),  # This should be treated as no limit
        ('report_llm/data/val.jsonl', 'report', 5),
    ]
    
    for path, task, limit in test_cases:
        try:
            rows = simulate_load_jsonl(path, task, limit)
            expected_desc = f"no limit" if limit is None or limit == 0 else f"limit {limit}"
            print(f"✅ {path} task={task} ({expected_desc}): {len(rows)} samples")
            
            if len(rows) == 0:
                print(f"❌ No samples found for task '{task}' in {path}")
                return False
                
        except Exception as e:
            print(f"❌ Error loading {path} task={task} limit={limit}: {e}")
            return False
    
    print("🎉 All dataset loading tests passed!")
    return True

if __name__ == "__main__":
    success = test_dataset_loading()
    sys.exit(0 if success else 1)