#!/usr/bin/env python
import json, argparse, random, os
from dataclasses import asdict
from .types import TelemetryNLIn, Contribution
from .prompt_builder import build_inputs_for_llm

def load_telemetry(path: str):
    """Load telemetry JSONL fully, then yield records.
    This ensures the file handle is closed before iteration (Windows-safe delete).
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    for r in rows:
        yield r

def to_struct(obj: dict) -> TelemetryNLIn:
    contribs = [Contribution(**c) for c in obj.get("contributions", [])]
    return TelemetryNLIn(
        p_hit_calib=obj["p_hit_calib"],
        p_kill_calib=obj["p_kill_calib"],
        p_hit_masked=obj["p_hit_masked"],
        p_kill_masked=obj["p_kill_masked"],
        spoof_risk=obj["spoof_risk"],
        flags=obj.get("flags", {}),
        exp=obj.get("exp", {}),
        meta=obj.get("meta", {}),
        contributions=contribs
    )

def main(args):
    rows = []
    for rec in load_telemetry(args.telemetry):
        t = to_struct(rec["inputs"] if "inputs" in rec else rec)
        prompt = build_inputs_for_llm(t, style=args.style)
        # Expect that targets are present if you have gold data; otherwise use silver if provided.
        targets = rec.get("targets") or {}
        one = targets.get("one_liner")
        rep = targets.get("report")
        if one:
            rows.append({"task":"one_liner","prompt":prompt,"target":one})
        if rep:
            rows.append({"task":"report","prompt":prompt,"target":rep})

    if not rows:
        raise SystemExit("No rows found. Ensure telemetry JSONL has 'targets' for supervision.")

    random.seed(42)
    random.shuffle(rows)
    n = len(rows)
    n_val = max(1, int(0.1 * n))
    n_test = max(1, int(0.1 * n))
    train = rows[: n - n_val - n_test]
    val   = rows[n - n_val - n_test : n - n_test]
    test  = rows[n - n_test :]

    os.makedirs(args.out_dir, exist_ok=True)
    for name, data in [("train", train), ("val", val), ("test", test)]:
        with open(os.path.join(args.out_dir, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(train)} train, {len(val)} val, {len(test)} test")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--telemetry", required=True, help="Path to telemetry JSONL with targets")
    p.add_argument("--out_dir", required=True, help="Output directory for train/val/test JSONL")
    p.add_argument("--style", default="resmi", choices=["resmi","madde","anlatımcı"])
    main(p.parse_args())

