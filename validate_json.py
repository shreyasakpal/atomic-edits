#!/usr/bin/env python3
"""
Validate parsed JSON files and produce a small summary CSV.
Input dir (default): artifacts/parsed
Output CSV:         artifacts/parsed_summary.csv

Columns: file, n_steps, n_requires_region, has_action_object_order, is_monotonic_order
"""
import json, sys, csv
from pathlib import Path

def check_file(p: Path):
    data = json.loads(p.read_text())
    steps = data.get("sub_instructions", [])
    n_steps = len(steps)
    n_region = sum(1 for s in steps if s.get("requires_region") is True)
    has_required = all(("action" in s) and ("object" in s) and ("order" in s) for s in steps)
    order = [s.get("order", i+1) for i, s in enumerate(steps)]
    is_monotonic = order == sorted(order)
    return n_steps, n_region, has_required, is_monotonic

def main():
    indir = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/parsed")
    files = sorted(indir.glob("*.json"))
    if not files:
        print(f"No JSON files found in {indir}", file=sys.stderr)
        sys.exit(2)

    out_csv = Path("artifacts/parsed_summary.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file","n_steps","n_requires_region","has_action_object_order","is_monotonic_order"])
        for p in files:
            n_steps, n_region, has_req, monot = check_file(p)
            w.writerow([p.name, n_steps, n_region, has_req, monot])
    print(f"Wrote summary -> {out_csv}")

if __name__ == "__main__":
    main()
