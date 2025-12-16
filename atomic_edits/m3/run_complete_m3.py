#!/usr/bin/env python3
"""
Master script to run complete Milestone 3 evaluation.
Executes all components in the correct order and generates final report.
"""
import argparse
import subprocess
import sys
from pathlib import Path
import json
import time

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    
    if result.returncode != 0:
        print(f"❌ Failed after {elapsed:.1f}s")
        print(f"Error: {result.stderr}")
        return False
    
    print(f"✅ Completed in {elapsed:.1f}s")
    if result.stdout:
        print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
    return True

def main():
    ap = argparse.ArgumentParser(description="Run complete Milestone 3 evaluation")
    ap.add_argument("--examples-root", default="artifacts/examples", type=Path,
                    help="Root directory containing example outputs from M2")
    ap.add_argument("--out-root", default="artifacts/m3", type=Path,
                    help="Output directory for M3 results")
    ap.add_argument("--config", default="atomic_edits/m3/config.yaml", type=Path)
    ap.add_argument("--skip-calibration", action="store_true",
                    help="Skip calibration step (use existing calibration.yml)")
    ap.add_argument("--skip-oneshot", action="store_true",
                    help="Skip one-shot comparison")
    ap.add_argument("--skip-ablation", action="store_true",
                    help="Skip ablation studies")
    ap.add_argument("--skip-order", action="store_true",
                    help="Skip order sensitivity analysis")
    ap.add_argument("--max-examples", type=int, default=10,
                    help="Maximum examples for ablation/order studies")
    ap.add_argument("--quick", action="store_true",
                    help="Quick mode: skip time-intensive studies")
    
    args = ap.parse_args()
    
    if args.quick:
        args.skip_ablation = True
        args.skip_order = True
        print("Quick mode enabled - skipping ablation and order sensitivity studies")
    
    # Create output directory
    args.out_root.mkdir(parents=True, exist_ok=True)
    
    # Track what was run
    components_run = []
    
    print("\n" + "="*60)
    print("MILESTONE 3 EVALUATION PIPELINE")
    print("="*60)
    
    # Step 1: Calibration (optional)
    calibration_file = args.out_root / "calibration.yml"
    if not args.skip_calibration:
        labels_file = args.out_root / "calib_labels.csv"
        if labels_file.exists():
            success = run_command([
                sys.executable, "-m", "atomic_edits.m3.calibrate",
                "--config", str(args.config),
                "--examples-root", str(args.examples_root),
                "--labels-csv", str(labels_file),
                "--out", str(calibration_file)
            ], "Calibration (finding optimal thresholds)")
            
            if success:
                components_run.append("calibration")
        else:
            print(f"⚠️ Skipping calibration - no labels file found at {labels_file}")
    
    # Step 2: Main evaluation with preservation metrics
    success = run_command([
        sys.executable, "-m", "atomic_edits.m3.evaluate",
        "--config", str(args.config),
        "--examples-root", str(args.examples_root),
        "--out-root", str(args.out_root),
        "--calibration", str(calibration_file) if calibration_file.exists() else ""
    ], "Main evaluation (VQA + CLIP + Preservation)")
    
    if success:
        components_run.append("main_evaluation")
    else:
        print("❌ Main evaluation failed - stopping")
        sys.exit(1)
    
    # Step 3: One-shot comparison (optional)
    comparison_file = args.out_root / "oneshot_comparison.json"
    if not args.skip_oneshot:
        success = run_command([
            sys.executable, "-m", "atomic_edits.m3.oneshot_baseline",
            "--examples-root", str(args.examples_root),
            "--config", str(args.config),
            "--calibration", str(calibration_file) if calibration_file.exists() else "",
            "--output", str(comparison_file)
        ], "One-shot vs Decomposed comparison")
        
        if success:
            components_run.append("oneshot_comparison")
    
    # Step 4: Ablation studies (optional, time-intensive)
    ablation_dir = args.out_root / "ablations"
    ablation_file = ablation_dir / "ablation_summary.json"
    if not args.skip_ablation:
        success = run_command([
            sys.executable, "-m", "atomic_edits.m3.ablation_study",
            "--examples-root", str(args.examples_root),
            "--config", str(args.config),
            "--output-dir", str(ablation_dir),
            "--max-examples", str(args.max_examples),
            "--calibration", str(calibration_file) if calibration_file.exists() else ""
        ], f"Ablation studies (max {args.max_examples} examples)")
        
        if success:
            components_run.append("ablation_studies")
    
    # Step 5: Order sensitivity (optional, time-intensive)
    order_dir = args.out_root / "order_sensitivity"
    order_file = order_dir / "order_sensitivity_summary.json"
    if not args.skip_order:
        success = run_command([
            sys.executable, "-m", "atomic_edits.m3.order_sensitivity",
            "--examples-root", str(args.examples_root),
            "--config", str(args.config),
            "--output-dir", str(order_dir),
            "--max-examples", str(args.max_examples),
            "--max-variants", "5",
            "--calibration", str(calibration_file) if calibration_file.exists() else ""
        ], f"Order sensitivity analysis (max {args.max_examples} examples)")
        
        if success:
            components_run.append("order_sensitivity")
    
    # Step 6: Generate comprehensive report
    report_cmd = [
        sys.executable, "-m", "atomic_edits.m3.report",
        "--results", str(args.out_root / "m3_results.jsonl"),
        "--out", str(args.out_root / "m3_report.md")
    ]
    
    if comparison_file.exists():
        report_cmd.extend(["--comparison", str(comparison_file)])
    if ablation_file.exists():
        report_cmd.extend(["--ablation", str(ablation_file)])
    if order_file.exists():
        report_cmd.extend(["--order", str(order_file)])
    
    success = run_command(report_cmd, "Generate comprehensive report")
    
    if success:
        components_run.append("report")
    
    # Final summary
    print("\n" + "="*60)
    print("MILESTONE 3 EVALUATION COMPLETE")
    print("="*60)
    print(f"✅ Components run: {', '.join(components_run)}")
    print(f"📊 Report generated: {args.out_root / 'm3_report.md'}")
    
    # Print key metrics if available
    results_file = args.out_root / "m3_results.jsonl"
    if results_file.exists():
        rows = []
        with open(results_file) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        
        if rows:
            passes = sum(1 for r in rows if r["decision"] == "PASS")
            pra = passes / len(rows)
            print(f"\n📈 Key Metrics:")
            print(f"   - Per-Requirement Accuracy (PRA): {pra:.3f}")
            print(f"   - Requirements evaluated: {len(rows)}")
            print(f"   - Requirements passed: {passes}")
    
    if comparison_file.exists():
        comp = json.load(open(comparison_file))["comparison"]
        print(f"   - One-shot PRA: {comp['oneshot_pra']:.3f}")
        print(f"   - Decomposed PRA: {comp['decomposed_pra']:.3f}")
        print(f"   - Improvement: {comp['delta_pra']:+.3f}")

if __name__ == "__main__":
    main()