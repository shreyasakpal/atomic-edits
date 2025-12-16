from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

def _read_jsonl(p: Path):
    with open(p) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def load_comparison_data(comparison_file: Path) -> Optional[Dict]:
    """Load one-shot vs decomposed comparison if available."""
    if comparison_file.exists():
        return json.load(open(comparison_file))
    return None

def load_ablation_data(ablation_file: Path) -> Optional[Dict]:
    """Load ablation study results if available."""
    if ablation_file.exists():
        return json.load(open(ablation_file))
    return None

def load_order_sensitivity(order_file: Path) -> Optional[Dict]:
    """Load order sensitivity analysis if available."""
    if order_file.exists():
        return json.load(open(order_file))
    return None

def run(
    results_jsonl: str,
    out_md: str,
    comparison_json: Optional[str] = None,
    ablation_json: Optional[str] = None,
    order_json: Optional[str] = None
):
    rows = list(_read_jsonl(Path(results_jsonl)))
    by_ex = defaultdict(list)
    for r in rows:
        by_ex[r["example_id"]].append(r)

    # Basic metrics
    passes = sum(1 for r in rows if r["decision"] == "PASS")
    pra = passes / max(1, len(rows))
    
    # Preservation metrics if available
    has_preservation = any("ssim_outside" in r for r in rows)
    if has_preservation:
        ssim_outside_values = [r["ssim_outside"] for r in rows if "ssim_outside" in r]
        lpips_outside_values = [r["lpips_outside"] for r in rows if "lpips_outside" in r]
        avg_ssim_outside = np.mean(ssim_outside_values) if ssim_outside_values else None
        avg_lpips_outside = np.mean(lpips_outside_values) if lpips_outside_values else None
    else:
        avg_ssim_outside = avg_lpips_outside = None
    
    # Load additional analyses
    comparison = load_comparison_data(Path(comparison_json)) if comparison_json else None
    ablation = load_ablation_data(Path(ablation_json)) if ablation_json else None
    order_sens = load_order_sensitivity(Path(order_json)) if order_json else None

    # Build report
    lines = [
        "# M3 Comprehensive Evaluation Report",
        "",
        "## Executive Summary",
        f"- **Per-Requirement Accuracy (PRA)**: {pra:.3f}",
        f"- **Total sub-instructions evaluated**: {len(rows)}",
        f"- **Passed requirements**: {passes}",
    ]
    
    if avg_ssim_outside is not None:
        lines.append(f"- **Average SSIM (outside mask)**: {avg_ssim_outside:.3f}")
    if avg_lpips_outside is not None:
        lines.append(f"- **Average LPIPS (outside mask)**: {avg_lpips_outside:.3f}")
    
    lines.extend(["", "---", ""])
    
    # One-shot vs Decomposed Comparison
    if comparison:
        comp = comparison["comparison"]
        lines.extend([
            "## One-Shot vs Decomposed Comparison",
            f"- **One-shot PRA**: {comp['oneshot_pra']:.3f}",
            f"- **Decomposed PRA**: {comp['decomposed_pra']:.3f}",
            f"- **Improvement (Δ)**: {comp['delta_pra']:+.3f}",
            "",
            "The decomposed approach " + 
            ("**outperforms**" if comp['delta_pra'] > 0 else "**underperforms**") +
            f" one-shot editing by {abs(comp['delta_pra']):.1%}",
            "", "---", ""
        ])
    
    # Ablation Studies
    if ablation and "summary" in ablation:
        lines.extend(["## Ablation Studies", ""])
        
        # Mask impact
        if "mask_impact" in ablation["summary"]:
            impact = ablation["summary"]["mask_impact"]
            lines.extend([
                "### Impact of Region Masks",
                f"- **ΔPRA (with mask - without)**: {impact['delta_pra']:+.3f}",
                f"- **ΔSSIM**: {impact['delta_ssim']:+.3f}",
                f"- **ΔLPIPS**: {impact['delta_lpips']:+.3f}",
                "",
                "Region masks " + 
                ("**improve**" if impact['delta_pra'] > 0 else "**degrade**") +
                f" accuracy by {abs(impact['delta_pra']):.1%}",
                ""
            ])
        
        # Configuration comparison table
        lines.extend([
            "### Configuration Performance",
            "| Configuration | PRA | SSIM | LPIPS |",
            "|---|---:|---:|---:|"
        ])
        
        for config_name, stats in ablation["summary"].items():
            if config_name != "mask_impact" and "pra_mean" in stats:
                lines.append(
                    f"| {config_name} | "
                    f"{stats['pra_mean']:.3f}±{stats['pra_std']:.3f} | "
                    f"{stats['ssim_mean']:.3f}±{stats['ssim_std']:.3f} | "
                    f"{stats['lpips_mean']:.3f}±{stats['lpips_std']:.3f} |"
                )
        
        lines.extend(["", "---", ""])
    
    # Order Sensitivity
    if order_sens:
        lines.extend([
            "## Order Sensitivity Analysis",
            f"- **Examples analyzed**: {order_sens['total_examples']}",
            f"- **Order-sensitive examples**: {order_sens['order_sensitive_examples']}",
            f"- **Sensitivity rate**: {order_sens['sensitivity_rate']:.1%}",
            ""
        ])
        
        # Show most sensitive examples
        if "examples" in order_sens:
            sensitive_examples = [
                ex for ex in order_sens["examples"] 
                if not ex.get("skipped") and ex.get("sensitivity_metrics", {}).get("is_order_sensitive")
            ]
            
            if sensitive_examples:
                lines.extend([
                    "### Most Order-Sensitive Examples",
                    "| Example | # Instructions | PRA Range | PRA StdDev |",
                    "|---|---:|---:|---:|"
                ])
                
                for ex in sorted(sensitive_examples, 
                               key=lambda x: x["sensitivity_metrics"]["pra_std"], 
                               reverse=True)[:5]:
                    metrics = ex["sensitivity_metrics"]
                    lines.append(
                        f"| {ex['example_id']} | {ex['num_instructions']} | "
                        f"{metrics['pra_range']:.3f} | {metrics['pra_std']:.3f} |"
                    )
        
        lines.extend(["", "---", ""])
    
    # Per-Example Details
    lines.extend(["## Detailed Per-Example Results", ""])
    
    for ex, subs in sorted(by_ex.items()):
        p = sum(1 for r in subs if r["decision"] == "PASS")
        ex_pra = p / len(subs)
        
        # Get preservation metrics for this example
        ex_ssim = None
        ex_lpips = None
        if has_preservation and subs:
            if "ssim_global" in subs[0]:
                ex_ssim = subs[0]["ssim_global"]  # Same for all subs in example
            if "lpips_global" in subs[0]:
                ex_lpips = subs[0]["lpips_global"]
        
        header = f"### {ex} — {p}/{len(subs)} passed (PRA: {ex_pra:.3f}"
        if ex_ssim is not None:
            header += f", SSIM: {ex_ssim:.3f}"
        if ex_lpips is not None:
            header += f", LPIPS: {ex_lpips:.3f}"
        header += ")"
        
        lines.extend([header, ""])
        
        # Sub-instruction table
        lines.extend([
            "| Sub | Text | Decision | VQA Conf | CLIP Margin | Preservation |",
            "|---|---|---:|---:|---:|---:|"
        ])
        
        for r in subs:
            pres_str = ""
            if "ssim_outside" in r:
                pres_str = f"SSIM: {r['ssim_outside']:.3f}"
                if "lpips_outside" in r:
                    pres_str += f", LPIPS: {r['lpips_outside']:.3f}"
            elif r.get("mask_path"):
                pres_str = "Mask available"
            else:
                pres_str = "Global"
            
            # Truncate text for readability
            text = r['text'][:40] + "..." if len(r['text']) > 40 else r['text']
            
            lines.append(
                f"| {r['sub_id']} | {text} | "
                f"**{r['decision']}** | {r['vqa_yes_conf']:.2f} | "
                f"{r['clip_margin']:.3f} | {pres_str} |"
            )
        
        lines.append("")
    
    # Summary Statistics
    lines.extend([
        "---", "",
        "## Summary Statistics",
        "",
        f"- **Overall PRA**: {pra:.3f} ({passes}/{len(rows)} passed)",
    ])
    
    # Example-level statistics
    ex_pra_values = [sum(1 for r in subs if r["decision"] == "PASS") / len(subs) 
                     for subs in by_ex.values()]
    lines.extend([
        f"- **Mean example PRA**: {np.mean(ex_pra_values):.3f}",
        f"- **Std example PRA**: {np.std(ex_pra_values):.3f}",
        f"- **Examples with 100% pass**: {sum(1 for v in ex_pra_values if v == 1.0)}/{len(ex_pra_values)}",
        f"- **Examples with 0% pass**: {sum(1 for v in ex_pra_values if v == 0.0)}/{len(ex_pra_values)}",
    ])
    
    # Failure analysis
    lines.extend(["", "## Failure Analysis", ""])
    
    failure_reasons = defaultdict(int)
    for r in rows:
        if r["decision"] == "FAIL":
            failure_reasons[r["rule"]] += 1
    
    if failure_reasons:
        lines.extend([
            "| Failure Reason | Count |",
            "|---|---:|"
        ])
        for reason, count in sorted(failure_reasons.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"| {reason} | {count} |")
    else:
        lines.append("No failures detected!")
    
    lines.extend(["", "---", "", 
                  "*Report generated for Milestone 3 evaluation*"])

    # Write report
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(out_md).write_text("\n".join(lines), encoding="utf-8")
    print(f"[M3] Wrote comprehensive report → {out_md}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="artifacts/m3/m3_results.jsonl")
    ap.add_argument("--out", default="artifacts/m3/m3_report.md")
    ap.add_argument("--comparison", default="artifacts/m3/oneshot_comparison.json", 
                    help="One-shot vs decomposed comparison JSON")
    ap.add_argument("--ablation", default="artifacts/m3/ablations/ablation_summary.json",
                    help="Ablation study results JSON")
    ap.add_argument("--order", default="artifacts/m3/order_sensitivity/order_sensitivity_summary.json",
                    help="Order sensitivity analysis JSON")
    args = ap.parse_args()
    
    run(
        args.results, 
        args.out,
        args.comparison,
        args.ablation,
        args.order
    )

if __name__ == "__main__":
    main()