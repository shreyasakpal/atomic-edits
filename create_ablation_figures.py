import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import json

def create_ablation_figures():
    """Create visual comparisons for the report"""
    
    # Load M3 results - use absolute path
    results = []
    results_file = Path("artifacts/m3/m3_results.jsonl")
    if not results_file.exists():
        print(f"Results file not found at {results_file}")
        return
        
    with open(results_file) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    
    # Group by example
    examples = {}
    for r in results:
        if r['example_id'] not in examples:
            examples[r['example_id']] = {}
        examples[r['example_id']][r['sub_id']] = r
    
    # Simple metrics summary
    print("\n=== METRICS SUMMARY ===")
    print("| Example | Total | Passed | PRA | SSIM | LPIPS |")
    print("|---------|-------|--------|-----|------|-------|")
    
    for example_id, subs in examples.items():
        total = len(subs)
        passed = sum(1 for s in subs.values() if s['decision'] == 'PASS')
        pra = passed / total if total > 0 else 0
        ssim_val = list(subs.values())[0].get('ssim_global', 0)
        lpips_val = list(subs.values())[0].get('lpips_global', 0)
        
        print(f"| {example_id[:20]} | {total} | {passed} | {pra:.2f} | {ssim_val:.3f} | {lpips_val:.3f} |")
    
    overall_total = len(results)
    overall_passed = sum(1 for r in results if r['decision'] == 'PASS')
    overall_pra = overall_passed / overall_total if overall_total > 0 else 0
    
    print(f"\nOVERALL: {overall_passed}/{overall_total} passed (PRA: {overall_pra:.2%})")

if __name__ == "__main__":
    create_ablation_figures()
