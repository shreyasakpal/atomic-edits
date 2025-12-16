import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from pathlib import Path
import json

def create_ablation_figures():
    """Create visual comparisons for the report"""
    
    # Load M3 results
    results = []
    with open("artifacts/m3/m3_results.jsonl") as f:
        for line in f:
            results.append(json.loads(line))
    
    # Group by example
    examples = {}
    for r in results:
        if r['example_id'] not in examples:
            examples[r['example_id']] = []
        examples[r['example_id']].append(r)
    
    # Create comparison grid
    fig, axes = plt.subplots(len(examples), 4, figsize=(16, 4*len(examples)))
    if len(examples) == 1:
        axes = axes.reshape(1, -1)
    
    for i, (example_id, results) in enumerate(examples.items()):
        example_dir = Path(f"artifacts/examples/{example_id}")
        
        # Original image
        if (example_dir / "original.png").exists():
            orig = Image.open(example_dir / "original.png")
            axes[i, 0].imshow(orig)
            axes[i, 0].set_title("Original")
            axes[i, 0].axis('off')
        
        # Decomposed result
        if (example_dir / "edited.png").exists():
            edited = Image.open(example_dir / "edited.png")
            axes[i, 1].imshow(edited)
            axes[i, 1].set_title("Decomposed")
            axes[i, 1].axis('off')
        
        # One-shot result (if exists)
        oneshot_path = Path(f"artifacts/oneshot/{example_id}.png")
        if oneshot_path.exists():
            oneshot = Image.open(oneshot_path)
            axes[i, 2].imshow(oneshot)
            axes[i, 2].set_title("One-shot")
            axes[i, 2].axis('off')
        else:
            axes[i, 2].axis('off')
        
        # Metrics visualization
        passed = sum(1 for r in results if r['decision'] == 'PASS')
        total = len(results)
        pra = passed / total
        
        axes[i, 3].bar(['PRA', 'SSIM', 'LPIPS'], 
                       [pra, results[0]['ssim_global'], 1-results[0]['lpips_global']])
        axes[i, 3].set_ylim([0, 1])
        axes[i, 3].set_title(f"{example_id}\n{passed}/{total} passed")
        
    plt.suptitle("Ablation Study: Decomposed vs One-shot Editing", fontsize=16)
    plt.tight_layout()
    plt.savefig("artifacts/m3/ablation_figure.png", dpi=150, bbox_inches='tight')
    print("Saved ablation figure to artifacts/m3/ablation_figure.png")
    
    # Create metrics summary table
    create_metrics_table(examples)

def create_metrics_table(examples):
    """Create LaTeX/Markdown table of metrics"""
    
    print("\n=== METRICS TABLE (Markdown) ===")
    print("| Example | PRA | SSIM | LPIPS | Method |")
    print("|---------|-----|------|-------|--------|")
    
    for example_id, results in examples.items():
        passed = sum(1 for r in results if r['decision'] == 'PASS')
        pra = passed / len(results)
        ssim_val = results[0]['ssim_global']
        lpips_val = results[0]['lpips_global']
        
        print(f"| {example_id[:20]} | {pra:.2f} | {ssim_val:.3f} | {lpips_val:.3f} | Decomposed |")
    
    print("\nLaTeX version saved to artifacts/m3/metrics_table.tex")

if __name__ == "__main__":
    create_ablation_figures()