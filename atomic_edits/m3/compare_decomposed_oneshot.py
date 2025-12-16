import json
from pathlib import Path
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
import torch
import lpips

# Import the actual editing functions directly
from atomic_edits.editor.region_edit import load_pipelines, masked_property_edit

def run_oneshot_edits():
    """Run combined single-prompt versions of all edits"""
    
    results = []
    loss_fn = lpips.LPIPS(net='alex')
    
    # Load editor pipeline once
    print("[Oneshot] Loading diffusion pipelines...")
    pipes = load_pipelines()
    
    # For each example, create combined prompt
    for example_dir in Path("artifacts/examples").iterdir():
        if not example_dir.is_dir():
            continue
            
        # Load sub-instructions
        sub_file = example_dir / "sub_instructions.json"
        if not sub_file.exists():
            continue
            
        data = json.load(open(sub_file))
        
        # Combine all instructions into one prompt
        combined_prompt = " and ".join([s['text'] for s in data['sub_instructions']])
        
        # Get original image
        orig_img_path = example_dir / "original.png"
        if not orig_img_path.exists():
            continue
        
        print(f"\nRunning one-shot for {example_dir.name}: {combined_prompt}")
        
        # Load original
        orig_img = Image.open(orig_img_path).convert("RGB")
        
        # Run one-shot edit (all instructions at once)
        out_dir = Path("artifacts/oneshot")
        out_dir.mkdir(exist_ok=True, parents=True)
        out_path = out_dir / f"{example_dir.name}.png"
        
        try:
            # Do one-shot edit with NO mask (global edit)
            oneshot_result = masked_property_edit(
                pipes=pipes,
                image=orig_img,
                prompt=combined_prompt,
                mask=None,  # No mask = global edit
                steps=15,
                guidance=7.0,
                image_guidance=1.5,
                seed=42
            )
            oneshot_result.save(out_path)
            print(f"  Saved → {out_path}")
        
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        
        # Compare with decomposed result
        decomposed_path = example_dir / "edited.png"
        if not decomposed_path.exists():
            decomposed_path = example_dir / "final.png"
        
        if not decomposed_path.exists():
            print(f"  WARNING: No decomposed result found")
            continue
        
        decomposed_img = Image.open(decomposed_path).convert("RGB")
        
        # Ensure same size
        if oneshot_result.size != decomposed_img.size:
            oneshot_result = oneshot_result.resize(decomposed_img.size, Image.LANCZOS)
        if orig_img.size != decomposed_img.size:
            orig_img = orig_img.resize(decomposed_img.size, Image.LANCZOS)
        
        # Calculate metrics (convert to grayscale for SSIM)
        oneshot_np = np.array(oneshot_result.convert('L')) / 255.0
        decomposed_np = np.array(decomposed_img.convert('L')) / 255.0
        orig_np = np.array(orig_img.convert('L')) / 255.0
        
        # SSIM comparisons
        ssim_oneshot = ssim(orig_np, oneshot_np, data_range=1.0)
        ssim_decomposed = ssim(orig_np, decomposed_np, data_range=1.0)
        ssim_compare = ssim(oneshot_np, decomposed_np, data_range=1.0)
        
        # LPIPS comparison (on RGB)
        with torch.no_grad():
            oneshot_rgb = np.array(oneshot_result) / 255.0
            decomposed_rgb = np.array(decomposed_img) / 255.0
            
            # Convert to tensors: [H,W,C] -> [1,C,H,W]
            oneshot_t = torch.from_numpy(oneshot_rgb).permute(2,0,1).unsqueeze(0).float()
            decomposed_t = torch.from_numpy(decomposed_rgb).permute(2,0,1).unsqueeze(0).float()
            
            # Normalize to [-1, 1] for LPIPS
            oneshot_t = oneshot_t * 2.0 - 1.0
            decomposed_t = decomposed_t * 2.0 - 1.0
            
            lpips_score = loss_fn(oneshot_t, decomposed_t).item()
        
        results.append({
            'example': example_dir.name,
            'combined_prompt': combined_prompt,
            'ssim_oneshot_vs_orig': float(ssim_oneshot),
            'ssim_decomposed_vs_orig': float(ssim_decomposed),
            'ssim_oneshot_vs_decomposed': float(ssim_compare),
            'lpips_difference': float(lpips_score),
            'better_method': 'decomposed' if ssim_decomposed > ssim_oneshot else 'oneshot',
            'oneshot_path': str(out_path),
            'decomposed_path': str(decomposed_path)
        })
        
        print(f"  SSIM oneshot:    {ssim_oneshot:.3f}")
        print(f"  SSIM decomposed: {ssim_decomposed:.3f}")
        print(f"  Winner: {results[-1]['better_method']}")
    
    return results

if __name__ == "__main__":
    results = run_oneshot_edits()
    
    # Save comparison table
    output_path = Path("artifacts/m3/oneshot_comparison.json")
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Calculate summary stats
    if results:
        avg_ssim_oneshot = np.mean([r['ssim_oneshot_vs_orig'] for r in results])
        avg_ssim_decomposed = np.mean([r['ssim_decomposed_vs_orig'] for r in results])
        decomposed_wins = sum(1 for r in results if r['better_method'] == 'decomposed')
        
        summary = {
            "total_examples": len(results),
            "avg_ssim_oneshot": float(avg_ssim_oneshot),
            "avg_ssim_decomposed": float(avg_ssim_decomposed),
            "decomposed_wins": decomposed_wins,
            "oneshot_wins": len(results) - decomposed_wins,
            "winner": "decomposed" if decomposed_wins > len(results)/2 else "oneshot"
        }
        
        print("\n=== DECOMPOSED vs ONE-SHOT COMPARISON ===")
        print(json.dumps(summary, indent=2))
        print(f"\nResults saved to: {output_path}")
    else:
        print("\n⚠️  No results generated - check if examples exist")