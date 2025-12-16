# m3/ablation_study.py
"""
Ablation studies to understand the impact of different components:
1. With vs without masks
2. Different mask dilation sizes
3. Different number of diffusion steps
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image
import numpy as np
import yaml
from tqdm import tqdm

from .evaluator.vqa_blip2 import BLIP2YesNo
from .evaluator.clip_checks import CLIPAttribute
from .evaluator.aggregation import decide_pass_polarity, Thresholds
from .evaluator.preservation_metrics import PreservationMetrics

def run_edit_with_config(
    example_dir: Path,
    output_dir: Path,
    use_mask: bool = True,
    diffusion_steps: int = 10,
    guidance_scale: float = 5.0,
    image_guidance_scale: float = 2.2,
    seed: int = 42,
    mask_dilation: Optional[int] = None
) -> Dict:
    """
    Run edit with specific configuration and return paths.
    """
    meta = json.load(open(example_dir / "sub_instructions.json"))
    
    # Prepare modified plan
    plan = {
        "image": str(example_dir / "original.png"),
        "seed": seed,
        "sub_instructions": meta["sub_instructions"]
    }
    
    # Modify plan based on ablation config
    if not use_mask:
        # Remove mask-related fields
        for sub in plan["sub_instructions"]:
            sub["requires_region"] = False
            sub.pop("mask_phrase", None)
    
    plan_path = output_dir / "ablation_plan.json"
    with open(plan_path, "w") as f:
        json.dump(plan, f, indent=2)
    
    # Build command
    cmd = [
        sys.executable, "-m", "atomic_edits.editor.cli",
        "--image", str(example_dir / "original.png"),
        "--json", str(plan_path),
        "--outdir", str(output_dir),
        "--steps", str(diffusion_steps),
        "--guidance-scale", str(guidance_scale),
        "--image-guidance-scale", str(image_guidance_scale),
        "--seed", str(seed)
    ]
    
    if not use_mask:
        cmd.append("--no-mask")  # If your CLI supports this
    
    if mask_dilation is not None:
        cmd.extend(["--mask-dilation", str(mask_dilation)])
    
    # Execute
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Edit failed: {result.stderr}")
    
    # Find output
    final_path = output_dir / "final.png"
    if not final_path.exists():
        for alt in ["edited.png", "sequential_final.png"]:
            if (output_dir / alt).exists():
                final_path = output_dir / alt
                break
    
    return {
        "edited_path": str(final_path),
        "plan_path": str(plan_path),
        "use_mask": use_mask,
        "steps": diffusion_steps
    }

def evaluate_ablation(
    example_dir: Path,
    edited_path: Path,
    config_path: Path,
    calibration_path: Optional[Path] = None
) -> Dict:
    """
    Evaluate a single ablation configuration.
    """
    cfg = yaml.safe_load(open(config_path))
    device = cfg.get("device", "cuda")
    
    vqa = BLIP2YesNo(cfg["vqa"]["model_name"], device, cfg["vqa"]["max_new_tokens"])
    clip = CLIPAttribute(cfg["clip"]["model_name"], cfg["clip"]["pretrained"], device)
    preservation = PreservationMetrics(device)
    
    th = Thresholds(
        vqa_yes_threshold=cfg["calibration"]["vqa_yes_threshold"],
        clip_margin_threshold=cfg["calibration"]["clip_margin_threshold"],
    )
    if calibration_path and calibration_path.exists():
        cal = yaml.safe_load(open(calibration_path))
        th.vqa_yes_threshold = float(cal["vqa_yes_threshold"])
        th.clip_margin_threshold = float(cal["clip_margin_threshold"])
    
    # Load images and metadata
    meta = json.load(open(example_dir / "sub_instructions.json"))
    original = Image.open(example_dir / "original.png").convert("RGB")
    edited = Image.open(edited_path).convert("RGB")
    
    results = {
        "sub_results": [],
        "preservation": {}
    }
    
    # Evaluate each sub-instruction
    for sub in meta["sub_instructions"]:
        phrase = sub.get("target_phrase") or sub["text"]
        question = sub["vqa_question"]
        expect_present = sub.get("expect_present", True)
        
        vqa_out = vqa.ask(edited, question, cfg["vqa"]["yes_words"], cfg["vqa"]["no_words"])
        clip_out = clip.score_phrase(
            edited,
            cfg["clip"]["positive_templates"],
            cfg["clip"]["negative_templates"],
            phrase
        )
        
        passed, rule = decide_pass_polarity(
            vqa_yes_conf=vqa_out["yes_conf"],
            clip_margin=clip_out["margin"],
            thresholds=th,
            expect_present=expect_present,
        )
        
        results["sub_results"].append({
            "sub_id": sub["id"],
            "passed": passed,
            "vqa_yes_conf": vqa_out["yes_conf"],
            "clip_margin": clip_out["margin"]
        })
    
    # Compute preservation
    pres = preservation.compute_outside_mask(original, edited)
    results["preservation"] = pres
    
    # Compute PRA
    results["pra"] = sum(1 for r in results["sub_results"] if r["passed"]) / len(results["sub_results"])
    
    return results

def run_ablations(
    examples_root: Path,
    config_path: Path,
    output_dir: Path,
    max_examples: int = 10,
    calibration_path: Optional[Path] = None
) -> Dict:
    """
    Run multiple ablation configurations and compare.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define ablation configurations
    ablation_configs = [
        {"name": "baseline", "use_mask": True, "steps": 10},
        {"name": "no_mask", "use_mask": False, "steps": 10},
        {"name": "more_steps", "use_mask": True, "steps": 20},
        {"name": "fewer_steps", "use_mask": True, "steps": 5},
        {"name": "high_guidance", "use_mask": True, "steps": 10, "guidance_scale": 9.0},
        {"name": "low_guidance", "use_mask": True, "steps": 10, "guidance_scale": 3.0},
    ]
    
    all_results = {config["name"]: [] for config in ablation_configs}
    
    # Process examples
    example_dirs = [d for d in examples_root.iterdir() 
                    if d.is_dir() and (d / "sub_instructions.json").exists()][:max_examples]
    
    for ex_dir in tqdm(example_dirs, desc="Running ablations"):
        example_results = {"example_id": ex_dir.name}
        
        for config in ablation_configs:
            config_name = config["name"]
            config_dir = output_dir / ex_dir.name / config_name
            config_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                # Run edit with this configuration
                edit_info = run_edit_with_config(
                    ex_dir,
                    config_dir,
                    use_mask=config.get("use_mask", True),
                    diffusion_steps=config.get("steps", 10),
                    guidance_scale=config.get("guidance_scale", 5.0),
                    seed=42
                )
                
                # Evaluate
                eval_results = evaluate_ablation(
                    ex_dir,
                    Path(edit_info["edited_path"]),
                    config_path,
                    calibration_path
                )
                
                example_results[config_name] = {
                    "pra": eval_results["pra"],
                    "preservation": eval_results["preservation"],
                    "config": config
                }
                
                all_results[config_name].append({
                    "example_id": ex_dir.name,
                    "pra": eval_results["pra"],
                    "ssim": eval_results["preservation"]["ssim_global"],
                    "lpips": eval_results["preservation"]["lpips_global"]
                })
                
            except Exception as e:
                print(f"Failed {config_name} for {ex_dir.name}: {e}")
                all_results[config_name].append({
                    "example_id": ex_dir.name,
                    "error": str(e)
                })
    
    # Compute aggregate statistics
    summary = {}
    for config_name, results in all_results.items():
        valid_results = [r for r in results if "pra" in r]
        if valid_results:
            pra_values = [r["pra"] for r in valid_results]
            ssim_values = [r["ssim"] for r in valid_results]
            lpips_values = [r["lpips"] for r in valid_results]
            
            summary[config_name] = {
                "num_examples": len(valid_results),
                "pra_mean": np.mean(pra_values),
                "pra_std": np.std(pra_values),
                "ssim_mean": np.mean(ssim_values),
                "ssim_std": np.std(ssim_values),
                "lpips_mean": np.mean(lpips_values),
                "lpips_std": np.std(lpips_values),
            }
    
    # Compute deltas from baseline
    if "baseline" in summary and "no_mask" in summary:
        summary["mask_impact"] = {
            "delta_pra": summary["baseline"]["pra_mean"] - summary["no_mask"]["pra_mean"],
            "delta_ssim": summary["baseline"]["ssim_mean"] - summary["no_mask"]["ssim_mean"],
            "delta_lpips": summary["baseline"]["lpips_mean"] - summary["no_mask"]["lpips_mean"],
        }
    
    return {
        "configurations": ablation_configs,
        "results": all_results,
        "summary": summary
    }

def main():
    ap = argparse.ArgumentParser(description="Run ablation studies")
    ap.add_argument("--examples-root", default="artifacts/examples", type=Path)
    ap.add_argument("--config", default="atomic_edits/m3/config.yaml", type=Path)
    ap.add_argument("--output-dir", default="artifacts/m3/ablations", type=Path)
    ap.add_argument("--max-examples", type=int, default=10)
    ap.add_argument("--calibration", default="artifacts/m3/calibration.yml", type=Path)
    
    args = ap.parse_args()
    
    results = run_ablations(
        args.examples_root,
        args.config,
        args.output_dir,
        args.max_examples,
        args.calibration
    )
    
    # Save results
    output_file = args.output_dir / "ablation_summary.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\nAblation Study Summary:")
    for config_name, stats in results["summary"].items():
        print(f"\n{config_name}:")
        print(f"  PRA: {stats['pra_mean']:.3f} ± {stats['pra_std']:.3f}")
        print(f"  SSIM: {stats['ssim_mean']:.3f} ± {stats['ssim_std']:.3f}")
        print(f"  LPIPS: {stats['lpips_mean']:.3f} ± {stats['lpips_std']:.3f}")
    
    if "mask_impact" in results["summary"]:
        impact = results["summary"]["mask_impact"]
        print(f"\nMask Impact (Baseline - No Mask):")
        print(f"  ΔPRA: {impact['delta_pra']:+.3f}")
        print(f"  ΔSSIM: {impact['delta_ssim']:+.3f}")
        print(f"  ΔLPIPS: {impact['delta_lpips']:+.3f}")

if __name__ == "__main__":
    main()