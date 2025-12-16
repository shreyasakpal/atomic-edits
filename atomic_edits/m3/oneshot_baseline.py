# m3/oneshot_baseline.py
"""
Run one-shot baseline: apply the full composite instruction in a single pass.
Compare against decomposed sequential execution.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image
import yaml

from .evaluator.vqa_blip2 import BLIP2YesNo
from .evaluator.clip_checks import CLIPAttribute
from .evaluator.aggregation import decide_pass_polarity, Thresholds
from .evaluator.preservation_metrics import PreservationMetrics

def _build_composite_prompt(sub_instructions: List[Dict]) -> str:
    """
    Reconstruct the original composite instruction from sub-instructions.
    Example: ["make the shirt blue", "remove the logo"] -> "make the shirt blue and remove the logo"
    """
    texts = []
    for sub in sub_instructions:
        text = sub.get("text", "")
        if text and text not in texts:  # Avoid duplicates
            texts.append(text)
    
    if len(texts) == 0:
        return ""
    elif len(texts) == 1:
        return texts[0]
    elif len(texts) == 2:
        return f"{texts[0]} and {texts[1]}"
    else:
        # Multiple instructions: "A, B, and C"
        return ", ".join(texts[:-1]) + f", and {texts[-1]}"

def run_oneshot_edit(
    image_path: Path,
    composite_prompt: str,
    output_path: Path,
    seed: int = 42,
    steps: int = 20,
    guidance: float = 7.0,
    image_guidance: float = 1.5,
) -> Path:
    """
    Execute one-shot edit using InstructPix2Pix directly on composite instruction.
    """
    cmd = [
        sys.executable, "-m", "atomic_edits.editor.cli_ip2p",
        "--image", str(image_path),
        "--prompt", composite_prompt,
        "--out", str(output_path),
        "--seed", str(seed),
        "--steps", str(steps),
        "--guidance-scale", str(guidance),
        "--image-guidance-scale", str(image_guidance),
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"One-shot edit failed: {result.stderr}")
    
    return output_path

def evaluate_oneshot(
    examples_root: Path,
    config_path: Path,
    calibration_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    Run one-shot baseline and evaluate using same criteria as decomposed.
    Returns comparative metrics.
    """
    cfg = yaml.safe_load(open(config_path))
    device = cfg.get("device", "cuda")
    
    # Initialize evaluators
    vqa = BLIP2YesNo(cfg["vqa"]["model_name"], device, cfg["vqa"]["max_new_tokens"])
    clip = CLIPAttribute(cfg["clip"]["model_name"], cfg["clip"]["pretrained"], device)
    preservation = PreservationMetrics(device)
    
    # Load thresholds
    th = Thresholds(
        vqa_yes_threshold=cfg["calibration"]["vqa_yes_threshold"],
        clip_margin_threshold=cfg["calibration"]["clip_margin_threshold"],
    )
    if calibration_path and calibration_path.exists():
        cal = yaml.safe_load(open(calibration_path))
        th.vqa_yes_threshold = float(cal["vqa_yes_threshold"])
        th.clip_margin_threshold = float(cal["clip_margin_threshold"])
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "oneshot": [],
        "decomposed": [],
        "comparison": {}
    }
    
    # Process each example
    for ex_dir in sorted(examples_root.iterdir()):
        if not ex_dir.is_dir() or not (ex_dir / "sub_instructions.json").exists():
            continue
        
        meta = json.load(open(ex_dir / "sub_instructions.json"))
        original = Image.open(ex_dir / "original.png").convert("RGB")
        
        # Build composite prompt
        composite_prompt = _build_composite_prompt(meta["sub_instructions"])
        if not composite_prompt:
            continue
        
        # Run one-shot edit
        oneshot_path = ex_dir / "oneshot_edited.png"
        if not oneshot_path.exists():
            print(f"Running one-shot for {ex_dir.name}: {composite_prompt}")
            oneshot_path = run_oneshot_edit(
                ex_dir / "original.png",
                composite_prompt,
                oneshot_path
            )
        
        oneshot_img = Image.open(oneshot_path).convert("RGB")
        
        # Also load decomposed result
        decomposed_path = ex_dir / "edited.png"
        if not decomposed_path.exists():
            decomposed_path = ex_dir / "final.png"  # Fallback
        decomposed_img = Image.open(decomposed_path).convert("RGB") if decomposed_path.exists() else None
        
        # Evaluate both on all sub-instructions
        ex_oneshot = {"example_id": ex_dir.name, "prompt": composite_prompt, "sub_results": []}
        ex_decomposed = {"example_id": ex_dir.name, "sub_results": []} if decomposed_img else None
        
        for sub in meta["sub_instructions"]:
            phrase = sub.get("target_phrase") or sub["text"]
            question = sub["vqa_question"]
            expect_present = sub.get("expect_present", True)
            
            # Evaluate one-shot
            vqa_out = vqa.ask(oneshot_img, question, cfg["vqa"]["yes_words"], cfg["vqa"]["no_words"])
            clip_out = clip.score_phrase(
                oneshot_img,
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
            
            ex_oneshot["sub_results"].append({
                "sub_id": sub["id"],
                "text": sub["text"],
                "passed": passed,
                "vqa_yes_conf": vqa_out["yes_conf"],
                "clip_margin": clip_out["margin"]
            })
            
            # Evaluate decomposed if available
            if decomposed_img:
                vqa_out_d = vqa.ask(decomposed_img, question, cfg["vqa"]["yes_words"], cfg["vqa"]["no_words"])
                clip_out_d = clip.score_phrase(
                    decomposed_img,
                    cfg["clip"]["positive_templates"],
                    cfg["clip"]["negative_templates"],
                    phrase
                )
                
                passed_d, _ = decide_pass_polarity(
                    vqa_yes_conf=vqa_out_d["yes_conf"],
                    clip_margin=clip_out_d["margin"],
                    thresholds=th,
                    expect_present=expect_present,
                )
                
                if ex_decomposed:
                    ex_decomposed["sub_results"].append({
                        "sub_id": sub["id"],
                        "text": sub["text"],
                        "passed": passed_d,
                        "vqa_yes_conf": vqa_out_d["yes_conf"],
                        "clip_margin": clip_out_d["margin"]
                    })
        
        # Compute preservation metrics
        pres_oneshot = preservation.compute_outside_mask(original, oneshot_img)
        ex_oneshot["preservation"] = pres_oneshot
        
        if decomposed_img:
            pres_decomposed = preservation.compute_outside_mask(original, decomposed_img)
            ex_decomposed["preservation"] = pres_decomposed
        
        results["oneshot"].append(ex_oneshot)
        if ex_decomposed:
            results["decomposed"].append(ex_decomposed)
    
    # Compute aggregate metrics
    oneshot_pra = sum(
        sum(1 for s in ex["sub_results"] if s["passed"])
        for ex in results["oneshot"]
    ) / max(1, sum(len(ex["sub_results"]) for ex in results["oneshot"]))
    
    decomposed_pra = sum(
        sum(1 for s in ex["sub_results"] if s["passed"])
        for ex in results["decomposed"]
    ) / max(1, sum(len(ex["sub_results"]) for ex in results["decomposed"])) if results["decomposed"] else 0
    
    results["comparison"] = {
        "oneshot_pra": oneshot_pra,
        "decomposed_pra": decomposed_pra,
        "delta_pra": decomposed_pra - oneshot_pra,
        "oneshot_examples": len(results["oneshot"]),
        "decomposed_examples": len(results["decomposed"])
    }
    
    return results

def main():
    ap = argparse.ArgumentParser(description="Run one-shot baseline comparison")
    ap.add_argument("--examples-root", default="artifacts/examples", type=Path)
    ap.add_argument("--config", default="atomic_edits/m3/config.yaml", type=Path)
    ap.add_argument("--calibration", default="artifacts/m3/calibration.yml", type=Path)
    ap.add_argument("--output", default="artifacts/m3/oneshot_comparison.json", type=Path)
    args = ap.parse_args()
    
    results = evaluate_oneshot(
        args.examples_root,
        args.config,
        args.calibration
    )
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"One-shot PRA: {results['comparison']['oneshot_pra']:.3f}")
    print(f"Decomposed PRA: {results['comparison']['decomposed_pra']:.3f}")
    print(f"Delta (Decomposed - One-shot): {results['comparison']['delta_pra']:+.3f}")

if __name__ == "__main__":
    main()