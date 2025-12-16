from __future__ import annotations
import argparse, json, os, yaml
from pathlib import Path
from typing import Dict, Iterable, Optional
from PIL import Image
from tqdm import tqdm

from .evaluator.vqa_blip2 import BLIP2YesNo
from .evaluator.clip_checks import CLIPAttribute
from .evaluator.aggregation import Thresholds, decide_pass_polarity
from .evaluator.preservation_metrics import PreservationMetrics  # NEW

REMOVAL_TOKENS = ("remove", "erase", "delete", "without", "hide", "get rid of", "crop out", "clean", "obliterate", "blur out", "cover")

def infer_expect_present(text: str, default: bool = True) -> bool:
    t = (text or "").lower()
    return not any(tok in t for tok in REMOVAL_TOKENS) if text else default

def _load_cfg(cfg_path: str):
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

def _maybe_load_thresholds(cfg: Dict, cal_path: Optional[str]) -> Thresholds:
    th = Thresholds(
        vqa_yes_threshold=cfg["calibration"]["vqa_yes_threshold"],
        clip_margin_threshold=cfg["calibration"]["clip_margin_threshold"],
    )
    # allow artifact override
    if cal_path and Path(cal_path).exists():
        cal = yaml.safe_load(open(cal_path))
        th.vqa_yes_threshold = float(cal["vqa_yes_threshold"])
        th.clip_margin_threshold = float(cal["clip_margin_threshold"])
    return th

def _find_mask_for_sub(ex_dir: Path, sub_id: str) -> Optional[str]:
    # Extract number from sub_id (e.g., "s1" -> 1)
    try:
        num = int(sub_id[1:]) if sub_id.startswith('s') else int(sub_id)
    except:
        num = 1
    
    candidates = [
        ex_dir / "masks" / f"{sub_id}.png",
        ex_dir / f"{sub_id}_mask.png",
        ex_dir / f"mask_step_{num:02d}.png",
        ex_dir / "masks" / f"{sub_id}.jpg",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None

def _iter_examples(root: Path) -> Iterable[Path]:
    for p in sorted(root.iterdir()):
        if p.is_dir() and (p / "sub_instructions.json").exists():
            yield p

def run(
    cfg_path: str,
    examples_root: str,
    out_root: str,
    edited_name: str = "edited.png",
    calibration_yml: Optional[str] = None,
    compute_preservation: bool = True,  # NEW parameter
):
    cfg = _load_cfg(cfg_path)
    device = cfg.get("device", "cuda")
    out_root = Path(out_root); out_root.mkdir(parents=True, exist_ok=True)

    vqa  = BLIP2YesNo(cfg["vqa"]["model_name"], device, cfg["vqa"]["max_new_tokens"])
    clip = CLIPAttribute(cfg["clip"]["model_name"], cfg["clip"]["pretrained"], device)
    th   = _maybe_load_thresholds(cfg, calibration_yml)
    
    # NEW: Initialize preservation metrics
    preservation = PreservationMetrics(device) if compute_preservation else None

    results_path = out_root / "m3_results.jsonl"
    n_rows = 0

    with open(results_path, "w") as fout:
        for ex_dir in tqdm(list(_iter_examples(Path(examples_root))), desc="[M3] Evaluating"):
            meta = json.load(open(ex_dir / "sub_instructions.json"))
            
            # Load original image for preservation metrics
            original = Image.open(ex_dir / "original.png").convert("RGB")
            
            edited_p = ex_dir / edited_name
            if not edited_p.exists():
                # fallbacks that are common in M2 outputs
                for alt in ("final.png", "sequential_final.png", "output.png"):
                    if (ex_dir / alt).exists():
                        edited_p = ex_dir / alt
                        break
            edited = Image.open(edited_p).convert("RGB")

            # NEW: Compute global preservation metrics once per example
            global_preservation = {}
            if preservation:
                global_pres = preservation.compute_outside_mask(original, edited, mask=None)
                global_preservation = {
                    "ssim_global": global_pres["ssim_global"],
                    "lpips_global": global_pres["lpips_global"]
                }

            for sub in meta["sub_instructions"]:
                phrase   = sub.get("target_phrase") or sub["text"]
                expect_present = sub.get("expect_present", infer_expect_present(sub.get("text")))
                question = sub["vqa_question"]
                mask_p   = _find_mask_for_sub(ex_dir, sub["id"])

                vqa_out  = vqa.ask(edited, question, cfg["vqa"]["yes_words"], cfg["vqa"]["no_words"])
                clip_out = clip.score_phrase(edited, cfg["clip"]["positive_templates"], cfg["clip"]["negative_templates"], phrase)

                passed, rule = decide_pass_polarity(
                    vqa_yes_conf=vqa_out["yes_conf"],
                    clip_margin=clip_out["margin"],
                    thresholds=th,
                    expect_present=expect_present,
                )
                
                # NEW: Compute preservation outside mask if mask exists
                preservation_metrics = {}
                if preservation and mask_p:
                    try:
                        mask_img = Image.open(mask_p).convert("L")
                        pres = preservation.compute_outside_mask(original, edited, mask_img)
                        preservation_metrics = {
                            "ssim_outside": pres["ssim_outside"],
                            "lpips_outside": pres["lpips_outside"],
                            "coverage": pres["coverage"]
                        }
                    except Exception as e:
                        print(f"Warning: Failed to compute preservation for {ex_dir.name}/{sub['id']}: {e}")

                row = {
                    "example_id": ex_dir.name,
                    "sub_id": sub["id"],
                    "text": sub["text"],
                    "target_phrase": phrase,
                    "expect_present": bool(expect_present),
                    "question": question,
                    "vqa_answer": vqa_out["answer"],
                    "vqa_yes_conf": float(vqa_out["yes_conf"]),
                    "clip_pos": float(clip_out["pos"]),
                    "clip_neg": float(clip_out["neg"]),
                    "clip_margin": float(clip_out["margin"]),
                    "decision": "PASS" if passed else "FAIL",
                    "rule": rule,
                    "mask_path": str(mask_p) if mask_p else None,
                    "edited_path": str(edited_p),
                    **global_preservation,  # Add global metrics
                    **preservation_metrics   # Add masked preservation if available
                }
                fout.write(json.dumps(row) + "\n")
                n_rows += 1

    print(f"[M3] Wrote {n_rows} rows → {results_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="atomic_edits/m3/config.yaml")
    ap.add_argument("--examples-root", default="artifacts/examples")
    ap.add_argument("--out-root", default="artifacts/m3")
    ap.add_argument("--edited-name", default="edited.png")
    ap.add_argument("--calibration", default="artifacts/m3/calibration.yml")
    ap.add_argument("--skip-preservation", action="store_true", 
                    help="Skip preservation metrics computation")
    args = ap.parse_args()
    
    run(
        args.config, 
        args.examples_root, 
        args.out_root, 
        args.edited_name, 
        args.calibration,
        compute_preservation=not args.skip_preservation
    )

if __name__ == "__main__":
    main()