from __future__ import annotations
import argparse, csv, os, json, yaml
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from .evaluator.vqa_blip2 import BLIP2YesNo
from .evaluator.clip_checks import CLIPAttribute

REMOVAL_TOKENS = ("remove", "erase", "delete", "without", "hide", "get rid of", "crop out", "clean", "obliterate", "blur out", "cover")

def infer_expect_present(text: str, default: bool = True) -> bool:
    t = (text or "").lower()
    return not any(tok in t for tok in REMOVAL_TOKENS) if text else default

def _load_cfg(cfg_path: str):
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

def run(
    cfg_path: str,
    examples_root: str,
    labels_csv: str,
    out_path: str,
):
    cfg = _load_cfg(cfg_path)
    device = cfg.get("device", "cuda")

    vqa  = BLIP2YesNo(cfg["vqa"]["model_name"], device, cfg["vqa"]["max_new_tokens"])
    clip = CLIPAttribute(cfg["clip"]["model_name"], cfg["clip"]["pretrained"], device)

    rows = list(csv.DictReader(open(labels_csv)))
    # search grid
    best = {"f1": -1.0}
    vqa_grid  = cfg.get("calibration", {}).get("grid_vqa",  [0.50,0.55,0.60,0.65,0.70,0.75])
    clip_grid = cfg.get("calibration", {}).get("grid_clip", [0.00,0.02,0.05,0.08,0.10,0.15])

    for t_vqa in vqa_grid:
        for t_clip in clip_grid:
            tp = fp = fn = 0
            for r in rows:
                ex  = r["example_id"]; sid = r["sub_id"]; gt = int(r["gt_pass"])
                ex_dir = Path(examples_root) / ex
                meta = json.load(open(ex_dir / "sub_instructions.json"))
                sub  = [s for s in meta["sub_instructions"] if s["id"] == sid][0]

                question = sub["vqa_question"]
                phrase   = sub.get("target_phrase") or sub["text"]
                expect_present = (
                    bool(int(r["expect_present"])) if "expect_present" in r
                    else sub.get("expect_present", infer_expect_present(sub.get("text")))
                )

                # Edited image name is configurable; default falls back to edited.png
                edited_name = cfg.get("paths", {}).get("edited_name", "edited.png")
                edited = Image.open(ex_dir / edited_name).convert("RGB")

                vqa_out  = vqa.ask(edited, question, cfg["vqa"]["yes_words"], cfg["vqa"]["no_words"])
                clip_out = clip.score_phrase(edited, cfg["clip"]["positive_templates"], cfg["clip"]["negative_templates"], phrase)

                # Polarity-aware scoring: if expect_present=False, invert VQA and flip CLIP margin sign
                vqa_score = vqa_out["yes_conf"] if expect_present else (1.0 - vqa_out["yes_conf"])
                clip_ok   = (clip_out["margin"] >=  t_clip) if expect_present else (clip_out["margin"] <= -t_clip)

                pred = int((vqa_score >= t_vqa) and clip_ok)
                if pred == 1 and gt == 1: tp += 1
                elif pred == 1 and gt == 0: fp += 1
                elif pred == 0 and gt == 1: fn += 1

            prec = tp / (tp + fp + 1e-9)
            rec  = tp / (tp + fn + 1e-9)
            f1   = 2 * prec * rec / (prec + rec + 1e-9)
            if f1 > best["f1"]:
                best = {"t_vqa": t_vqa, "t_clip": t_clip, "f1": f1, "precision": prec, "recall": rec}

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(
        {
            "vqa_yes_threshold": float(best["t_vqa"]),
            "clip_margin_threshold": float(best["t_clip"]),
            "dev_f1": float(best["f1"]),
            "dev_precision": float(best["precision"]),
            "dev_recall": float(best["recall"]),
        },
        open(out_path, "w")
    )
    print(f"[M3] Saved calibration → {out_path} :: {best}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="atomic_edits/m3/config.yaml")
    ap.add_argument("--examples-root", default="artifacts/examples")
    ap.add_argument("--labels-csv", default="artifacts/m3/calib_labels.csv")
    ap.add_argument("--out", default="artifacts/m3/calibration.yml")
    args = ap.parse_args()
    run(args.config, args.examples_root, args.labels_csv, args.out)

if __name__ == "__main__":
    main()
