import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import yaml

from atomic_edits.editor.region_edit import load_pipelines, masked_property_edit, inpaint_edit
from atomic_edits.m3.evaluator.vqa_blip2 import BLIP2YesNo
from atomic_edits.m3.evaluator.clip_checks import CLIPAttribute
from atomic_edits.m3.evaluator.aggregation import Thresholds, decide_pass_polarity


def load_config(cfg_path="atomic_edits/m3/config.yaml"):
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def load_thresholds(cfg, cal_path="artifacts/m3/calibration.yml"):
    th = Thresholds(
        vqa_yes_threshold=cfg["calibration"]["vqa_yes_threshold"],
        clip_margin_threshold=cfg["calibration"]["clip_margin_threshold"],
    )
    if Path(cal_path).exists():
        cal = yaml.safe_load(open(cal_path))
        th.vqa_yes_threshold = float(cal["vqa_yes_threshold"])
        th.clip_margin_threshold = float(cal["clip_margin_threshold"])
    return th


def build_prompt(sub_dict):
    """Build edit prompt from sub-instruction dict"""
    text = sub_dict.get("text", "")
    if text:
        return text
    
    action = (sub_dict.get("action") or "").lower()
    obj = sub_dict.get("object", "object")
    val = sub_dict.get("value")
    
    if action in {"change_color", "recolor"} and val:
        return f"make the {obj} {val}"
    elif action == "remove":
        return f"remove the {obj}"
    else:
        return f"{action} the {obj}"


def repair_edit(pipes, image, sub_dict, mask, attempt, seed=42):
    """
    Re-execute edit with escalating parameters.
    attempt: 1, 2, 3, ...
    """
    prompt = build_prompt(sub_dict)
    action = (sub_dict.get("action") or "").lower()
    
    # Escalate parameters based on attempt
    strength = 1.0 + (attempt - 1) * 0.25  # 1.0, 1.25, 1.5, 1.75
    guidance = 5.0 + (attempt - 1) * 1.5   # 5.0, 6.5, 8.0, 9.5
    steps = 10 + (attempt - 1) * 5         # 10, 15, 20, 25
    
    # Change seed on later attempts
    repair_seed = seed + attempt * 10 if attempt > 2 else seed
    
    print(f"    Attempt {attempt}: guidance={guidance:.1f}, steps={steps}, seed={repair_seed}")
    
    # Execute repair
    if action in {"remove", "erase", "delete"}:
        return inpaint_edit(
            pipes, image,
            mask or Image.new("L", image.size, 0),
            prompt,
            steps=min(30, steps * 2),
            guidance=guidance * 1.2,
            seed=repair_seed
        )
    else:
        return masked_property_edit(
            pipes, image, prompt,
            mask=mask,
            steps=steps,
            guidance=guidance,
            image_guidance=2.2 * strength,
            seed=repair_seed,
        )


def main():
    # Load config
    cfg = load_config()
    device = cfg.get("device", "cuda")
    
    # Load M3 results
    results_path = Path("artifacts/m3/m3_results.jsonl")
    if not results_path.exists():
        print(f"❌ ERROR: M3 results not found at {results_path}")
        print("   Run M3 evaluation first: python -m atomic_edits.m3.evaluate")
        return
    
    # Read all results and filter failures
    failures = []
    with open(results_path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("decision") == "FAIL":
                failures.append(row)
    
    if not failures:
        print("✅ No failed edits to repair (all passed!)")
        return
    
    print(f"🔧 Found {len(failures)} failed edits to repair")
    
    # Load evaluators
    print("[Repair] Loading evaluators...")
    vqa = BLIP2YesNo(cfg["vqa"]["model_name"], device, cfg["vqa"]["max_new_tokens"])
    clip = CLIPAttribute(cfg["clip"]["model_name"], cfg["clip"]["pretrained"], device)
    th = load_thresholds(cfg)
    
    # Load editor
    print("[Repair] Loading editor pipelines...")
    pipes = load_pipelines()
    
    # Attempt repairs
    repair_log = []
    max_attempts = 3
    
    for fail_entry in tqdm(failures, desc="Repairing failures"):
        ex_id = fail_entry["example_id"]
        sub_id = fail_entry["sub_id"]
        
        print(f"\n[{ex_id} / {sub_id}] Repairing...")
        
        # Load example metadata
        ex_dir = Path("artifacts/examples") / ex_id
        meta_path = ex_dir / "sub_instructions.json"
        
        if not meta_path.exists():
            print(f"  ⚠️  Metadata not found, skipping")
            continue
        
        meta = json.load(open(meta_path))
        sub = [s for s in meta["sub_instructions"] if s["id"] == sub_id][0]
        
        # Load current image
        edited_path = ex_dir / "edited.png"
        if not edited_path.exists():
            edited_path = ex_dir / "final.png"
        if not edited_path.exists():
            print(f"  ⚠️  Edited image not found, skipping")
            continue
        
        current = Image.open(edited_path).convert("RGB")
        
        # Load mask if exists
        mask_path = ex_dir / "masks" / f"{sub_id}.png"
        mask = Image.open(mask_path).convert("L") if mask_path.exists() else None
        
        # Repair attempts
        success = False
        attempts_made = 0
        
        for attempt in range(1, max_attempts + 1):
            attempts_made = attempt
            
            # Re-execute edit
            try:
                repaired = repair_edit(pipes, current, sub, mask, attempt, seed=42)
            except Exception as e:
                print(f"    ❌ Edit failed: {e}")
                continue
            
            # Re-evaluate
            phrase = sub.get("target_phrase") or sub["text"]
            expect_present = sub.get("expect_present", True)
            question = sub["vqa_question"]
            
            vqa_out = vqa.ask(repaired, question,
                            cfg["vqa"]["yes_words"], cfg["vqa"]["no_words"])
            clip_out = clip.score_phrase(repaired,
                                        cfg["clip"]["positive_templates"],
                                        cfg["clip"]["negative_templates"],
                                        phrase)
            
            passed, rule = decide_pass_polarity(
                vqa_yes_conf=vqa_out["yes_conf"],
                clip_margin=clip_out["margin"],
                thresholds=th,
                expect_present=expect_present,
            )
            
            print(f"      VQA: {vqa_out['yes_conf']:.2f}, CLIP: {clip_out['margin']:.3f} → {'PASS' if passed else 'FAIL'}")
            
            if passed:
                success = True
                # Save repaired image
                repaired_path = ex_dir / f"repaired_{sub_id}_attempt{attempt}.png"
                repaired.save(repaired_path)
                print(f"    ✅ SUCCESS! Saved → {repaired_path}")
                break
        
        repair_log.append({
            "example_id": ex_id,
            "sub_id": sub_id,
            "text": sub["text"],
            "original_decision": "FAIL",
            "repair_success": success,
            "attempts_made": attempts_made,
        })
    
    # Save repair log
    log_path = Path("artifacts/m3/repair_log.json")
    log_path.parent.mkdir(exist_ok=True, parents=True)
    
    with open(log_path, "w") as f:
        json.dump(repair_log, f, indent=2)
    
    # Summary
    success_count = sum(1 for r in repair_log if r["repair_success"])
    total = len(repair_log)
    
    print("\n" + "="*60)
    print("🔧 REPAIR SUMMARY")
    print("="*60)
    print(f"Total failures:  {total}")
    print(f"Repaired:        {success_count} ({success_count/total*100:.1f}%)")
    print(f"Still failing:   {total - success_count}")
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    main()