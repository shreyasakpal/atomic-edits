# atomic_edits/pipeline/cli_e2e.py
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Dict, Any, List

# Parse providers (already in your repo)
from ..parser.parse import parse_with_ollama, parse_with_openai

def _to_dict(obj):
    if is_dataclass(obj): return asdict(obj)
    if hasattr(obj, "__dict__"): return obj.__dict__
    return obj

def _prompt_from_step(s):
    act = s["action"]; obj = s["object"]; val = s.get("value")
    q = " ".join(s.get("qualifiers", [])).strip()
    target = (obj + (f" {q}" if q else "")).strip()

    if act == "change_color":
        return f"make the {target} {val}"
    if act == "remove":
        # normalize common text/label cases
        if obj == "text":
            any_all = ("any" in [x.lower() for x in s.get("qualifiers", [])])
            return "remove any text" if any_all else "remove the text"
        return f"remove the {target}"
    if act == "blur":
        adv = str(val).lower() if isinstance(val, str) else ""
        return f"{(adv + ' ') if adv else ''}blur the {target}"
    if act in ("increase","decrease"):
        attr = s.get("attribute", "brightness")
        verb = "increase" if act == "increase" else "decrease"
        return f"{verb} the {attr} of the {target}"
    return " ".join([act.replace("_"," "), "the", target] + (["to", str(val)] if val else []))


def _blur_strength(val) -> float:
    if isinstance(val, (int, float)): return float(val)
    lut = {"slightly":0.25, "lightly":0.25, "moderately":0.5, "strongly":0.8, "heavily":0.8}
    return lut.get(str(val).lower(), 0.25)

def _mask_phrase(s):
    # choose detector-friendly synonyms
    obj = s["object"].lower()
    if obj == "mug":
        return "coffee cup"
    if obj == "text":
        return "text"
    if obj == "background":
        return "background"
    return (obj + (" " + " ".join(s.get("qualifiers", [])) if s.get("qualifiers") else "")).strip()

def _build_plan(image: Path, parse_dict: Dict[str, Any]) -> Dict[str, Any]:
    steps: List[Dict[str, Any]] = parse_dict["sub_instructions"]
    # Optional polish: remove before recolor to reduce color bleed
    steps_sorted = sorted(steps, key=lambda s: (0 if s["action"]=="remove" else 1, s["order"]))
    plan_steps = []
    for s in steps_sorted:
        entry = {
            "order": s["order"],
            "prompt": _prompt_from_step(s),
            "requires_region": bool(s.get("requires_region", False)),
            "mask_phrase": _mask_phrase(s) if s.get("requires_region", False) else None,
            "det": {"detector": "gdino", "sam": "sam-hq", "threshold": 0.30}
        }
        if s["action"] == "blur":
            entry["strength"] = _blur_strength(s.get("value"))
        plan_steps.append(entry)
    return {"image": str(image), "plan": "", "seed": 42, "steps": plan_steps}

def _run_module(mod: str, args: List[str], env: Dict[str,str] | None = None):
    cmd = [sys.executable, "-m", mod] + args
    res = subprocess.run(cmd, env=env, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed ({res.returncode}): {' '.join(cmd)}")

def main():
    ap = argparse.ArgumentParser(description="Atomic Edits — end-to-end runner")
    ap.add_argument("-i","--image", required=True, help="Path to input image")
    ap.add_argument("-t","--text", required=True, help="Edit instruction text")
    ap.add_argument("-o","--outdir", default=None, help="Output directory (default: artifacts/e2e_<ts>)")
    ap.add_argument("-b","--backend", choices=["ollama","openai"], default="ollama")
    ap.add_argument("--ollama-model", default="llama3.1:8b-instruct-q4_K_M")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--openai-model", default="gpt-4o-mini")
    # editor params
    ap.add_argument("--steps", type=int, default=9)
    ap.add_argument("--guidance-scale", type=float, default=4.8)
    ap.add_argument("--image-guidance-scale", type=float, default=2.4)
    ap.add_argument("--mask-thresh", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--baseline", action="store_true", help="Also run no-mask baseline (cli_ip2p)")
    ap.add_argument("--reveal", action="store_true", help="Open output folder in Finder when done")
    args = ap.parse_args()

    image = Path(args.image).expanduser().resolve()
    if not image.exists():
        ap.error(f"Image not found: {image}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    outroot = Path(args.outdir) if args.outdir else Path("artifacts") / f"e2e_{ts}"
    parsedir = outroot / "parsed"
    plandir  = outroot / "plans"
    editsdir = outroot / "edits"
    for d in (parsedir, plandir, editsdir): d.mkdir(parents=True, exist_ok=True)

    # 1) Parse
    if args.backend == "ollama":
        res = parse_with_ollama(args.text, ollama_model=args.ollama_model, url=args.ollama_url)
    else:
        res = parse_with_openai(args.text, openai_model=args.openai_model)
    parse_dict = _to_dict(res)
    (parsedir / "parse.json").write_text(json.dumps(parse_dict, indent=2))

    # 2) Plan
    plan = _build_plan(image, parse_dict)
    plan_path = plandir / "plan.json"
    plan["plan"] = str(plan_path)
    plan_path.write_text(json.dumps(plan, indent=2))

    # 3) Region-guided editor (as module)
    env = os.environ.copy()
    env.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO","0.0")  # macOS MPS stability
    _run_module(
        "atomic_edits.editor.cli_region",
        [
            "--image", str(image),
            "--json", str(plan_path),                     # or str(parse_path) if you prefer
            "--outdir", str(editsdir),
            "--steps", str(args.steps),
            "--guidance-scale", str(args.guidance_scale),
            "--image-guidance-scale", str(args.image_guidance_scale),
            "--mask-thresh", str(args.mask_thresh),
            "--seed", str(args.seed),

            # >>> forward your detector/masker settings <<<
            "--masker", "gdino_sam",
            "--gdino-cfg", "checkpoints/GroundingDINO_SwinT_OGC.py",
            "--gdino-weights", "checkpoints/groundingdino_swint_ogc.pth",
            "--sam-weights", "checkpoints/sam_vit_b.pth",
            "--det-box-thresh", "0.35",
            "--det-text-thresh", "0.25",
            "--freeze-mask",
            "--save-mask-debug",
        ],
        env=env
    )

    # 4) Optional: no-mask baseline
    if args.baseline:
        # join human-readable prompts from parsed steps
        prompt = "; ".join(_prompt_from_step(s) for s in parse_dict["sub_instructions"])
        _run_module(
            "atomic_edits.editor.cli_ip2p",
            [
                "--image", str(image),
                "--prompt", prompt,
                "--out", str(editsdir / "baseline_nomask_01.png"),
                "--steps", str(args.steps),
                "--guidance-scale", str(args.guidance_scale),
                "--image-guidance-scale", str(args.image_guidance_scale),
                "--seed", str(args.seed),
            ],
            env=env
        )

    if args.reveal and sys.platform == "darwin":
        subprocess.run(["open", str(outroot)], check=False)

if __name__ == "__main__":
    main()
