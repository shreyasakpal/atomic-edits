# --------------------------- cli_region.py ---------------------------
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, List  

import torch
from PIL import Image

from atomic_edits.region.ovd_masker import OVDMasker
from atomic_edits.region.gdino_sam_masker import GDinoSAMMasker
from atomic_edits.editor.region_edit import (
    load_pipelines,
    masked_property_edit,
    inpaint_edit,
)

# --------------------------- helpers ---------------------------

def _order_key(s: Dict[str, Any]) -> int:
    try:
        return int(s.get("order", 1))
    except Exception:
        return 1

def _is_remove_like(a: Optional[str]) -> bool:
    return (a or "").lower() in {"remove", "erase", "delete", "replace"}

_COLOR_WORDS = {
    "blue","red","green","yellow","purple","orange","black","white",
    "gray","grey","brown","pink","cyan","magenta","beige","maroon",
    "navy","teal","gold","silver"
}
def _color_word(step: Dict[str, Any]) -> Optional[str]:
    val = step.get("value")
    if isinstance(val, str) and val.lower() in _COLOR_WORDS:
        return val.lower()
    return None

def _coerce_color_edit(step: Dict[str, Any]) -> Dict[str, Any]:
    a = (step.get("action") or "").lower()
    val = step.get("value")
    if a == "replace" and isinstance(val, str) and val.lower() in _COLOR_WORDS:
        step["action"] = "change_color"
        step["attribute"] = "color"
    return step

def _build_phrase(step: Dict[str, Any]) -> str:
    obj = (step.get("object") or "").strip()
    if not obj:
        return "object"
    obj_l = obj.lower()
    synonyms = {
        "billboard": "billboard screen",
        "logo": "logo",
        "mug": "coffee cup",
        "cup": "coffee cup",
        "shirt": "shirt",
        "sky": "sky",
    }
    for k, v in synonyms.items():
        if k in obj_l:
            return v
    return obj

def _to_prompt(step: Dict[str, Any]) -> str:
    a = (step.get("action") or "").lower()
    obj = step.get("object")
    attr = step.get("attribute")
    val = step.get("value")

    if a in {"change_color", "recolor", "tint"}:
        return f"make the {obj if obj else 'object'} {val if val else 'the target color'}"
    if _is_remove_like(a) and a != "replace":
        return f"remove the {obj if obj else 'object'}"
    if a == "replace":
        return f"replace the {obj if obj else 'object'} with {val if val else 'the new item'}"
    if a in {"increase", "decrease"} and attr:
        direction = "increase" if a == "increase" else "decrease"
        scope = f"of the {obj}" if obj else "overall"
        return f"{direction} {attr} {scope}"

    parts = []
    if a: parts.append(a)
    if obj: parts.append(str(obj))
    if attr and val: parts.append(f"{attr} to {val}")
    return " ".join(parts) if parts else "apply the edit"

def _save_mask_debug(img: Image.Image, mask: Image.Image, path: Path):
    import numpy as np
    # Always align the mask to the image being visualized
    maskL = mask.convert("L").resize(img.size, Image.NEAREST)
    a = np.array(img)
    m = (np.array(maskL) > 0)

    # Broadcast mask to channels (H,W,1) if needed
    if m.ndim == 2:
        m = m[..., None]

    # Dim the background, keep masked region unchanged
    out = np.where(m, a, (a * 0.35).astype(a.dtype))
    Image.fromarray(out).save(path)


def _read_steps_tolerant(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Accept any of: sub_instructions (parse JSON), steps (plan JSON), edits/instructions.
    Normalize into a list of dicts with keys we use downstream.
    """
    if "sub_instructions" in plan:
        return list(plan["sub_instructions"])

    if "steps" in plan:
        out = []
        for i, s in enumerate(plan["steps"], 1):
            out.append({
                "order": s.get("order", i),
                "prompt": s.get("prompt"),
                "mask_phrase": s.get("mask_phrase"),
                "requires_region": s.get("requires_region", bool(s.get("mask_phrase"))),
                "action": s.get("action"),        # keep if present
                "object": s.get("object"),        # keep if present
                "value": s.get("value"),
                "attribute": s.get("attribute"),
            })
        return out

    for key in ("edits", "instructions"):
        if key in plan:
            out = []
            for i, s in enumerate(plan[key], 1):
                out.append({
                    "order": s.get("order", i),
                    "prompt": s.get("prompt") or s.get("instruction") or s.get("text"),
                    "mask_phrase": s.get("mask_phrase") or s.get("region"),
                    "requires_region": s.get("requires_region", bool(s.get("mask_phrase") or s.get("region"))),
                    "action": s.get("action"),
                    "object": s.get("object"),
                    "value": s.get("value"),
                    "attribute": s.get("attribute"),
                })
            return out
    return []

def _prefer_prompt(step: Dict[str, Any]) -> str:
    return (step.get("prompt") or _to_prompt(step) or "apply the edit").strip()

def _prefer_phrase(step: Dict[str, Any]) -> Optional[str]:
    return step.get("mask_phrase") or _build_phrase(step)

def _fit_long_side(img: Image.Image, max_long=1024, min_short=512):
    w, h = img.size
    long_side = max(w, h); short_side = min(w, h)
    s = min(1.0, max_long / long_side)
    sw, sh = int(w * s), int(h * s)
    if min(sw, sh) < min_short:
        k = min_short / min(sw, sh)
        sw, sh = int(sw * k), int(sh * k)
    return img.resize((sw, sh), Image.LANCZOS), (w, h)

# --------------------------- CLI main ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Region-aware sequence runner")
    ap.add_argument("--image", required=True)
    ap.add_argument("--json", required=True, help="Parse or plan JSON")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--guidance-scale", type=float, default=5.0)
    ap.add_argument("--image-guidance-scale", type=float, default=2.2)
    ap.add_argument("--mask-thresh", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-ip2p-color", action="store_true")
    ap.add_argument("--masker", choices=["gdino_sam", "ovd"], default="gdino_sam")
    ap.add_argument("--freeze-mask", action="store_true")
    ap.add_argument("--mask-path", type=str, default=None)
    ap.add_argument("--gdino-cfg", type=str, default="checkpoints/GroundingDINO_SwinT_OGC.py")
    ap.add_argument("--gdino-weights", type=str, default="checkpoints/groundingdino_swint_ogc.pth")
    ap.add_argument("--sam-weights", type=str, default="checkpoints/sam_vit_b.pth")
    ap.add_argument("--det-box-thresh", type=float, default=0.35)
    ap.add_argument("--det-text-thresh", type=float, default=0.25)
    ap.add_argument("--save-mask-debug", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    log: Dict[str, Any] = {"image": args.image, "plan": args.json, "seed": args.seed, "steps": []}

    # Load + normalize image size
    img = Image.open(args.image).convert("RGB")
    current, orig_size = _fit_long_side(img, max_long=1024, min_short=512)

    # Load plan/parse JSON
    plan = json.loads(Path(args.json).read_text())
    steps_seq = sorted(_read_steps_tolerant(plan), key=_order_key)
    if not steps_seq:
        print("[warn] no steps found; writing input as final without edits.")
        final_path = outdir / "final.png"
        img.save(final_path)
        log["final_image"] = str(final_path)
        (outdir / "edit_log.json").write_text(json.dumps(log, indent=2))
        print(f"[done] final -> {final_path}")
        return

    # Init pipelines + masker
    pipes = load_pipelines()
    try:
        if args.masker == "gdino_sam":
            masker = GDinoSAMMasker(
                gdino_cfg=args.gdino_cfg,
                gdino_weights=args.gdino_weights,
                sam_weights=args.sam_weights,
            )
        else:
            masker = OVDMasker()
    except Exception as e:
        print(f"[warn] GDINO/SAM init failed ({e}); falling back to OWL-ViT.")
        masker = OVDMasker()

    prev_mask = None

    for i, raw_step in enumerate(steps_seq, start=1):
        step = _coerce_color_edit(dict(raw_step))

        # Decide regional and phrase
        needs_region = bool(step.get("requires_region") or step.get("mask_phrase"))
        phrase = _prefer_phrase(step) if needs_region else None

        # Build prompt & detect remove-like even without action
        prompt = _prefer_prompt(step)
        act = (step.get("action") or "").lower()
        is_remove = _is_remove_like(act) or prompt.lower().startswith(("remove ", "erase ", "delete "))

        mask = None
        det_info = None
        det = None

        if needs_region:
            # special-case background: invert previous mask if we have one
            if phrase and phrase.lower().strip() in {"background", "bg"} and prev_mask is not None:
                mask = Image.eval(prev_mask, lambda p: 255 - p)
            else:
                if args.mask_path:
                    mask = Image.open(args.mask_path).convert("L").resize(current.size, Image.NEAREST)
                elif args.freeze_mask and log.get("frozen_mask_used"):
                    if log.get("frozen_mask_path"):
                        mask = Image.open(log["frozen_mask_path"]).convert("L").resize(current.size, Image.NEAREST)
                else:
                    if isinstance(masker, GDinoSAMMasker):
                        mask, det = masker.phrase_to_mask(
                            current, phrase,
                            box_thresh=args.det_box_thresh,
                            text_thresh=args.det_text_thresh,
                        )
                    else:
                        mask, det = masker.phrase_to_mask(current, phrase, score_thresh=args.mask_thresh)
            
            if mask is not None and mask.size != current.size:
                mask = mask.resize(current.size, Image.NEAREST)

            if mask is None or mask.getbbox() is None:
                print(f"[warn] step {i}: requires_region=True but no mask for '{phrase}'. Skipping.")
                continue

            if args.freeze_mask and not log.get("frozen_mask_used"):
                frozen_path = outdir / "frozen_mask.png"
                mask.save(frozen_path)
                log["frozen_mask_used"] = True
                log["frozen_mask_path"] = str(frozen_path)

            if args.save_mask_debug:
                _save_mask_debug(current, mask, outdir / f"mask_step_{i:02}.png")

            det_info = {"phrase": phrase}
            if det is not None:
                det_info["box"] = getattr(det, "box", None)
                det_info["score"] = getattr(det, "score", None)

        color = _color_word(step)

        if is_remove and act != "change_color":
            next_img = inpaint_edit(
                pipes, current,
                mask if mask is not None else Image.new("L", current.size, 0),
                prompt, steps=25, guidance=7.0, seed=args.seed
            )
        else:
            next_img = masked_property_edit(
                pipes, current, prompt,
                mask=mask if needs_region else None,
                steps=args.steps,
                guidance=args.guidance_scale,
                image_guidance=args.image_guidance_scale,
                seed=args.seed,
                color_word=color,
                use_ip2p_color=args.use_ip2p_color,
            )

        step_path = outdir / f"step_{i:02d}.png"
        next_img.save(step_path)
        log["steps"].append({
            "order": i,
            "prompt": prompt,
            "requires_region": needs_region,
            "mask_phrase": phrase,
            "det": det_info,
            "image": str(step_path),
        })
        current = next_img
        if needs_region:
            prev_mask = mask
        if hasattr(torch, "mps"):
            try: torch.mps.empty_cache()
            except Exception: pass

    final = current.resize(orig_size, Image.LANCZOS)
    final_path = outdir / "final.png"
    final.save(final_path)
    log["final_image"] = str(final_path)
    (outdir / "edit_log.json").write_text(json.dumps(log, indent=2))
    print(f"[done] final -> {final_path}")

if __name__ == "__main__":
    main()
# --------------------------- end file ----------------------------------------
