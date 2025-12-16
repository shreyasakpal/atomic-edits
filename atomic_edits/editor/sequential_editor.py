from pathlib import Path
from typing import Dict, List
import json, torch
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from .prompt_builder import build_edit_prompt

def _device_dtype():
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32

def load_instruct_pix2pix(model_id: str = "timbrooks/instruct-pix2pix"):
    device, dtype = _device_dtype()
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_id, torch_dtype=dtype, safety_checker=None
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    return pipe, device

def _sanitize_size(img: Image.Image, max_side: int = 768):
    w, h = img.size
    scale = min(max_side / max(w, h), 1.0)
    return img if scale == 1.0 else img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)

def run_sequence(
    image_path: str,
    steps: List[Dict],
    outdir: str = "artifacts/edits/run1",
    model_id: str = "timbrooks/instruct-pix2pix",
    seed: int = 42,
    num_inference_steps: int = 20,
    guidance_scale: float = 7.0,
    image_guidance_scale: float = 1.5,
):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    pipe, device = load_instruct_pix2pix(model_id)
    gen = torch.Generator(device=device).manual_seed(seed)

    img = Image.open(image_path).convert("RGB")
    img = _sanitize_size(img)

    log = {"image": image_path, "model": model_id, "seed": seed, "steps": []}

    for step in sorted(steps, key=lambda s: int(s.get("order", 1))):
        prompt = build_edit_prompt(step)
        res = pipe(
            prompt=prompt,
            image=img,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            image_guidance_scale=image_guidance_scale,
            generator=gen,
        )
        img = res.images[0]
        step_idx = int(step.get("order", 1))
        pth = out / f"step_{step_idx:02d}.png"
        img.save(pth)
        log["steps"].append({"order": step_idx, "prompt": prompt, "image": str(pth)})

    final_path = out / "final.png"
    img.save(final_path)
    log["final_image"] = str(final_path)
    (out / "edit_log.json").write_text(json.dumps(log, indent=2))
    return str(final_path)
