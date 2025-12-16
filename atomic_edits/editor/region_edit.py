# --------------------------- region_edit.py (FULL) ---------------------------
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import numpy as np
import cv2
from PIL import Image, ImageFilter

from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
    StableDiffusionInpaintPipeline,
)
from diffusers.schedulers import DPMSolverMultistepScheduler


# --------------------------- device / dtype helpers ---------------------------

def _pick_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def _to_dtype():
    # float16 on MPS/cuda; float32 on CPU for stability
    if torch.backends.mps.is_available() or torch.cuda.is_available():
        return torch.float16
    return torch.float32


# --------------------------- pipelines container ---------------------------

@dataclass
class Pipelines:
    ip2p: StableDiffusionInstructPix2PixPipeline
    inpaint: Optional[StableDiffusionInpaintPipeline]
    device: str


# def load_pipelines() -> Pipelines:
#     """
#     Load IP2P immediately; load the inpaint pipeline lazily (only if a step needs it).
#     Enable memory-saving features for M-series Macs / limited VRAM.
#     """
#     device = _pick_device()
#     dtype = _to_dtype()

#     ip2p = StableDiffusionInstructPix2PixPipeline.from_pretrained(
#         "timbrooks/instruct-pix2pix", torch_dtype=dtype, safety_checker=None
#     ).to(device)

#     # Memory savers
#     ip2p.enable_attention_slicing()
#     ip2p.enable_vae_slicing()
#     ip2p.enable_vae_tiling()
#     # A smoother scheduler often helps keep results natural on MPS
#     try:
#         ip2p.scheduler = DPMSolverMultistepScheduler.from_config(ip2p.scheduler.config)
#     except Exception:
#         pass

#     return Pipelines(ip2p=ip2p, inpaint=None, device=device)

def load_pipelines() -> Pipelines:
    device = _pick_device()  # "mps" on your Mac
    # fp32 on CPU/MPS for stability; fp16 only on CUDA
    dtype = torch.float32 if device in ("mps", "cpu") else torch.float16

    ip2p = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix",
        torch_dtype=dtype,
        safety_checker=None,
    ).to(device)

    # memory savers
    ip2p.enable_attention_slicing()
    ip2p.vae.enable_slicing()  # prefer module-level call

    # smoother scheduler on MPS often helps
    try:
        ip2p.scheduler = DPMSolverMultistepScheduler.from_config(ip2p.scheduler.config)
    except Exception:
        pass

    # IMPORTANT: do not build the inpaint pipe here; keep it lazy
    return Pipelines(ip2p=ip2p, inpaint=None, device=device)


# --------------------------- image utilities ---------------------------

def _pil_to_mask(mask: Image.Image) -> Image.Image:
    return mask.convert("L") if mask.mode != "L" else mask


def _sanitize_pair(image: Image.Image, mask: Image.Image | None, max_side: int = 512):
    """Downscale the pair to max_side; keeps aspect ratio; resizes mask with NEAREST."""
    w, h = image.size
    scale = min(max_side / max(w, h), 1.0)
    if scale == 1.0:
        return image, mask
    new = (int(w * scale), int(h * scale))
    image = image.resize(new, Image.LANCZOS)
    if mask is not None:
        mask = mask.resize(new, Image.NEAREST)
    return image, mask


def _feather(mask: Image.Image, radius: int = 8) -> Image.Image:
    return _pil_to_mask(mask).filter(ImageFilter.GaussianBlur(radius))


def composite_masked(base: Image.Image, edited: Image.Image, mask: Image.Image) -> Image.Image:
    """
    Blend edited into base using a single-channel mask.
    Ensure all sizes match; feather edges to avoid a visible rectangle.
    """
    base_size = base.size
    if edited.size != base_size:
        edited = edited.resize(base_size, Image.LANCZOS)
    if mask.size != base_size:
        mask = mask.resize(base_size, Image.NEAREST)
    fmask = _feather(mask, radius=8)
    return Image.composite(edited, base, fmask)


# --------------------------- photoreal recolor (classical) ---------------------------

# 1) increase feather a little for softer edge
def _feather(mask: Image.Image, radius: int = 10) -> Image.Image:
    return _pil_to_mask(mask).filter(ImageFilter.GaussianBlur(radius))

# basic hue lookup on OpenCV's [0,179] hue scale
_HUES = {
    "red": 0, "orange": 15, "yellow": 30, "green": 60,
    "teal": 90, "cyan": 90, "blue": 120, "navy": 120,
    "magenta": 150, "purple": 150, "pink": 165,
    "black": 0, "white": 0, "gray": 0, "grey": 0, "silver": 0, "gold": 30,
}

def recolor_masked_rgb(
    image: Image.Image,
    mask: Image.Image,
    color_word: str, sat_gain: float = 1.35, val_gain: float = 1.20):
    """
    Photoreal recolor inside mask using HSV; keeps original texture/lighting.
    """
    color_word = (color_word or "").lower()
    hue = _HUES.get(color_word, 120)  # default to blue
    rgb = np.array(image.convert("RGB"))
    m = np.array(_pil_to_mask(mask)) > 128

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0][m] = hue
    hsv[..., 1][m] = np.clip(hsv[..., 1][m] * sat_gain, 0, 255)
    hsv[..., 2][m] = np.clip(hsv[..., 2][m] * val_gain, 0, 255)

    recolored = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    recolored_pil = Image.fromarray(recolored)
    return composite_masked(image, recolored_pil, mask)


# --------------------------- IP2P recolor with crop + seamless blend ---------------------------

def _bbox_pad(bbox, W, H, pad=12):
    x1, y1, x2, y2 = bbox
    return max(0, x1 - pad), max(0, y1 - pad), min(W, x2 + pad), min(H, y2 + pad)


def _seamless_clone(base_rgb, edit_rgb, mask_L, bbox):
    # center for seamlessClone
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    base_bgr = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR)
    edit_bgr = cv2.cvtColor(edit_rgb, cv2.COLOR_RGB2BGR)
    mask = (np.array(mask_L) > 0).astype(np.uint8) * 255
    out = cv2.seamlessClone(edit_bgr, base_bgr, mask, (cx, cy), cv2.NORMAL_CLONE)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def ip2p_recolor_crop_blend(
    pipes: Pipelines,
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    steps: int = 4,
    guidance: float = 4.4,
    image_guidance: float = 3.2,
    seed: int = 42,
) -> Image.Image:
    """
    Recolor via IP2P on a tight crop, then seamless-blend into the original.
    Gentler params reduce stylization.
    """
    W, H = image.size
    bbox = mask.getbbox()
    if not bbox:
        return image
    bbox = _bbox_pad(bbox, W, H, pad=12)
    x1, y1, x2, y2 = bbox

    crop_img = image.crop(bbox)
    full_prompt = f"{prompt}, photorealistic, keep same material and reflections, minimal change"
    negative = "painting, illustration, oil paint, oversaturated, high contrast, artifacts"
    generator = torch.Generator(device=pipes.device).manual_seed(seed)

    try:
        pipes.ip2p.scheduler = DPMSolverMultistepScheduler.from_config(pipes.ip2p.scheduler.config)
    except Exception:
        pass

    edit = pipes.ip2p(
        prompt=full_prompt,
        negative_prompt=negative,
        image=crop_img,
        num_inference_steps=steps,
        guidance_scale=guidance,
        image_guidance_scale=image_guidance,
        generator=generator,
    ).images[0]

    if edit.size != crop_img.size:
        edit = edit.resize(crop_img.size, Image.LANCZOS)

    m_crop = _feather(mask.crop(bbox), radius=8)

    base_rgb = np.array(image.convert("RGB"))
    edit_rgb = base_rgb.copy()
    edit_rgb[y1:y2, x1:x2, :] = np.array(edit.convert("RGB"))
    blended_rgb = _seamless_clone(base_rgb, edit_rgb, m_crop, bbox)
    return Image.fromarray(blended_rgb)


# --------------------------- routed edit functions ---------------------------

def masked_property_edit(
    pipes: Pipelines,
    image: Image.Image,
    prompt: str,
    mask: Optional[Image.Image],
    steps: int = 10,
    guidance: float = 5.0,
    image_guidance: float = 2.2,
    seed: Optional[int] = 42,
    *,
    color_word: Optional[str] = None,
    use_ip2p_color: bool = False,
) -> Image.Image:
    """
    Property edit (e.g., color/brightness/sharpness).
    If a color_word + mask is provided, recolor with classical HSV (photoreal) by default,
    or with IP2P crop-blend when use_ip2p_color=True.
    Otherwise fall back to masked IP2P.
    """
    image, mask = _sanitize_pair(image, mask, max_side=512)

    if color_word and mask is not None:
        if use_ip2p_color:
            return ip2p_recolor_crop_blend(
                pipes, image, mask,
                prompt=f"slightly make the {color_word} color on the object",
                steps=max(2, steps),
                guidance=max(3.8, guidance),
                image_guidance=max(3.0, image_guidance),
                seed=seed or 42,
            )
        return recolor_masked_rgb(image, mask, color_word)

    negative = "painting, illustration, oil paint, oversharpened, high contrast, artifacts"
    full_prompt = f"{prompt}, photorealistic, minimal change outside region, preserve background details"
    generator = torch.Generator(device=pipes.device).manual_seed(seed) if seed is not None else None

    result = pipes.ip2p(
        prompt=full_prompt,
        negative_prompt=negative,
        image=image,
        num_inference_steps=steps,
        guidance_scale=guidance,
        image_guidance_scale=image_guidance,
        generator=generator,
    ).images[0]

    if mask is None:
        return result
    return composite_masked(image, result, mask)


def inpaint_edit(
    pipes: Pipelines,
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    steps: int = 25,
    guidance: float = 7.0,
    seed: Optional[int] = 42,
) -> Image.Image:
    """
    Remove/replace tasks (fill the masked region naturally).
    Uses SD-1.5 inpaint for lower memory footprint on MPS.
    """
    image, mask = _sanitize_pair(image, mask, max_side=512)

    if pipes.inpaint is None:
        dtype = torch.float32 if pipes.device in ("mps", "cpu") else torch.float16
        pipes.inpaint = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=dtype,
            safety_checker=None,
        ).to(pipes.device)
        # memory savers
        pipes.inpaint.enable_attention_slicing()
        pipes.inpaint.vae.enable_slicing()
        # CRITICAL: avoid tiled-encode crash on CPU/MPS
        pipes.inpaint.vae.disable_tiling()

    negative = "painting, illustration, oversharpened, artifacts"
    full_prompt = f"{prompt}, photorealistic, fill naturally, match lighting and perspective"
    generator = torch.Generator(device=pipes.device).manual_seed(seed) if seed is not None else None

    return pipes.inpaint(
        prompt=full_prompt,
        negative_prompt=negative,
        image=image,
        mask_image=_pil_to_mask(mask),
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    ).images[0]

# --------------------------- end file ----------------------------------------
