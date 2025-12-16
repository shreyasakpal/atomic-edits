#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from .sequential_editor import run_sequence

def main():
    ap = argparse.ArgumentParser(description="Atomic Edits — Sequential Editor (InstructPix2Pix)")
    ap.add_argument("--image", required=True, help="Path to the original image")
    ap.add_argument("--json",  required=True, help="Path to the Model-1 JSON (one prompt’s result)")
    ap.add_argument("--outdir", default="artifacts/edits/run1", help="Where to save outputs")
    ap.add_argument("--model",  default="timbrooks/instruct-pix2pix", help="HF model id")
    ap.add_argument("--seed",   type=int, default=42)
    ap.add_argument("--steps",  type=int, default=20, help="diffusion steps")
    ap.add_argument("--guidance-scale", type=float, default=7.0)
    ap.add_argument("--image-guidance-scale", type=float, default=1.5)
    ap.add_argument("--negative", default="painting, illustration, lowres, artifacts, oversaturated, unrealistic")
    args = ap.parse_args()

    steps = json.loads(Path(args.json).read_text()).get("sub_instructions", [])
    final = run_sequence(
        image_path=args.image,
        steps=steps,
        outdir=args.outdir,
        model_id=args.model,
        seed=args.seed,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        image_guidance_scale=args.image_guidance_scale
    )
    print(f"Final image -> {final}")

if __name__ == "__main__":
    main()
