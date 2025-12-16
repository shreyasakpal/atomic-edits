# atomic_edits/region/cli.py
import argparse
from pathlib import Path
from PIL import Image
from .ovd_masker import OVDMasker

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--phrase", required=True, help='e.g., "logo", "mug", "sky"')
    ap.add_argument("--out", required=True, help="mask output path (PNG)")
    ap.add_argument("--thresh", type=float, default=0.20)
    args = ap.parse_args()

    img = Image.open(args.image).convert("RGB")
    masker = OVDMasker()
    mask, det = masker.phrase_to_mask(img, args.phrase, score_thresh=args.thresh)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    mask.save(args.out)
    if det:
        print(f"[mask] {args.phrase} -> box={det.box}, score={det.score:.3f}")
    else:
        print("[mask] no detection above threshold; saved empty mask")

if __name__ == "__main__":
    main()
