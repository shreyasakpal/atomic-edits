from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import torch
from PIL import Image

@dataclass
class DetResult:
    box: list[float]   # [x1,y1,x2,y2] in pixels
    score: float

class GDinoSAMMasker:
    """
    Grounding-DINO -> box  →  SAM -> binary mask
    phrase_to_mask(...) returns (mask_L_PIL, DetResult|None)
    """
    def __init__(self, gdino_cfg: str, gdino_weights: str, sam_weights: str,
                 sam_model_type: str = "vit_b", device: Optional[str] = None):
        if device is None:
            if torch.backends.mps.is_available(): device = "mps"
            elif torch.cuda.is_available(): device = "cuda"
            else: device = "cpu"
        self.device = device

        # Grounding-DINO
        from groundingdino.util.inference import Model
        self.det = Model(model_config_path=gdino_cfg, model_checkpoint_path=gdino_weights, device=self.device)

        # SAM
        from segment_anything import sam_model_registry, SamPredictor
        sam = sam_model_registry[sam_model_type](checkpoint=sam_weights)
        sam.to(device=self.device, dtype=torch.float32)
        self.sam_pred = SamPredictor(sam)

    def _detect_top_box(self, image_pil: Image.Image, phrase: str,
                        box_thresh: float, text_thresh: float) -> Tuple[Optional[np.ndarray], Optional[float]]:
        import numpy as np, torch

        img = np.array(image_pil)

        # Call GDINO (different releases return 2-tuple, 3-tuple, or a Detections object)
        out = self.det.predict_with_caption(
            image=img, caption=phrase, box_threshold=box_thresh, text_threshold=text_thresh
        )

        # Normalize return to (boxes_obj, scores_like)
        if isinstance(out, (list, tuple)):
            if len(out) == 3:
                boxes_obj, scores_like, _ = out
            elif len(out) == 2:
                boxes_obj, scores_like = out; _ = None
            else:
                boxes_obj, scores_like = out, None
        else:
            boxes_obj, scores_like = out, None

        if boxes_obj is None:
            return None, None

        # If it's a `supervision.Detections` object, read xyxy & confidence directly
        if hasattr(boxes_obj, "xyxy"):
            arr = np.asarray(boxes_obj.xyxy)         # (N,4)
            if arr.size == 0:
                return None, None
            conf = getattr(boxes_obj, "confidence", None)
            idx = int(np.argmax(conf)) if conf is not None else 0
            xyxy = arr[idx, :4].astype(np.float32)
            score_val = float(conf[idx]) if conf is not None else 1.0

        else:
            # Choose best index using scores (if given), otherwise first
            if scores_like is None:
                idx, score_val = 0, 1.0
            else:
                t = torch.as_tensor(scores_like)
                sc = torch.sigmoid(t) if t.dtype.is_floating_point else t
                idx = int(torch.argmax(sc).item())
                score_val = float(sc[idx].item())

            # Robustly flatten whatever structure and take first 4 numeric values
            flat = []
            for v in np.array(boxes_obj, dtype=object).reshape(-1).tolist():
                try: flat.append(float(v))
                except: pass
            if len(flat) < 4:
                return None, None
            xyxy = np.array(flat[:4], dtype=np.float32)

        # Convert normalized -> pixel coords if needed, then clamp
        w, h = image_pil.size
        if float(np.nanmax(xyxy)) <= 1.5:
            xyxy = xyxy * np.array([w, h, w, h], dtype=np.float32)
        xyxy[0::2] = np.clip(xyxy[0::2], 0, w - 1)
        xyxy[1::2] = np.clip(xyxy[1::2], 0, h - 1)

        return xyxy.astype(np.float32), score_val


    def _sam_mask_from_box(self, image_pil: Image.Image, box_xyxy) -> np.ndarray:
        import numpy as np
        img = np.array(image_pil)
        self.sam_pred.set_image(img)
        masks, scores, _ = self.sam_pred.predict(box=np.asarray(box_xyxy, dtype=np.float32),
                                                 multimask_output=True)
        idx = int(np.argmax(scores))
        return masks[idx].astype(bool)

    def phrase_to_mask(self, image_pil: Image.Image, phrase: str,
                       box_thresh: float = 0.35, text_thresh: float = 0.25):
        box, score = self._detect_top_box(image_pil, phrase, box_thresh, text_thresh)
        if box is None:
            return None, None
        mask_bool = self._sam_mask_from_box(image_pil, box)
        mask_L = Image.fromarray((mask_bool.astype(np.uint8) * 255), mode="L")
        return mask_L, DetResult(box=box.astype(float).tolist(), score=score)
