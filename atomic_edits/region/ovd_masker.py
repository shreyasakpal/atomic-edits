# atomic_edits/region/ovd_masker.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
from PIL import Image, ImageDraw
from transformers import OwlViTProcessor, OwlViTForObjectDetection  

_MODEL_NAME = "google/owlvit-base-patch16"  # stable, widely supported

@dataclass
class Detection:
    box: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    score: float
    label: str

class OVDMasker:
    """
    Open-vocabulary detector using OWL-ViT (v1).
    Returns a rectangular mask for the best matching query phrase.
    """

    def __init__(self, device: Optional[str] = None):
        # IMPORTANT: set device first, then move model to it
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.processor = OwlViTProcessor.from_pretrained(_MODEL_NAME)
        self.model = OwlViTForObjectDetection.from_pretrained(_MODEL_NAME).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def detect(self, image: Image.Image, phrases: List[str], score_thresh: float = 0.20) -> List[Detection]:
        # OWL-ViT expects a batch of texts and images
        inputs = self.processor(text=[phrases], images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        # post-process to original image size (h, w)
        target_sizes = torch.tensor([image.size[::-1]], device=self.device)
        results = self.processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=score_thresh
        )[0]

        dets: List[Detection] = []
        for box, score, label_id in zip(results["boxes"], results["scores"], results["labels"]):
            x1, y1, x2, y2 = box.round().to("cpu").numpy().astype(int).tolist()
            label = phrases[int(label_id)]
            dets.append(Detection((x1, y1, x2, y2), float(score.item()), label))

        dets.sort(key=lambda d: d.score, reverse=True)
        return dets

    @staticmethod
    def rect_mask(image: Image.Image, box: Tuple[int, int, int, int], pad: int = 4) -> Image.Image:
        w, h = image.size
        x1, y1, x2, y2 = box
        x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
        m = Image.new("L", (w, h), 0)
        ImageDraw.Draw(m).rectangle([x1, y1, x2, y2], fill=255)
        return m

    def phrase_to_mask(
        self,
        image: Image.Image,
        phrase: str,
        extra_phrases: Optional[List[str]] = None,
        score_thresh: float = 0.20,
    ) -> Tuple[Image.Image, Optional[Detection]]:
        queries = [phrase] + (extra_phrases or [])
        dets = self.detect(image, queries, score_thresh=score_thresh)
        if not dets:
            return Image.new("L", image.size, 0), None
        best = dets[0]
        return self.rect_mask(image, best.box), best
