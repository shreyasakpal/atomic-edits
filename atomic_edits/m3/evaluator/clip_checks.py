# m3/evaluator/clip_checks.py
from typing import Dict, List
import torch, open_clip
from PIL import Image

class CLIPAttribute:
    def __init__(self, model_name="ViT-B-32", pretrained="openai", device="cuda"):
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=self.device)
        self.tokenizer = open_clip.get_tokenizer(model_name)

    @torch.inference_mode()
    def score_phrase(self, image, positive_templates: List[str], negative_templates: List[str], phrase: str) -> Dict:
        img = self.preprocess(image).unsqueeze(0).to(self.device)
        pos_texts = [t.format(phrase=phrase) for t in positive_templates]
        neg_texts = [t.format(phrase=phrase) for t in negative_templates]
        texts = pos_texts + neg_texts
        tokens = self.tokenizer(texts).to(self.device)
        img_feat = self.model.encode_image(img)
        txt_feat = self.model.encode_text(tokens)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        sims = (img_feat @ txt_feat.t()).squeeze(0)  # [n_texts]
        pos = sims[:len(pos_texts)].mean().item()
        neg = sims[len(pos_texts):].mean().item()
        return {"pos": float(pos), "neg": float(neg), "margin": float(pos - neg)}
