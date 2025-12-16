from typing import Dict
import torch
from PIL import Image

class BLIP2YesNo:
    def __init__(self, model_name: str, device: str = "cpu", max_new_tokens: int = 8):
        self.device = device
        self.max_new_tokens = max_new_tokens
        
        # Use BLIP (v1) instead of BLIP-2 to avoid tokenizer issues
        from transformers import BlipProcessor, BlipForQuestionAnswering
        
        try:
            # Try BLIP-2 first
            from transformers import AutoProcessor, Blip2ForConditionalGeneration
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch.float32
            ).to(device)
            self.model_type = "blip2"
            print(f"Loaded BLIP-2: {model_name}")
        except:
            # Fallback to BLIP v1 which is more stable
            print(f"BLIP-2 failed, using BLIP v1 as fallback")
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
            self.model = BlipForQuestionAnswering.from_pretrained(
                "Salesforce/blip-vqa-base"
            ).to(device)
            self.model_type = "blip1"
        
        self.model.eval()
    
    @torch.inference_mode()
    def ask(self, image, question: str, yes_words=None, no_words=None) -> Dict:
        """Returns dict with answer and yes/no confidence"""
        
        if self.model_type == "blip2":
            prompt = f"Question: {question} Answer:"
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            ans = self.processor.decode(out[0], skip_special_tokens=True).strip()
        else:
            # BLIP v1
            inputs = self.processor(image, question, return_tensors="pt").to(self.device)
            out = self.model.generate(**inputs, max_length=20)
            ans = self.processor.decode(out[0], skip_special_tokens=True).strip()
        
        # Parse answer
        ans_lower = ans.lower()
        yes_words = set(yes_words or ["yes", "true", "present"])
        no_words = set(no_words or ["no", "false", "absent"])
        
        # Simple scoring
        if any(w in ans_lower for w in yes_words):
            yes_conf = 0.8
        elif any(w in ans_lower for w in no_words):
            yes_conf = 0.2
        else:
            yes_conf = 0.5
        
        return {
            "answer": ans,
            "is_yes": yes_conf > 0.5,
            "yes_conf": float(yes_conf)
        }
