# m3/evaluator/preservation_metrics.py
import torch
import lpips
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from typing import Dict, Optional, Tuple

class PreservationMetrics:
    """
    Compute preservation metrics (SSIM/LPIPS) outside target regions.
    Measures how well non-target areas are preserved after editing.
    """
    
    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        # Initialize LPIPS with AlexNet (lightweight and effective)
        self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)
        self.lpips_fn.eval()
    
    def compute_outside_mask(
        self, 
        original: Image.Image, 
        edited: Image.Image, 
        mask: Optional[Image.Image] = None,
        dilate_mask: int = 5
    ) -> Dict[str, float]:
        """
        Compute preservation metrics outside the masked region.
        
        Args:
            original: Original image
            edited: Edited image  
            mask: Binary mask (255 for target region, 0 for background)
            dilate_mask: Pixels to dilate mask to avoid boundary artifacts
            
        Returns:
            Dict with ssim_outside, lpips_outside, and coverage metrics
        """
        # Ensure same size
        if edited.size != original.size:
            edited = edited.resize(original.size, Image.LANCZOS)
        
        orig_np = np.array(original.convert("RGB"))
        edit_np = np.array(edited.convert("RGB"))
        
        # Global metrics if no mask provided
        if mask is None:
            ssim_val = ssim(orig_np, edit_np, channel_axis=2, data_range=255)
            lpips_val = self._compute_lpips(original, edited)
            return {
                "ssim_global": float(ssim_val),
                "lpips_global": float(lpips_val),
                "ssim_outside": float(ssim_val),
                "lpips_outside": float(lpips_val),
                "coverage": 1.0
            }
        
        # Process mask
        if mask.size != original.size:
            mask = mask.resize(original.size, Image.NEAREST)
        
        mask_np = np.array(mask.convert("L"))
        
        # Dilate mask to avoid boundary artifacts
        if dilate_mask > 0:
            from scipy.ndimage import binary_dilation
            mask_binary = mask_np > 128
            mask_binary = binary_dilation(mask_binary, iterations=dilate_mask)
            mask_np = mask_binary.astype(np.uint8) * 255
        
        # Create inverse mask for outside region
        outside_mask = (mask_np < 128).astype(np.float32)
        coverage = outside_mask.sum() / outside_mask.size
        
        if coverage < 0.01:  # Almost no outside region
            return {
                "ssim_outside": 1.0,
                "lpips_outside": 0.0,
                "coverage": float(coverage),
                "note": "minimal_outside_region"
            }
        
        # Compute masked SSIM
        ssim_map = self._compute_ssim_map(orig_np, edit_np)
        ssim_outside = float((ssim_map * outside_mask).sum() / outside_mask.sum())
        
        # Compute masked LPIPS
        lpips_outside = self._compute_lpips_masked(original, edited, outside_mask)
        
        # Also compute global metrics for comparison
        ssim_global = ssim(orig_np, edit_np, channel_axis=2, data_range=255)
        lpips_global = self._compute_lpips(original, edited)
        
        return {
            "ssim_global": float(ssim_global),
            "lpips_global": float(lpips_global),
            "ssim_outside": float(ssim_outside),
            "lpips_outside": float(lpips_outside),
            "coverage": float(coverage)
        }
    
    def _compute_ssim_map(self, img1_np, img2_np, window_size=7):
        """Compute pixel-wise SSIM map."""
        # Use smaller window for local SSIM
        _, ssim_map = ssim(
            img1_np, img2_np, 
            channel_axis=2, 
            data_range=255,
            win_size=window_size,
            full=True
        )
        return ssim_map.mean(axis=2)  # Average across channels
    
    @torch.no_grad()
    def _compute_lpips(self, img1: Image.Image, img2: Image.Image) -> float:
        """Compute global LPIPS distance."""
        # Convert to tensors and normalize to [-1, 1]
        t1 = self._pil_to_tensor(img1)
        t2 = self._pil_to_tensor(img2)
        dist = self.lpips_fn(t1, t2)
        return dist.item()
    
    @torch.no_grad()
    def _compute_lpips_masked(
        self, 
        img1: Image.Image, 
        img2: Image.Image, 
        outside_mask: np.ndarray
    ) -> float:
        """
        Compute LPIPS only on outside regions by masking the difference.
        """
        t1 = self._pil_to_tensor(img1)
        t2 = self._pil_to_tensor(img2)
        
        # Get LPIPS features
        diff = self.lpips_fn.forward(t1, t2)
        
        # Downsample mask to match LPIPS spatial dims
        h, w = diff.shape[2:]
        mask_resized = Image.fromarray((outside_mask * 255).astype(np.uint8))
        mask_resized = mask_resized.resize((w, h), Image.NEAREST)
        mask_tensor = torch.from_numpy(
            np.array(mask_resized).astype(np.float32) / 255.0
        ).to(self.device)
        
        # Apply mask and recompute mean
        masked_diff = diff.squeeze() * mask_tensor
        lpips_val = masked_diff.sum() / mask_tensor.sum().clamp(min=1)
        
        return lpips_val.item()
    
    def _pil_to_tensor(self, img: Image.Image) -> torch.Tensor:
        """Convert PIL image to normalized tensor."""
        img = img.convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        arr = arr * 2.0 - 1.0  # Normalize to [-1, 1]
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)