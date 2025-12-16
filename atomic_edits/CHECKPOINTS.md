# Model Checkpoints Setup

This project requires the following model checkpoints. Due to file size, they are not included in the repository.

## Required Downloads

### GroundingDINO
- **Config**: `checkpoints/GroundingDINO_SwinT_OGC.py` (included in repo)
- **Weights**: `checkpoints/groundingdino_swint_ogc.pth`
  - Download: https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

### Segment Anything (SAM)
- **Weights**: `checkpoints/sam_vit_b.pth`
  - Download: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
  - Rename to `sam_vit_b.pth`

- **Weights (optional, higher quality)**: `checkpoints/sam_vit_h_4b8939.pth`
  - Download: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

## Setup Script
```bash
#!/bin/bash
mkdir -p checkpoints

# Download GroundingDINO
wget -P checkpoints/ https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# Download SAM ViT-B
wget -P checkpoints/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
mv checkpoints/sam_vit_b_01ec64.pth checkpoints/sam_vit_b.pth

echo "Checkpoints downloaded successfully"
```

## Directory Structure
After downloading, your `checkpoints/` should look like:
```
checkpoints/
├── GroundingDINO_SwinT_OGC.py
├── groundingdino_swint_ogc.pth
└── sam_vit_b.pth
```