# Zero-Click Image Inpainting Pipeline

[![Open In Colab](https://colab.research.google.com/drive/1gscDRmVlumfdt0i-t6_dFGYgE_1c1OCU#scrollTo=_khoCNOCuHNd)

An automated, zero-shot image editing pipeline that dynamically detects objects using GroundingDINO, generates padded segmentation masks via OpenCV, and executes high-fidelity replacement using Stable Diffusion XL. 

Unlike standard inpainting workflows that require manual brush masking, this pipeline is entirely text-guided and programmatic. 

## Architecture

1. **Object Detection:** `IDEA-Research/grounding-dino-base` locates the target object based on a natural language text prompt.
2. **Dynamic Masking:** Bounding boxes are processed with auto-calculated padding and Gaussian blurring to prevent hard edges and visual artifacts during generation. 
3. **Inpainting Generation:** `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` reconstructs the masked area using the target prompt, automatically scaling to SDXL-friendly dimensions while maintaining the original aspect ratio. 

## Quick Start

### Installation
```bash
git clone [https://github.com/YOUR_USERNAME/zero-click-inpainter.git](https://github.com/YOUR_USERNAME/zero-click-inpainter.git)
cd zero-click-inpainter
pip install -r requirements.txt