import torch
import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageDraw
from diffusers import StableDiffusionXLInpaintPipeline
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from utils import get_sdxl_dimensions

class ZeroClickInpainter:
    def __init__(self, device="cuda"):
        self.device = device
        print("Loading Models... (This takes a moment)")

        # Load GroundingDINO
        self.dino_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(device)

        # Load Stable Diffusion XL
        self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        self.pipe.enable_model_cpu_offload()

    def process(self, input_image, detect_prompt, target_prompt, negative_prompt="", scale_mode="auto"):
        print(f"Detecting '{detect_prompt}'...")
        inputs = self.dino_processor(images=input_image, text=detect_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.dino_model(**inputs)

        results = self.dino_processor.image_processor.post_process_object_detection(
            outputs, target_sizes=[input_image.size[::-1]]
        )[0]
        best_box = results["boxes"][results["scores"].argmax()].cpu().numpy().tolist()

        x1, y1, x2, y2 = best_box
        car_w = x2 - x1
        car_h = y2 - y1
        img_w, img_h = input_image.size
        center_x = (x1 + x2) / 2

        raw_mask_array = np.zeros((img_h, img_w), dtype=np.uint8)

        if scale_mode == "auto":
            pad_x = car_w * 0.02
            pad_y = car_h * 0.02
            canvas_x1 = max(0, x1 - pad_x)
            canvas_y1 = max(0, y1 - pad_y)
            canvas_x2 = min(img_w, x2 + pad_x)
            canvas_y2 = min(img_h, y2 + pad_y)
            cv2.rectangle(raw_mask_array, (int(canvas_x1), int(canvas_y1)), (int(canvas_x2), int(canvas_y2)), 255, -1)

        elif scale_mode == "center_seat":
            person_w = img_w * 0.35
            person_h = img_h * 0.65
            mask_y2 = min(img_h, y2 + (car_h * 0.2))
            mask_y1 = max(0, mask_y2 - person_h)
            mask_x1 = max(0, center_x - (person_w / 2))
            mask_x2 = min(img_w, center_x + (person_w / 2))
            cv2.rectangle(raw_mask_array, (int(mask_x1), int(mask_y1)), (int(mask_x2), int(mask_y2)), 255, -1)

        mask_img = Image.fromarray(raw_mask_array)
        final_mask = mask_img.filter(ImageFilter.GaussianBlur(radius=8))
        
        bbox_log = input_image.copy()
        draw = ImageDraw.Draw(bbox_log)
        draw.rectangle(best_box, outline="red", width=3)

        print(f"Generating new image: '{target_prompt}'...")
        sd_w, sd_h = get_sdxl_dimensions(img_w, img_h)
        image_gen = input_image.resize((sd_w, sd_h), Image.Resampling.LANCZOS)
        mask_gen = final_mask.resize((sd_w, sd_h), Image.Resampling.NEAREST)

        output = self.pipe(
            prompt=target_prompt,
            negative_prompt=negative_prompt,
            image=image_gen,
            mask_image=mask_gen,
            width=sd_w,
            height=sd_h,
            num_inference_steps=60,
            strength=0.99,
            guidance_scale=7.5
        ).images[0]

        output = output.resize((img_w, img_h), Image.Resampling.LANCZOS)
        return output, final_mask, bbox_log