import requests
from io import BytesIO
import PIL
from PIL import Image

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = PIL.Image.new('RGB', size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")

def get_sdxl_dimensions(width, height):
    """Calculates SDXL-friendly dimensions (multiples of 64) maintaining aspect ratio."""
    target_area = 1024 * 1024
    aspect_ratio = width / height

    new_height = int((target_area / aspect_ratio) ** 0.5)
    new_width = int(new_height * aspect_ratio)

    # Round to nearest multiple of 64
    new_width = max(64, (new_width // 64) * 64)
    new_height = max(64, (new_height // 64) * 64)

    return new_width, new_height