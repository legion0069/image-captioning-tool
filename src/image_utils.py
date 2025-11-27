from PIL import Image
from pathlib import Path

def load_image(path: Path):
    img = Image.open(path).convert("RGB")
    return img

def get_image_size(img):
    return img.size
