from PIL import Image
import torch
from torchvision import transforms
from pathlib import Path

from src.image_utils import load_image

# ImageNet normalization values (used by pretrained CNNs)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def preprocess_image(image_path: Path):
    img = load_image(image_path)
    transform = get_transform()
    tensor = transform(img)
    return tensor

def tensor_stats(tensor):
    return {
        "shape": tuple(tensor.shape),
        "min": float(tensor.min()),
        "max": float(tensor.max())
    }
