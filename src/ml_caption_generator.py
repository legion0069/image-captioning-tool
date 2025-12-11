# src/ml_caption_generator.py
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

MODEL_NAME = "Salesforce/blip-image-captioning-base"

class MLCaptionGenerator:
    def __init__(self, device=None):
        # choose device: cuda if available else cpu
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Processor + model will auto-download on first run
        self.processor = BlipProcessor.from_pretrained(MODEL_NAME)
        self.model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME).to(self.device)
        self.model.eval()

    def generate_caption(self, image: Image.Image) -> str:
        """
        image: PIL.Image (RGB)
        returns: decoded caption string
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(**inputs)
        caption = self.processor.decode(output[0], skip_special_tokens=True)
        return caption
