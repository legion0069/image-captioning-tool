from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Model will auto-download on first run (DO NOT COMMIT IT)
MODEL_NAME = "Salesforce/blip-image-captioning-base"

class MLCaptionGenerator:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = BlipProcessor.from_pretrained(MODEL_NAME)
        self.model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME).to(self.device)

    def generate_caption(self, image: Image.Image) -> str:
        inputs = self.processor(image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model.generate(**inputs)

        caption = self.processor.decode(output[0], skip_special_tokens=True)
        return caption
