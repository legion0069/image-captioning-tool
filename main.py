from pathlib import Path
from src.config import SAMPLE_IMAGES_DIR
from src.image_utils import load_image, get_image_size

def main():
    samples = list(SAMPLE_IMAGES_DIR.glob("*.jpg")) + list(SAMPLE_IMAGES_DIR.glob("*.png"))
    if not samples:
        print("No sample images found. Add an image to sample_images folder.")
        return
    
    img_path = samples[0]
    print("Loaded:", img_path.name)

    img = load_image(img_path)
    print("Image size:", get_image_size(img))

if __name__ == "__main__":
    main()
