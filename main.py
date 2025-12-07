from src.config import SAMPLE_IMAGES_DIR
from src.image_utils import load_image, get_image_size

def main():
    images = list(SAMPLE_IMAGES_DIR.glob("*.jpg")) + list(SAMPLE_IMAGES_DIR.glob("*.png"))
    if not images:
        print("No images found. Add a .jpg or .png file inside sample_images/")
        return

    img_path = images[0]
    print("Loaded image:", img_path.name)

    img = load_image(img_path)
    print("Image size:", get_image_size(img))

if __name__ == "__main__":
    main()
