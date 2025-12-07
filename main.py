from src.config import SAMPLE_IMAGES_DIR
from src.image_utils import load_image, get_image_size
from src.preprocessing import preprocess_image, tensor_stats
from src.feature_extractor import extract_features
import numpy as np

def main():
    images = list(SAMPLE_IMAGES_DIR.glob("*.jpg")) + list(SAMPLE_IMAGES_DIR.glob("*.png"))

    if not images:
        print("No images found. Add a .jpg or .png file inside sample_images/")
        return

    img_path = images[0]
    print("Loaded image:", img_path.name)

    img = load_image(img_path)
    print("Original image size:", get_image_size(img))

    tensor = preprocess_image(img_path)
    stats = tensor_stats(tensor)

    print("Tensor shape:", stats["shape"])
    print("Tensor range:", stats["min"], "to", stats["max"])

    features = extract_features(tensor)
    print("Feature vector shape:", features.shape)

    # Save features
    np.save("image_features.npy", features.numpy())
    print("Features saved to image_features.npy")

if __name__ == "__main__":
    main()
