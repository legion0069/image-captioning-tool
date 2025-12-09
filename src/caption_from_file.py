import numpy as np
from src.caption_generator import generate_basic_caption

def caption_from_features_file(file_path="image_features.npy"):
    features = np.load(file_path)
    caption = generate_basic_caption(features)
    return caption
