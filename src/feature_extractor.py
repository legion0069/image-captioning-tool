import torch
from src.model_utils import load_resnet18_feature_extractor

def extract_features(image_tensor):
    """
    Takes a preprocessed image tensor of shape (3, 224, 224)
    Returns a feature vector.
    """
    model = load_resnet18_feature_extractor()

    # Add batch dimension: (1, 3, 224, 224)
    image_tensor = image_tensor.unsqueeze(0)

    with torch.no_grad():
        features = model(image_tensor)

    return features.squeeze()  # Remove batch dimension
