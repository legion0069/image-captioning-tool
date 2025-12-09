import numpy as np

def generate_basic_caption(features: np.ndarray) -> str:
    """
    Rule-based caption generation using feature vector.
    This is a placeholder for ML-based captioning later.
    """

    avg_value = features.mean()

    if avg_value > 0.5:
        return "A bright and detailed image."
    elif avg_value > 0.2:
        return "An image with moderate visual details."
    else:
        return "A dark or low-detail image."
