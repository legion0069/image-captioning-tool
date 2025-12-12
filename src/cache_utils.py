# src/cache_utils.py
from pathlib import Path
import numpy as np

FEATURE_DIR = Path("data") / "features"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

def feature_path_for_key(key: str) -> Path:
    return FEATURE_DIR / f"{key}.npy"

def save_features(key: str, features: np.ndarray) -> None:
    """
    Save features numpy array under data/features/{key}.npy
    """
    p = feature_path_for_key(key)
    np.save(p, features)
    return

def load_features(key: str):
    """
    Return numpy array or None if not found
    """
    p = feature_path_for_key(key)
    if p.exists():
        try:
            return np.load(p)
        except Exception:
            return None
    return None

def has_features(key: str) -> bool:
    return feature_path_for_key(key).exists()
