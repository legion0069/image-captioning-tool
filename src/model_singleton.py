# src/model_singleton.py
"""
Simple singleton wrapper to lazily load ML models (BLIP) once per process.
This avoids repeated downloads and repeated heavy initialization.
"""

from threading import Lock
from typing import Optional
from PIL import Image

# Lazy import to avoid heavy imports during module import
_model_instance = None
_lock = Lock()

def get_ml_caption_generator():
    """
    Returns a singleton MLCaptionGenerator instance.
    The heavy imports happen inside this function on first call.
    """
    global _model_instance
    if _model_instance is not None:
        return _model_instance

    with _lock:
        # double-check in case another thread created it
        if _model_instance is not None:
            return _model_instance

        # perform lazy import here
        from src.ml_caption_generator import MLCaptionGenerator
        _model_instance = MLCaptionGenerator()
        return _model_instance
