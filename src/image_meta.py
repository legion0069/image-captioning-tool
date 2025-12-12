# src/image_meta.py
from pathlib import Path
from PIL import Image
import hashlib
import os

THUMB_DIR = Path("data") / "thumbs"
THUMB_DIR.mkdir(parents=True, exist_ok=True)

def image_hash(path: Path) -> str:
    """
    Compute a short sha1-based hash for an image file (used as cache key).
    """
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def create_thumbnail(path: Path, size=(256, 256)):
    """
    Create a JPG thumbnail for display and return its Path.
    """
    thumb_path = THUMB_DIR / f"{path.stem}_thumb.jpg"
    try:
        img = Image.open(path)
        img.thumbnail(size)
        img = img.convert("RGB")
        img.save(thumb_path, format="JPEG", quality=85)
        return thumb_path
    except Exception:
        # on fail, return None
        return None

def filesize_bytes(path: Path) -> int:
    try:
        return os.path.getsize(path)
    except Exception:
        return 0
