# app/streamlit_app.py
# Streamlit demo for Image Captioning (clean, robust, import-friendly)

import sys
from pathlib import Path
import io

# Ensure project root is on Python path so imports like "from src..." work
root = Path(__file__).resolve().parents[1]
root_str = str(root)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

import streamlit as st
from PIL import Image
import numpy as np

# Import your project modules (these must exist in src/)
from src.preprocessing import preprocess_image, tensor_stats
from src.feature_extractor import extract_features
from src.caption_generator import generate_basic_caption
from src.ml_caption_generator import MLCaptionGenerator

# ----------------------
# Streamlit UI
# ----------------------
st.set_page_config(page_title="Image Captioning", layout="centered")
st.title("AI Image Captioning")
st.write("Upload an image to generate captions using ML or fallback logic.")
st.info("The BLIP model downloads on first run; the first response may take longer.")

# File upload
uploaded = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])
if not uploaded:
    st.caption("No image uploaded yet.")
    st.stop()

# Read and display image
try:
    bytes_data = uploaded.read()
    img = Image.open(io.BytesIO(bytes_data)).convert("RGB")
except Exception as e:
    st.error(f"Could not read image: {e}")
    st.stop()

st.image(img, caption="Uploaded image", use_container_width=True)
st.markdown("---")

# Preprocess / show tensor stats
st.subheader("Preprocessing")
TEMP_DIR = root / "sample_images"
TEMP_DIR.mkdir(parents=True, exist_ok=True)
TEMP_PATH = TEMP_DIR / "upload_preview_temp.jpg"

try:
    # Save uploaded image temporarily so existing preprocess utilities (which accept a path) work
    img.save(TEMP_PATH)
except Exception as e:
    st.error(f"Failed to save uploaded image for preprocessing: {e}")
    st.stop()

try:
    tensor = preprocess_image(TEMP_PATH)  # expects a Path
    stats = tensor_stats(tensor)
    st.write(f"Tensor shape: `{stats['shape']}`")
    st.write(f"Tensor value range: min=`{stats['min']:.4f}` max=`{stats['max']:.4f}`")
except Exception as e:
    st.error(f"Preprocessing failed: {e}")
    st.stop()

# Feature extraction
# Feature extraction (cache-aware)
st.subheader("Features")
from src.image_meta import image_hash  # small helper we'll add in Commit 2
from src.cache_utils import load_features, save_features, has_features

try:
    # compute cache key from temp path
    key = image_hash(TEMP_PATH)
    cached = load_features(key)
    if cached is not None:
        feat_np = cached
        st.write("Loaded features from cache.")
    else:
        features = extract_features(tensor)  # torch tensor -> features (torch.Tensor)
        feat_np = features.cpu().numpy()
        save_features(key, feat_np)
        st.write("Extracted features and saved to cache.")
    st.write(f"Feature vector shape: `{feat_np.shape}`")
except Exception as e:
    st.error(f"Feature extraction failed: {e}")
    st.stop()


# Fallback caption
st.subheader("Fallback Caption")
try:
    fallback_caption = generate_basic_caption(feat_np)
    st.write(fallback_caption)
except Exception as e:
    st.error(f"Fallback caption failed: {e}")

# ML caption
# inside app/streamlit_app.py — replace ML caption block with this snippet
# (ensure you already have the top-of-file sys.path fix and imports for st, Image, io, np, etc.)
from src.model_singleton import get_ml_caption_generator

# ... earlier code for upload, preprocess, features, fallback caption ...

# ML caption (behind a button, uses singleton)
st.subheader("ML Caption (BLIP)")
if st.button("Generate ML Caption"):
    with st.spinner("Loading model (first run may take a minute) and generating caption..."):
        try:
            # get singleton model (loads once per process)
            generator = get_ml_caption_generator()
            ml_caption = generator.generate_caption(img)
            st.success("ML caption generated:")
            st.write(ml_caption)
        except Exception as e:
            st.error(f"ML captioning failed: {e}")


# Download features as .npy
st.markdown("---")
st.subheader("Download features")
try:
    # Save numpy array to an in-memory buffer as .npy
    buf = io.BytesIO()
    # np.save writes .npy header; ensure buffer at start for download
    np.save(buf, feat_np)
    buf.seek(0)
    st.download_button(
        label="Download feature vector (.npy)",
        data=buf,
        file_name="image_features.npy",
        mime="application/octet-stream"
    )
except Exception as e:
    st.error(f"Download failed: {e}")
