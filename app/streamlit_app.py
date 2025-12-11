import streamlit as st
from pathlib import Path
from PIL import Image
import io
import numpy as np

# Import your modules
from src.preprocessing import preprocess_image, tensor_stats
from src.feature_extractor import extract_features
from src.caption_generator import generate_basic_caption
from src.ml_caption_generator import MLCaptionGenerator

st.set_page_config(page_title="Image Captioning Demo", layout="centered")
st.title("AI Image Captioning — Demo")
st.write("Upload an image and get an ML caption (BLIP) + fallback caption.")
st.info("Model downloads on first run; first response may take longer.")

uploaded = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])

if not uploaded:
    st.caption("No image uploaded yet.")
    st.stop()

# Read and display image
bytes_data = uploaded.read()
try:
    img = Image.open(io.BytesIO(bytes_data)).convert("RGB")
except Exception as e:
    st.error(f"Could not read image: {e}")
    st.stop()

st.image(img, caption="Uploaded image", use_column_width=True)
st.markdown("---")

# Preprocess / show tensor stats
st.subheader("Preprocessing")
# Save temporarily to sample_images to reuse preprocess path functions (clean approach)
TEMP_PATH = Path("sample_images") / "upload_preview_temp.jpg"
TEMP_PATH.parent.mkdir(parents=True, exist_ok=True)
img.save(TEMP_PATH)

try:
    tensor = preprocess_image(TEMP_PATH)
    stats = tensor_stats(tensor)
    st.write(f"Tensor shape: `{stats['shape']}`")
    st.write(f"Tensor value range: min=`{stats['min']:.4f}` max=`{stats['max']:.4f}`")
except Exception as e:
    st.error(f"Preprocessing failed: {e}")
    st.stop()

# Feature extraction
st.subheader("Features")
try:
    features = extract_features(tensor)  # torch tensor -> features (torch.Tensor)
    feat_np = features.cpu().numpy()
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
st.subheader("ML Caption (BLIP)")
with st.spinner("Generating ML caption (may take a moment)..."):
    try:
        generator = MLCaptionGenerator()
        ml_caption = generator.generate_caption(img)
        st.write(ml_caption)
    except Exception as e:
        st.error(f"ML captioning failed: {e}")

# Download features as .npy
st.markdown("---")
st.subheader("Download features")
try:
    st.download_button(
        label="Download feature vector (.npy)",
        data=io.BytesIO(feat_np.tobytes()),
        file_name="image_features.npy",
        mime="application/octet-stream"
    )
except Exception as e:
    st.error(f"Download failed: {e}")
