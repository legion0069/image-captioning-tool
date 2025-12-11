import streamlit as st
from pathlib import Path
from PIL import Image
import io

st.set_page_config(page_title="Image Captioning Demo", layout="centered")

st.title("AI Image Captioning — Demo")
st.write("Upload an image and get an ML caption (BLIP) + fallback caption.")
st.info("Model downloads on first run; first response may take longer.")

uploaded = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded:
    try:
        bytes_data = uploaded.read()
        img = Image.open(io.BytesIO(bytes_data)).convert("RGB")
    except Exception as e:
        st.error(f"Could not read image: {e}")
        st.stop()

    st.image(img, caption="Uploaded image", use_column_width=True)
    st.markdown("---")
    st.write("Now you can generate captions using the ML model (BLIP) and the fallback rule-based caption.")
else:
    st.caption("No image uploaded yet.")
