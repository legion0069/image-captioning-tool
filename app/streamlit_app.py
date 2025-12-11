import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Image Captioning Demo", layout="centered")

st.title("AI Image Captioning — Demo")
st.write("Upload an image and get an ML caption (BLIP) + fallback caption.")

st.info("This demo uses a pretrained BLIP model which downloads on first run. The model may take a minute to load.")
