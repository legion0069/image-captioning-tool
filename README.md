# AI Image Captioning Tool 🖼️🤖

Project 2 of my 100 Days of Code journey.

## Day 1 – Setup Completed
- Clean project structure created
- GitHub repo connected
- Python config added
- Image loader tested

## Tech Stack
- Python
- PyTorch (coming next)
- Pillow
- NumPy

## Next Steps
- Day 2: Image preprocessing & tensor conversion
- Day 3: Feature extraction using CNN (ResNet)
# Day 14 — Caption Generation Phase

✅ Implemented rule-based caption generator  
✅ Image features converted to text captions  
✅ Full pipeline working:
Image → Preprocess → ResNet → Features → Caption  
✅ Caption printing added in main.py  

### Next Phase (Day 15)
- Replace rule-based captions with ML-based caption model
- Integrate NLP decoder or pretrained caption model

# Day 15 — ML-Based Image Captioning ✅

✅ Integrated BLIP (pretrained image captioning model)  
✅ Real-world AI captions now generated  
✅ Rule-based captions kept as fallback  
✅ Full pipeline working:
Image → CNN → Features → Transformer → Caption  

### Key Upgrade
- Shifted from rule-based NLP to Transformer-based ML captioning
- Produces natural human-like captions

### Next (Day 16)
- Add Streamlit Web UI
- Upload image from browser
- Show caption output on screen

## Run locally (Day 16/17 demo)

1. Clone repo:
```bash
git clone https://github.com/legion0069/image-captioning-tool.git
cd image-captioning-tool


Create & activate venv

python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

to run stream lit
.\.venv\Scripts\streamlit.exe run app/streamlit_app.py


## Day 18 — Caching & Demo GIF
- Features are cached under `data/features/{image_hash}.npy` (auto-created).
- Thumbnails stored in `data/thumbs/`.
- Create demo GIF: `python scripts/make_demo_gif.py --input-dir sample_images --output demo.gif`




## 🚀 Live Demo
https://ai-captioning-by-tejaverukonda.streamlit.app/

Upload an image and get:
- ML-based caption (BLIP)
- Fallback caption
- Feature vector download
- Image metadata & thumbnail
