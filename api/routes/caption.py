from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import io

from src.model_singleton import get_ml_caption_generator

router = APIRouter()

@router.post("/caption")
async def generate_caption(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    generator = get_ml_caption_generator()
    caption = generator.generate_caption(image)

    return {
        "filename": file.filename,
        "caption": caption
    }
