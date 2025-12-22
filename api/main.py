from fastapi import FastAPI
from api.routes.caption import router as caption_router

app = FastAPI(
    title="AI Captioning API",
    description="REST API for AI Image Captioning using BLIP",
    version="1.0"
)

@app.get("/health")
def health():
    return {"status": "ok"}

app.include_router(caption_router, prefix="/api")
