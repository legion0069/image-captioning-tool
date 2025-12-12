# scripts/make_demo_gif.py
"""
Simple helper to create a demo GIF from images stored in sample_images/.
Usage:
    python scripts/make_demo_gif.py --input-dir sample_images/ --output demo.gif --duration 500
"""

from pathlib import Path
from PIL import Image
import argparse

def make_gif(input_dir: Path, output_path: Path, duration=500):
    files = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    if not files:
        print("No images found in", input_dir)
        return
    imgs = []
    for p in files:
        im = Image.open(p).convert("RGB")
        imgs.append(im.copy())
    imgs[0].save(output_path, save_all=True, append_images=imgs[1:], duration=duration, loop=0)
    print("Saved GIF to", output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="sample_images")
    parser.add_argument("--output", type=str, default="demo.gif")
    parser.add_argument("--duration", type=int, default=500)
    args = parser.parse_args()
    make_gif(Path(args.input_dir), Path(args.output), duration=args.duration)
