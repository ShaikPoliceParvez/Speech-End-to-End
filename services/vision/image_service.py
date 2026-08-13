"""Small, CPU-friendly image validation and preprocessing helpers."""

from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def load_image(source: str | Path | bytes) -> Image.Image:
    if isinstance(source, bytes):
        image = Image.open(BytesIO(source))
    else:
        image = Image.open(source)
    return ImageOps.exif_transpose(image).convert("RGB")


def prepare_image(source: str | Path | bytes, max_side: int = 1024) -> Image.Image:
    image = load_image(source)
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image