"""Image preparation shared by handwriting collection and prediction."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")


def normalize_handwriting_image(image: Image.Image) -> Image.Image:
    """Convert dark ink on a light background to a centered MNIST-like 28x28 image."""
    grayscale = ImageOps.grayscale(image)
    inverted = ImageOps.invert(grayscale)
    ink = inverted.point(lambda value: 255 if value > 30 else 0)
    bounds = ink.getbbox()
    if bounds is None:
        raise ValueError("The image is blank; draw a digit before saving.")

    digit = ink.crop(bounds)
    digit.thumbnail((20, 20), _LANCZOS)
    normalized = Image.new("L", (28, 28), 0)
    normalized.paste(digit, ((28 - digit.width) // 2, (28 - digit.height) // 2))
    return _center_by_mass(normalized)


def _center_by_mass(image: Image.Image) -> Image.Image:
    """Shift the digit so its center of mass is in the middle, as MNIST does."""
    pixels = np.asarray(image, dtype=np.float64)
    rows, columns = np.indices(pixels.shape)
    total = pixels.sum()
    center_row = (rows * pixels).sum() / total
    center_column = (columns * pixels).sum() / total
    centered = Image.new("L", (28, 28), 0)
    centered.paste(image, (round(13.5 - center_column), round(13.5 - center_row)))
    return centered


def load_digit_image(path: Path) -> Image.Image:
    """Read a saved digit image; 28x28 files are already normalized, so they are not changed."""
    with Image.open(path) as image:
        if image.size == (28, 28):
            return image.convert("L")
        return normalize_handwriting_image(image)
