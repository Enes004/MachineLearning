"""Image preparation shared by handwriting collection and prediction."""

from pathlib import Path

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
    return normalized


def load_handwriting_image(path: Path) -> Image.Image:
    """Read and normalize a user-provided image."""
    with Image.open(path) as image:
        return normalize_handwriting_image(image)
