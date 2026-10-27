"""Split a photo of a handwritten school number into 28x28 MNIST-style digit images."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from handwriting_utils import normalize_handwriting_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("photo", type=Path, help="Photo of the digits written on paper.")
    parser.add_argument(
        "--labels", required=True,
        help="The digits in the photo from left to right, used in the file names.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/handwriting"))
    parser.add_argument("--preview", type=Path, default=Path("artifacts/handwriting_digits.png"))
    args = parser.parse_args()
    if not args.labels.isdigit():
        parser.error("labels must contain digits only.")
    return args


def find_ink(photo: Image.Image) -> np.ndarray:
    """Return a boolean ink mask that is robust to uneven lighting on the paper."""
    gray = photo.convert("L")
    background = gray.filter(ImageFilter.MedianFilter(41))
    pixels = np.asarray(gray, dtype=np.float32)
    flat = pixels / np.maximum(np.asarray(background, dtype=np.float32), 1.0)
    ink = Image.fromarray(((flat < 0.85) * 255).astype(np.uint8))
    # Thin pen strokes can break into pieces; a 3x3 dilation reconnects them.
    return np.asarray(ink.filter(ImageFilter.MaxFilter(3))) > 0


def connected_components(mask: np.ndarray) -> list[np.ndarray]:
    """Group ink pixels into 8-connected components."""
    height, width = mask.shape
    seen = np.zeros_like(mask)
    components = []
    for start in zip(*np.nonzero(mask)):
        if seen[start]:
            continue
        seen[start] = True
        stack = [start]
        pixels = []
        while stack:
            row, column = stack.pop()
            pixels.append((row, column))
            for d_row in (-1, 0, 1):
                for d_column in (-1, 0, 1):
                    r, c = row + d_row, column + d_column
                    if 0 <= r < height and 0 <= c < width and mask[r, c] and not seen[r, c]:
                        seen[r, c] = True
                        stack.append((r, c))
        components.append(np.array(pixels))
    return components


def main() -> None:
    args = parse_args()
    with Image.open(args.photo) as photo:
        mask = find_ink(photo)

    # Ignore specks of noise: a real digit is tall and has enough ink pixels.
    digits = [
        pixels for pixels in connected_components(mask)
        if len(pixels) >= 40 and np.ptp(pixels[:, 0]) + 1 >= 20
    ]
    digits.sort(key=lambda pixels: pixels[:, 1].min())
    if len(digits) != len(args.labels):
        raise SystemExit(
            f"Found {len(digits)} digits but {len(args.labels)} labels were given; "
            "check the photo (lighting, touching digits) before continuing."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.preview.parent.mkdir(parents=True, exist_ok=True)
    preview = Image.new("L", (len(digits) * 120 + 8, 120), 128)
    for position, (pixels, label) in enumerate(zip(digits, args.labels), start=1):
        # Draw the digit as black ink on white, the input normalize_handwriting_image expects.
        top, left = pixels.min(axis=0) - 12
        bottom, right = pixels.max(axis=0) + 13
        canvas = np.full((bottom - top, right - left), 255, dtype=np.uint8)
        canvas[pixels[:, 0] - top, pixels[:, 1] - left] = 0
        normalized = normalize_handwriting_image(Image.fromarray(canvas))
        filename = f"digit_{position:02d}_label_{label}.png"
        normalized.save(args.output_dir / filename)
        preview.paste(normalized.resize((112, 112), Image.NEAREST), (8 + (position - 1) * 120, 4))
        print(f"Saved {args.output_dir / filename}")

    preview.save(args.preview)
    print(f"Saved preview to {args.preview}")


if __name__ == "__main__":
    main()
