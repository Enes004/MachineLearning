"""Predict student-created digit images using a saved MNIST MLP checkpoint."""

import argparse
import csv
import re
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from handwriting_utils import load_digit_image
from mlp_model import MNISTMLP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, default=Path("data/handwriting"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/best_mnist_mlp.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/handwriting_predictions.csv"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested, but it is not available.")
    return args


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {args.checkpoint}")
    image_paths = sorted(
        path
        for path in args.images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    ) if args.images_dir.is_dir() else []
    if not image_paths:
        raise FileNotFoundError(
            f"No digit images found in {args.images_dir}. Run collect_handwriting.py first."
        )

    device = torch.device(
        "cuda" if args.device == "cuda" or (
            args.device == "auto" and torch.cuda.is_available()
        ) else "cpu"
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = MNISTMLP(checkpoint["hidden_dims"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    correct = 0
    known_labels = 0
    with args.output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(("image", "expected_digit", "predicted_digit", "confidence", "correct"))
        for path in image_paths:
            image = load_digit_image(path)
            tensor = pil_to_tensor(image).float().div_(255.0).unsqueeze(0).to(device)
            with torch.inference_mode():
                probabilities = model(tensor).softmax(dim=1)[0]
            predicted = int(probabilities.argmax().item())
            match = re.search(r"_label_([0-9])$", path.stem)
            expected = match.group(1) if match else ""
            is_correct = expected != "" and predicted == int(expected)
            if expected:
                known_labels += 1
                correct += int(is_correct)
            writer.writerow(
                (path.name, expected, predicted, f"{probabilities[predicted].item():.6f}",
                 is_correct if expected else "")
            )
            print(
                f"{path.name}: prediction={predicted}, "
                f"confidence={probabilities[predicted].item():.1%}"
                + (f", expected={expected}, correct={is_correct}" if expected else "")
            )

    if known_labels:
        print(f"Handwriting accuracy: {correct}/{known_labels} = {correct / known_labels:.2%}")
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
