import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from handwriting_utils import load_digit_image, normalize_handwriting_image
from mlp_model import MNISTMLP


class MNISTMLPTests(unittest.TestCase):
    def test_two_hidden_layers_return_ten_class_logits(self) -> None:
        model = MNISTMLP((32, 16))
        logits = model(torch.zeros(4, 1, 28, 28))
        self.assertEqual(tuple(logits.shape), (4, 10))

    def test_model_requires_at_least_two_hidden_layers(self) -> None:
        with self.assertRaises(ValueError):
            MNISTMLP((32,))


class HandwritingNormalizationTests(unittest.TestCase):
    def test_image_is_28_by_28_with_white_ink_on_black_background(self) -> None:
        image = Image.new("L", (280, 280), 255)
        ImageDraw.Draw(image).line((100, 50, 170, 220), fill=0, width=20)
        normalized = normalize_handwriting_image(image)
        self.assertEqual(normalized.size, (28, 28))
        self.assertEqual(normalized.getpixel((0, 0)), 0)
        self.assertGreater(normalized.getextrema()[1], 0)

    def test_blank_image_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_handwriting_image(Image.new("L", (28, 28), 255))

    def test_digit_center_of_mass_is_centered(self) -> None:
        image = Image.new("L", (280, 280), 255)
        ImageDraw.Draw(image).line((40, 40, 60, 240), fill=0, width=20)
        normalized = normalize_handwriting_image(image)
        pixels = np.asarray(normalized, dtype=np.float64)
        rows, columns = np.indices(pixels.shape)
        self.assertAlmostEqual((rows * pixels).sum() / pixels.sum(), 13.5, delta=0.5)
        self.assertAlmostEqual((columns * pixels).sum() / pixels.sum(), 13.5, delta=0.5)

    def test_saved_28x28_digit_is_not_normalized_twice(self) -> None:
        image = Image.new("L", (280, 280), 255)
        ImageDraw.Draw(image).line((100, 50, 170, 220), fill=0, width=20)
        normalized = normalize_handwriting_image(image)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "digit.png"
            normalized.save(path)
            reloaded = load_digit_image(path)
        self.assertEqual(list(reloaded.getdata()), list(normalized.getdata()))


if __name__ == "__main__":
    unittest.main()
