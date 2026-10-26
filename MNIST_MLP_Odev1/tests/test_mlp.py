import unittest

import torch
from PIL import Image, ImageDraw

from handwriting_utils import normalize_handwriting_image
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


if __name__ == "__main__":
    unittest.main()
