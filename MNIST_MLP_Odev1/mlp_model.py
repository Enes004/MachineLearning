"""MLP model for MNIST digit classification."""

from collections.abc import Sequence

import torch
from torch import nn


class MNISTMLP(nn.Module):
    """A fully connected network; this model contains no convolutional layers."""

    def __init__(self, hidden_dims: Sequence[int] = (256, 128)) -> None:
        super().__init__()
        if len(hidden_dims) < 2:
            raise ValueError("At least two hidden layers are required.")
        if any(width <= 0 for width in hidden_dims):
            raise ValueError("Hidden-layer widths must be positive.")

        layers: list[nn.Module] = [nn.Flatten()]
        input_dim = 28 * 28
        for width in hidden_dims:
            layers.extend((nn.Linear(input_dim, width), nn.ReLU()))
            input_dim = width
        layers.append(nn.Linear(input_dim, 10))
        self.network = nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)
