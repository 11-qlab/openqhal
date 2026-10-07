"""
CNN feature extractor for the atom grid observation.

Splits the flat observation into grid channels and scalars.
The grid is reshaped into (channels, H, W) and passed through a
small CNN. The output is concatenated with the scalars and
projected to a fixed-size feature vector for the policy / value heads.
"""
from __future__ import annotations
import torch
import torch.nn as nn

try:
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    _HAS_SB3 = True
except ImportError:
    BaseFeaturesExtractor = nn.Module
    _HAS_SB3 = False


class AtomGridCNN(BaseFeaturesExtractor):
    """CNN feature extractor for grid-shaped atom observations."""

    def __init__(self, observation_space, features_dim: int = 256,
                 grid_size: int = 5):
        if _HAS_SB3:
            super().__init__(observation_space, features_dim)
        else:
            super().__init__()

        self.grid_size = grid_size
        input_dim = int(observation_space.shape[0])

        self.grid_elements = grid_size * grid_size
        self.channels = max(1, input_dim // self.grid_elements)
        self.scalar_dim = input_dim - self.channels * self.grid_elements

        self.cnn = nn.Sequential(
            nn.Conv2d(self.channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        cnn_out = 64 * self.grid_size * self.grid_size

        self.linear = nn.Sequential(
            nn.Linear(cnn_out + self.scalar_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        grid_dim = self.channels * self.grid_elements
        grid_data = observations[:, :grid_dim]
        scalars = observations[:, grid_dim:]

        grid_2d = grid_data.view(-1, self.channels,
                                  self.grid_size, self.grid_size)
        cnn_features = self.cnn(grid_2d)

        if self.scalar_dim > 0:
            combined = torch.cat((cnn_features, scalars), dim=1)
        else:
            combined = cnn_features

        return self.linear(combined)
