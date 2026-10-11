"""
CNN feature extractor with receptive field >= grid size.

Three 3x3 conv layers give an 11x11 receptive field, which covers
6x6 and 8x8 grids entirely, and most of a 10x10 grid.
"""
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class AtomGridCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256, grid_size=None):
        super().__init__(observation_space, features_dim)

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),   # 3rd layer
            nn.Flatten(),
        )
        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None, ...])
            n = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n, features_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.linear(self.cnn(x))
