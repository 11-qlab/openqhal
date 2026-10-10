"""Inference wrapper for trained RL rearrangement models."""
from dataclasses import dataclass
import numpy as np
import torch

from .gym_wrapper import ShapedAtomGym

try:
    from stable_baselines3 import PPO
except ImportError:
    PPO = None


@dataclass
class SolveResult:
    success: bool
    steps: int


class AtomRearranger:
    def __init__(self, checkpoint_path, grid_size, n_atoms, max_steps=None):
        if PPO is None:
            raise RuntimeError("stable_baselines3 not installed")
        self.grid_size = grid_size
        self.n_atoms = n_atoms
        self.max_steps = max_steps or grid_size * 3
        self.env = ShapedAtomGym(
            grid_size=grid_size, n_atoms=n_atoms,
            max_steps=self.max_steps, scramble=1,
            shape_scale=0.0, seed=0)
        self.model = PPO.load(checkpoint_path, device="cpu")

    def set_configuration(self, initial, target):
        self.env.env.grid = torch.tensor(initial, dtype=torch.int8)
        self.env.env.target = torch.tensor(target, dtype=torch.int8)
        self.env.env.steps = 0
        self.env.env.prev_dist = self.env.env._dist()
        obs = np.asarray(self.env.env._obs())
        self.env._last_potential = self.env._potential(obs)

    def solve(self, initial, target, greedy=True):
        self.set_configuration(initial, target)
        obs = np.asarray(self.env.env._obs(), dtype=np.float32)
        done = False
        steps = 0
        info = {}
        while not done and steps < self.max_steps:
            action, _ = self.model.predict(obs, deterministic=greedy)
            obs, _, term, trunc, info = self.env.step(int(action))
            obs = np.asarray(obs, dtype=np.float32)
            done = term or trunc
            steps += 1
        return SolveResult(
            success=bool(info.get("is_success", False)),
            steps=steps)
