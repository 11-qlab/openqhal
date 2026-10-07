"""
Neutral atom rearrangement env with curriculum support.

Key change: target_scramble is now mutable, so the training loop can
increase difficulty over time (1-move targets -> 2-move -> 3-move).
Reward is stronger so useful moves dominate the step penalty.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import torch


@dataclass
class StepResult:
    state: torch.Tensor
    reward: float
    done: bool
    info: dict


class AtomArrangementEnv:
    def __init__(
        self,
        grid_size: int = 3,
        n_atoms: int = 3,
        max_steps: int = 10,
        target_scramble: int = 1,
        seed: Optional[int] = None,
    ):
        self.H = grid_size
        self.W = grid_size
        self.n_atoms = n_atoms
        self.max_steps = max_steps
        self.target_scramble = target_scramble
        self.rng = np.random.default_rng(seed)

        self.n_axes = 2
        self.n_indices = max(self.H, self.W)
        self.n_dirs = 2
        self.n_mags = 2
        self.n_actions = (
            self.n_axes * self.n_indices * self.n_dirs * self.n_mags
        )

        self.grid_size = self.H * self.W
        # State: grid + target + diff + progress
        self.state_dim = 3 * self.grid_size + 1

        self.grid = None
        self.target = None
        self.steps = 0
        self.prev_dist = 0

    def decode_action(self, action: int):
        a = action
        mag = a % self.n_mags;     a //= self.n_mags
        d   = a % self.n_dirs;     a //= self.n_dirs
        idx = a % self.n_indices;  a //= self.n_indices
        ax  = a
        return ax, idx, (-1 if d == 0 else 1), (mag + 1)

    def encode_action(self, axis, index, direction, magnitude) -> int:
        return (
            axis * self.n_indices * self.n_dirs * self.n_mags
            + index * self.n_dirs * self.n_mags
            + (0 if direction == -1 else 1) * self.n_mags
            + (magnitude - 1)
        )

    def _apply_move(self, grid, action: int):
        axis, index, direction, magnitude = self.decode_action(action)
        shift = direction * magnitude
        g = grid.clone()

        if axis == 0:
            if 0 <= index < self.H:
                row = g[index].clone()
                new_index = index + shift
                if 0 <= new_index < self.H:
                    g[index] = 0
                    g[new_index] = (g[new_index] | row)
        else:
            if 0 <= index < self.W:
                col = g[:, index].clone()
                new_index = index + shift
                if 0 <= new_index < self.W:
                    g[:, index] = 0
                    g[:, new_index] = (g[:, new_index] | col)
        return g

    def _obs(self) -> torch.Tensor:
        progress = np.array([self.steps / self.max_steps], dtype=np.float32)
        diff = (self.grid != self.target).flatten().float()
        return torch.cat([
            self.grid.flatten().float(),
            self.target.flatten().float(),
            diff,
            torch.tensor(progress),
        ])

    def _dist(self) -> int:
        return int((self.grid != self.target).sum().item())

    def reset(self) -> torch.Tensor:
        flat = np.zeros(self.grid_size, dtype=np.int8)
        idx = self.rng.choice(self.grid_size, size=self.n_atoms, replace=False)
        flat[idx] = 1
        self.grid = torch.tensor(flat, dtype=torch.int8).reshape(self.H, self.W)

        # Generate reachable target by applying target_scramble random moves
        target = self.grid.clone()
        for _ in range(self.target_scramble):
            action = int(self.rng.integers(0, self.n_actions))
            target = self._apply_move(target, action)

        # Retry if target equals initial (rare but possible)
        attempts = 0
        while (target == self.grid).all() and attempts < 10:
            action = int(self.rng.integers(0, self.n_actions))
            target = self._apply_move(self.grid.clone(), action)
            attempts += 1

        self.target = target
        self.steps = 0
        self.prev_dist = self._dist()
        return self._obs()

    def action_mask(self) -> torch.Tensor:
        """Boolean mask: True where the action is valid on this grid."""
        mask = torch.ones(self.n_actions, dtype=torch.bool)
        for a in range(self.n_actions):
            axis, idx, direction, mag = self.decode_action(a)
            limit = self.H if axis == 0 else self.W
            if idx >= limit:
                mask[a] = False
                continue
            # Out of bounds shift
            new_idx = idx + direction * mag
            if new_idx < 0 or new_idx >= limit:
                mask[a] = False
        return mask

    def step(self, action: int) -> StepResult:
        self.grid = self._apply_move(self.grid, action)
        self.steps += 1
        new_dist = self._dist()

        delta = self.prev_dist - new_dist
        # Amplified potential-based reward (still Ng et al. 1999 form,
        # just scaled 3x so the signal dominates any exploration noise)
        reward = 0.5 * float(delta)

        self.prev_dist = new_dist

        if new_dist == 0:
            reward += 5.0
            done = True
        elif self.steps >= self.max_steps:
            reward -= 1.0              # failure penalty, applied once
            done = True
        else:
            done = False

        info = {"distance": new_dist, "success": new_dist == 0, "steps": self.steps}
        return StepResult(self._obs(), reward, done, info)

    def render_ascii(self) -> str:
        out = []
        for r in range(self.H):
            row = ""
            for c in range(self.W):
                a = int(self.grid[r, c].item())
                t = int(self.target[r, c].item())
                if a and t:   row += "O"
                elif a:       row += "o"
                elif t:       row += "."
                else:         row += " "
            out.append(row)
        return "\n".join(out)
