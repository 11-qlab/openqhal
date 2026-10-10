"""
Gym wrapper for atom rearrangement on an H x W grid.

Row/column AOD-style actions (2 * max(H,W) * 2 * 2 total),
3-channel observation (state, target, diff), potential-based reward.
"""
from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces


DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


class ShapedAtomGym(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, grid_size: int = 5, n_atoms: int = 5,
                 max_steps: int = 15, scramble: int = 1,
                 shape_scale: float = 0.5, step_penalty: float = 0.01,
                 seed: int = None, W: int = None):
        super().__init__()
        self.H = grid_size
        self.W = W if W is not None else grid_size
        self.n_cells = self.H * self.W
        self.n_atoms = n_atoms
        self.max_steps = max_steps
        self.target_scramble = scramble
        self.shape_scale = shape_scale
        self.step_penalty = step_penalty

        # Row/column AOD-style action space
        self.n_axes = 2
        self.n_indices = max(self.H, self.W)
        self.n_dirs = 2
        self.n_mags = 2
        self.n_actions = self.n_axes * self.n_indices * self.n_dirs * self.n_mags

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3, self.H, self.W), dtype=np.float32)
        self.action_space = spaces.Discrete(self.n_actions)

        self.rng = np.random.default_rng(seed)
        self.grid = None
        self.target = None
        self.steps = 0
        self.prev_dist = 0.0
        self._last_potential = 0.0

    # ── action encoding ──
    def decode_action(self, action):
        a = int(action)
        mag = a % self.n_mags;     a //= self.n_mags
        d   = a % self.n_dirs;     a //= self.n_dirs
        idx = a % self.n_indices;  a //= self.n_indices
        ax  = a
        return ax, idx, (-1 if d == 0 else 1), (mag + 1)

    def action_mask(self):
        mask = np.ones(self.n_actions, dtype=bool)
        for a in range(self.n_actions):
            ax, idx, d, m = self.decode_action(a)
            limit = self.H if ax == 0 else self.W
            if idx >= limit:
                mask[a] = False
                continue
            if idx + d * m < 0 or idx + d * m >= limit:
                mask[a] = False
        return mask

    # ── move ──
    def _apply_move(self, grid, action):
        ax, idx, d, m = self.decode_action(action)
        shift = d * m
        g = grid.clone()
        if ax == 0:
            if 0 <= idx < self.H and 0 <= idx + shift < self.H:
                row = g[idx].clone()
                g[idx] = 0
                g[idx + shift] |= row
        else:
            if 0 <= idx < self.W and 0 <= idx + shift < self.W:
                col = g[:, idx].clone()
                g[:, idx] = 0
                g[:, idx + shift] |= col
        return g

    # ── observation ──
    def _get_obs(self):
        state = self.grid.float().numpy()
        targ = self.target.float().numpy()
        diff = state - targ
        return np.stack([state, targ, diff], axis=0).astype(np.float32)

    def _dist(self):
        import torch
        return float((self.grid != self.target).sum().item())

    # ── gym API ──
    def reset(self, seed=None, options=None):
        import torch
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.steps = 0
        flat = np.zeros(self.n_cells, dtype=np.int8)
        idx = self.rng.choice(self.n_cells, size=self.n_atoms, replace=False)
        flat[idx] = 1
        self.grid = torch.tensor(flat, dtype=torch.int8).reshape(self.H, self.W)

        tgt = self.grid.clone()
        for _ in range(self.target_scramble):
            a = int(self.rng.integers(0, self.n_actions))
            tgt = self._apply_move(tgt, a)
        self.target = tgt
        self.prev_dist = self._dist()
        obs = self._get_obs()
        self._last_potential = -self.prev_dist
        return obs, {}

    def set_curriculum(self, scramble=None, max_steps=None, shape_scale=None):
        if scramble is not None:    self.target_scramble = scramble
        if max_steps is not None:   self.max_steps = max_steps
        if shape_scale is not None: self.shape_scale = shape_scale

    def step(self, action):
        self.grid = self._apply_move(self.grid, action)
        self.steps += 1
        new_dist = self._dist()

        terminated = bool(new_dist == 0)
        truncated  = self.steps >= self.max_steps

        # Potential-based reward
        reward = self.shape_scale * (self.prev_dist - new_dist)
        reward -= self.step_penalty
        if terminated:
            reward += 5.0
        self.prev_dist = new_dist

        info = {"is_success": terminated}
        return self._get_obs(), float(reward), terminated, truncated, info
