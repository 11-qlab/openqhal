"""
Atom rearrangement env with probabilistic loss during transport.

Each move has probability `p_loss` of dropping an atom. The episode
succeeds when the surviving atoms all sit on target positions (a
subset of the full target). The agent learns to reach that subset
as fast as possible while losing atoms.

Reward uses partial Hungarian matching: current atoms matched to
any subset of target positions, with unmatched current atoms free.
"""
from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.optimize import linear_sum_assignment


class LossyAtomEnv(gym.Env):
    """
    Same action space as ShapedAtomGym, but atoms can disappear.
    """
    metadata = {"render_modes": []}

    def __init__(self, grid_size=5, n_atoms=5, max_steps=20,
                 scramble=1, shape_scale=1.0, step_penalty=0.01,
                 p_loss=0.02, seed=42):
        super().__init__()
        self.H = grid_size
        self.W = grid_size
        self.n_cells = self.H * self.W
        self.n_atoms = n_atoms
        self.max_steps = max_steps
        self.target_scramble = scramble
        self.shape_scale = shape_scale
        self.step_penalty = step_penalty
        self.p_loss = p_loss

        self.n_axes = 2
        self.n_idx = max(self.H, self.W)
        self.n_actions = self.n_axes * self.n_idx * 2 * 2

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(3, self.H, self.W), dtype=np.float32)
        self.action_space = spaces.Discrete(self.n_actions)

        self.rng = np.random.default_rng(seed)
        self.grid = None
        self.target = None
        self.steps = 0
        self.prev_dist = 0.0

    # ── action decode ──
    def decode_action(self, action):
        a = int(action)
        mag = a % 2;  a //= 2
        d   = a % 2;  a //= 2
        idx = a % self.n_idx;  a //= self.n_idx
        return a, idx, (-1 if d == 0 else 1), (mag + 1)

    def is_valid(self, mv):
        ax, idx, d, m = mv
        limit = self.H if ax == 0 else self.W
        if idx >= limit:
            return False
        new = idx + d * m
        return 0 <= new < limit

    # ── move + loss ──
    def _apply_move(self, grid, mv):
        import torch
        ax, idx, d, m = mv
        shift = d * m
        g = grid.clone()
        if ax == 0:
            if 0 <= idx < self.H and 0 <= idx + shift < self.H:
                row = g[idx].clone()
                # loss on the atoms being moved
                mask = (torch.rand_like(row.float()) > self.p_loss).int()
                row = row * mask
                g[idx] = 0
                g[idx + shift] |= row
        else:
            if 0 <= idx < self.W and 0 <= idx + shift < self.W:
                col = g[:, idx].clone()
                mask = (torch.rand_like(col.float()) > self.p_loss).int()
                col = col * mask
                g[:, idx] = 0
                g[:, idx + shift] |= col
        return g

    # ── partial matching distance ──
    def _distance(self, grid, target):
        a = np.argwhere(grid.numpy() > 0)
        t = np.argwhere(target.numpy() > 0)
        if len(a) == 0:
            return float("inf")
        if len(a) == len(t):
            D = np.abs(a[:, None, :] - t[None, :, :]).sum(axis=-1)
            r, c = linear_sum_assignment(D)
            return float(D[r, c].sum())
        # Partial: M < N. Pad targets with M free columns at cost 0.
        M, N = len(a), len(t)
        cost = np.zeros((M, N + M))
        cost[:, :N] = np.abs(a[:, None, :] - t[None, :, :]).sum(axis=-1)
        r, c = linear_sum_assignment(cost)
        return float(cost[r, c].sum())

    def _obs(self):
        s = self.grid.float().numpy()
        t = self.target.float().numpy()
        return np.stack([s, t, s - t], axis=0).astype(np.float32)

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
            mv = self.decode_action(a)
            if self.is_valid(mv):
                tgt = self._apply_move_no_loss(tgt, mv)
        self.target = tgt
        self.prev_dist = self._distance(self.grid, self.target)
        return self._obs(), {}

    def _apply_move_no_loss(self, grid, mv):
        import torch
        ax, idx, d, m = mv
        shift = d * m
        g = grid.clone()
        if ax == 0 and 0 <= idx < self.H and 0 <= idx + shift < self.H:
            row = g[idx].clone()
            g[idx] = 0
            g[idx + shift] |= row
        elif ax == 1 and 0 <= idx < self.W and 0 <= idx + shift < self.W:
            col = g[:, idx].clone()
            g[:, idx] = 0
            g[:, idx + shift] |= col
        return g

    def set_curriculum(self, p_loss=None, scramble=None, max_steps=None):
        if p_loss is not None:    self.p_loss = p_loss
        if scramble is not None:  self.target_scramble = scramble
        if max_steps is not None: self.max_steps = max_steps

    def step(self, action):
        mv = self.decode_action(int(action))
        if self.is_valid(mv):
            self.grid = self._apply_move(self.grid, mv)
        self.steps += 1

        new_dist = self._distance(self.grid, self.target)
        terminated = (new_dist == 0)
        truncated  = self.steps >= self.max_steps

        reward = self.shape_scale * (self.prev_dist - new_dist)
        reward -= self.step_penalty
        if terminated:
            reward += 10.0
        self.prev_dist = new_dist

        return self._obs(), float(reward), terminated, truncated, {
            "is_success": terminated,
            "atoms_remaining": int(self.grid.sum().item())}
