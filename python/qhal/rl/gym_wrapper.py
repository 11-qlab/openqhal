"""
Gym.Env wrapper around ShapedAtomEnv for Stable-Baselines3.

Provides the (obs, reward, terminated, truncated, info) API that SB3
expects, while delegating the actual physics to the existing env.

The wrapper also exposes set_curriculum() to change scramble and
horizon between stages without rebuilding the env.
"""
from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .env import AtomArrangementEnv


class ShapedAtomGym(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, grid_size: int, n_atoms: int,
                 max_steps: int = 12, scramble: int = 1,
                 shape_scale: float = 0.15, step_penalty: float = 0.01,
                 seed: int = None):
        super().__init__()
        self.grid_size = grid_size
        self.H = grid_size
        self.W = grid_size

        self.env = AtomArrangementEnv(
            grid_size=grid_size, n_atoms=n_atoms,
            max_steps=max_steps, target_scramble=scramble,
            seed=seed)

        self.max_steps = max_steps
        self.shape_scale = shape_scale
        self.step_penalty = step_penalty
        self._last_potential = 0.0

        # Spaces — SB3 needs these to build the model
        state_dim = self.env.state_dim
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(state_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.env.n_actions)

    # ── potential via Manhattan bipartite matching ──
    def _potential(self, obs: np.ndarray) -> float:
        area = self.H * self.W
        curr = obs[:area].reshape(self.H, self.W) > 0.5
        targ = obs[area:2 * area].reshape(self.H, self.W) > 0.5
        ci = np.argwhere(curr)
        ti = np.argwhere(targ)
        if len(ci) == 0 or len(ci) != len(ti):
            return 0.0
        dists = np.abs(ci[:, None, :] - ti[None, :, :]).sum(axis=-1)
        # Greedy assignment (faster than full Hungarian and adequate)
        order = np.argsort(dists, axis=1)
        used = set()
        total = 0
        for i in range(len(ci)):
            for j in order[i]:
                if j not in used:
                    used.add(j)
                    total += dists[i, j]
                    break
        return -float(total)

    # ── curriculum control ──
    def set_curriculum(self, scramble: int = None, max_steps: int = None,
                        shape_scale: float = None):
        if scramble is not None:
            self.env.target_scramble = scramble
        if max_steps is not None:
            self.env.max_steps = max_steps
            self.max_steps = max_steps
        if shape_scale is not None:
            self.shape_scale = shape_scale

    # ── gym API ──
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.env.rng = np.random.default_rng(seed)
        obs = self.env.reset()
        self._last_potential = self._potential(np.asarray(obs))
        return np.asarray(obs, dtype=np.float32), {}

    def step(self, action: int):
        r = self.env.step(int(action))
        obs = np.asarray(r.state, dtype=np.float32)
        new_potential = self._potential(obs)

        # Potential-based shaping (Ng et al. 1999)
        potential_delta = new_potential - self._last_potential
        self._last_potential = new_potential
        bonus = float(self.shape_scale * potential_delta) - self.step_penalty

        reward = float(r.reward) + bonus
        terminated = bool(r.done and r.info.get("success", False))
        truncated = bool(r.done and not r.info.get("success", False))
        info = {"is_success": bool(r.info.get("success", False)),
                "distance": int(r.info.get("distance", -1)),
                "steps": int(r.info.get("steps", 0))}
        return obs, reward, terminated, truncated, info

    def action_mask(self) -> np.ndarray:
        """Boolean mask: True where action is valid."""
        return self.env.action_mask().numpy()


# Backwards-compat alias
ShapedAtomEnv = ShapedAtomGym
