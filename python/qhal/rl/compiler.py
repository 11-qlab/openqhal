"""
RL-based atom rearrangement compiler.

Wraps a trained policy and produces a sequence of AOD moves that
transform an initial atom configuration into a target configuration.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import torch

from .env import AtomArrangementEnv
from .policy import PolicyValueNet
from .reinforce import load_model


@dataclass
class Move:
    """A single AOD move."""
    axis: int          # 0 = row, 1 = column
    index: int         # which row/column
    direction: int     # -1 or +1
    magnitude: int     # 1 or 2

    def __repr__(self):
        ax = "row" if self.axis == 0 else "col"
        d = "-" if self.direction < 0 else "+"
        return f"Move({ax} {self.index} {d}{self.magnitude})"


class RLCompiler:
    """RL-based compiler for atom rearrangement."""

    def __init__(self, model_path: Optional[str] = None, grid_size: int = 4):
        self.grid_size = grid_size
        self.env = AtomArrangementEnv(grid_size=grid_size, n_atoms=6)
        if model_path:
            self.model = load_model(model_path, self.env)
        else:
            self.model = PolicyValueNet(self.env.state_dim,
                                        self.env.n_actions, hidden=128)

    def compile(
        self,
        initial: np.ndarray,
        target: np.ndarray,
        max_steps: int = 20,
        greedy: bool = True,
    ) -> Tuple[List[Move], bool]:
        """
        Compile a rearrangement from `initial` to `target`.

        Returns (moves, success). Success is True if the final
        configuration equals the target.
        """
        # Load into env
        self.env.grid = torch.tensor(initial, dtype=torch.int8)
        self.env.target = torch.tensor(target, dtype=torch.int8)
        self.env.steps = 0
        self.env.prev_dist = self.env._dist()

        moves: List[Move] = []
        state = self.env._obs()
        done = False

        while not done and self.env.steps < max_steps:
            with torch.no_grad():
                logits, _ = self.model(state)
            if greedy:
                action = logits.argmax().item()
            else:
                from torch.distributions import Categorical
                action = Categorical(logits=logits).sample().item()

            axis, index, direction, magnitude = self.env.decode_action(action)
            moves.append(Move(axis, index, direction, magnitude))

            result = self.env.step(action)
            state = result.state
            done = result.done

        success = (self.env._dist() == 0)
        return moves, success


# ── Integration with OpenQHAL ──

def moves_to_pulse_timeline(moves: List[Move],
                            duration_per_move_ns: float = 200.0):
    """
    Convert AOD moves into a pulse timeline compatible with OpenQHAL's
    Schedule format. Each move becomes a single global operation
    (all atoms in the selected row/column move together).
    """
    timeline = []
    for i, m in enumerate(moves):
        timeline.append({
            "step": i,
            "t_start_ns": i * duration_per_move_ns,
            "duration_ns": duration_per_move_ns,
            "operation": "aod_move",
            "axis": "row" if m.axis == 0 else "column",
            "index": m.index,
            "direction": m.direction,
            "magnitude": m.magnitude,
        })
    return timeline
