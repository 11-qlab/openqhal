"""
Real-time loss recovery wrapper.

Classical planner produces the initial plan. When hardware reports
an atom-loss event, the RL policy suggests the next move in ~1 ms
and the classical planner is re-run to close the remaining gap.

Classical replanning:    5-200 ms
RL policy forward pass:  1-5 ms
Measured speedup (TBD):  to be benchmarked
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import torch
import time

from .planner import ClassicalPlanner, Move


@dataclass
class RecoveryEvent:
    step: int
    atoms_before: int
    atoms_after: int
    rl_latency_ms: float
    classical_latency_ms: float
    solver_used: str


class RealTimeRecovery:
    def __init__(self, H: int, W: int, rl_model, rl_env,
                 loss_threshold: int = 1):
        self.H, self.W = H, W
        self.planner = ClassicalPlanner(H, W, n_restarts=2)
        self.rl_model = rl_model
        self.rl_env = rl_env
        self.loss_threshold = loss_threshold

    def _rl_next_move(self, grid: np.ndarray,
                      target: np.ndarray) -> Move:
        self.rl_env.grid = torch.tensor(grid, dtype=torch.int8)
        self.rl_env.target = torch.tensor(target, dtype=torch.int8)
        self.rl_env.steps = 0
        obs = np.asarray(self.rl_env._get_obs(), dtype=np.float32)
        with torch.no_grad():
            action, _ = self.rl_model.predict(obs, deterministic=True)
        ax, idx, d, m = self.rl_env.decode_action(int(action))
        return Move(ax, idx, d, m)

    def plan_with_recovery(self, initial: np.ndarray,
                           target: np.ndarray,
                           max_total_moves: int = 100):
        moves, _ = self.planner.plan(initial, target)
        current = initial.copy()
        events: List[RecoveryEvent] = []
        applied: List[Move] = []

        for step, mv in enumerate(moves):
            if len(applied) >= max_total_moves:
                break
            atoms_before = int(current.sum())
            current = self.planner.apply_move(current, mv)
            atoms_after = int(current.sum())

            if atoms_before - atoms_after >= self.loss_threshold:
                t0 = time.perf_counter()
                rl_move = self._rl_next_move(current, target)
                rl_ms = (time.perf_counter() - t0) * 1000

                t0 = time.perf_counter()
                _, _ = self.planner.plan(current, target)
                classical_ms = (time.perf_counter() - t0) * 1000

                events.append(RecoveryEvent(
                    step=step,
                    atoms_before=atoms_before,
                    atoms_after=atoms_after,
                    rl_latency_ms=rl_ms,
                    classical_latency_ms=classical_ms,
                    solver_used="rl"))

                applied.append(rl_move)
                current = self.planner.apply_move(current, rl_move)
            else:
                applied.append(mv)

            if np.array_equal(current, target):
                return applied, True, events

        return applied, False, events
