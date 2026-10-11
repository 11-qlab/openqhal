"""
End-to-end atom rearrangement pipeline.

Classical planner + deterministic verifier + optional RL fallback.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from .planner import ClassicalPlanner, Move


@dataclass
class PipelineResult:
    success: bool
    moves: List[Move]
    solver: str
    n_moves: int
    verified: bool
    final_grid: np.ndarray


class RearrangementPipeline:
    """
    Solve atom rearrangement with the best tool available.

    Order:
      1. Classical planner (fast, exact on solvable instances)
      2. RL policy as a fallback (best-effort)
      3. Verification always
    """

    def __init__(self, H: int, W: int,
                 rl_model=None, rl_env=None,
                 max_moves: int = 200):
        self.H = H
        self.W = W
        self.planner = ClassicalPlanner(H, W, max_moves=max_moves)
        self.rl_model = rl_model
        self.rl_env = rl_env
        self.max_moves = max_moves

    def solve(self, initial: np.ndarray,
              target: np.ndarray) -> PipelineResult:
        # ── Stage 1: classical ──
        report = self.planner.plan_and_verify(initial, target)
        if report["verified_success"]:
            return PipelineResult(
                success=True,
                moves=report["moves"],
                solver="classical",
                n_moves=report["n_moves"],
                verified=True,
                final_grid=report["final_grid"])

        # ── Stage 2: RL fallback ──
        if self.rl_model is not None and self.rl_env is not None:
            rl_moves = self._rl_attempt(initial, target)
            if rl_moves is not None:
                final = self.planner.simulate(initial, rl_moves)
                if np.array_equal(final, target):
                    return PipelineResult(
                        success=True,
                        moves=rl_moves,
                        solver="rl",
                        n_moves=len(rl_moves),
                        verified=True,
                        final_grid=final)

        # ── Stage 3: give up ──
        return PipelineResult(
            success=False,
            moves=[],
            solver="failed",
            n_moves=0,
            verified=False,
            final_grid=initial)

    def _rl_attempt(self, initial, target) -> Optional[List[Move]]:
        import torch
        self.rl_env.state = torch.tensor(initial, dtype=torch.int8)
        self.rl_env.target = torch.tensor(target, dtype=torch.int8)
        self.rl_env.steps = 0

        obs = np.asarray(self.rl_env._get_obs(), dtype=np.float32)
        moves: List[Move] = []
        for _ in range(self.max_moves):
            action, _ = self.rl_model.predict(obs, deterministic=True)
            ax, idx, d, m = self.rl_env.decode_action(int(action))
            moves.append(Move(ax, idx, d, m))
            obs, _, term, trunc, info = self.rl_env.step(int(action))
            obs = np.asarray(obs, dtype=np.float32)
            if info.get("is_success"):
                return moves
            if term or trunc:
                break
        return None
