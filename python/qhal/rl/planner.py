"""
Classical atom rearrangement planner with multi-restart search.

Inner loop: greedy best-improvement move selection with random
escape when stuck. Outer loop: retry with different escape seeds.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Move:
    axis: int
    index: int
    direction: int
    magnitude: int

    def __repr__(self):
        ax = "row" if self.axis == 0 else "col"
        s = "-" if self.direction < 0 else "+"
        return f"Move({ax} {self.index} {s}{self.magnitude})"

    def to_dict(self):
        return {"axis": self.axis, "index": self.index,
                "direction": self.direction,
                "magnitude": self.magnitude}


class ClassicalPlanner:

    def __init__(self, H: int, W: int,
                 max_moves: int = 200,
                 n_restarts: int = 5,
                 seed: int = 0):
        self.H = H
        self.W = W
        self.max_moves = max_moves
        self.n_restarts = n_restarts
        self.rng = np.random.default_rng(seed)

        self.n_axes = 2
        self.n_idx = max(H, W)
        self.n_actions = self.n_axes * self.n_idx * 2 * 2

    # ─────────────────────────────────────────────────────────────
    def decode(self, action: int) -> Move:
        a = int(action)
        mag = a % 2;  a //= 2
        d   = a % 2;  a //= 2
        idx = a % self.n_idx;  a //= self.n_idx
        ax  = a
        return Move(ax, idx, -1 if d == 0 else 1, mag + 1)

    def is_valid(self, mv: Move) -> bool:
        limit = self.H if mv.axis == 0 else self.W
        if mv.index >= limit:
            return False
        new = mv.index + mv.direction * mv.magnitude
        return 0 <= new < limit

    @staticmethod
    def apply_move(grid: np.ndarray, mv: Move) -> np.ndarray:
        g = grid.copy()
        H, W = g.shape
        shift = mv.direction * mv.magnitude
        if mv.axis == 0:
            if 0 <= mv.index < H and 0 <= mv.index + shift < H:
                row = g[mv.index].copy()
                g[mv.index] = 0
                g[mv.index + shift] |= row
        else:
            if 0 <= mv.index < W and 0 <= mv.index + shift < W:
                col = g[:, mv.index].copy()
                g[:, mv.index] = 0
                g[:, mv.index + shift] |= col
        return g

    def simulate(self, initial: np.ndarray,
                 moves: List[Move]) -> np.ndarray:
        g = initial.copy()
        for mv in moves:
            g = self.apply_move(g, mv)
        return g

    @staticmethod
    def distance(grid: np.ndarray, target: np.ndarray) -> float:
        a = np.argwhere(grid > 0)
        t = np.argwhere(target > 0)
        if len(a) == 0 or len(a) != len(t):
            return float("inf")
        D = np.abs(a[:, None, :] - t[None, :, :]).sum(axis=-1)
        r, c = linear_sum_assignment(D)
        return float(D[r, c].sum())

    # ─────────────────────────────────────────────────────────────
    #  Single greedy attempt
    # ─────────────────────────────────────────────────────────────
    def _greedy_attempt(self, initial, target, rng) -> List[Move]:
        grid = initial.copy()
        moves: List[Move] = []
        stall = 0

        for _ in range(self.max_moves):
            if np.array_equal(grid, target):
                return moves

            cur_dist = self.distance(grid, target)
            best_dist = cur_dist
            best_move = None
            best_grid = None

            # Evaluate all moves
            for action in range(self.n_actions):
                mv = self.decode(action)
                if not self.is_valid(mv):
                    continue
                ng = self.apply_move(grid, mv)
                if ng.sum() != grid.sum():
                    continue
                d = self.distance(ng, target)
                if d < best_dist:
                    best_dist = d
                    best_move = mv
                    best_grid = ng

            if best_move is None:
                # Random escape: pick a legal move that doesn't drop atoms
                stall += 1
                if stall > 3:
                    return moves  # give up on this restart
                candidates = []
                for action in range(self.n_actions):
                    mv = self.decode(action)
                    if not self.is_valid(mv):
                        continue
                    ng = self.apply_move(grid, mv)
                    if ng.sum() == grid.sum() and not np.array_equal(ng, grid):
                        candidates.append((mv, ng))
                if not candidates:
                    return moves
                idx = int(rng.integers(len(candidates)))
                mv, ng = candidates[idx]
                grid = ng
                moves.append(mv)
                continue

            stall = 0
            grid = best_grid
            moves.append(best_move)

        return moves if np.array_equal(grid, target) else []

    # ─────────────────────────────────────────────────────────────
    #  Multi-restart
    # ─────────────────────────────────────────────────────────────
    def plan(self, initial, target) -> Tuple[List[Move], bool]:
        for restart in range(self.n_restarts):
            rng = np.random.default_rng(restart)
            moves = self._greedy_attempt(initial, target, rng)
            if moves is not None and np.array_equal(
                    self.simulate(initial, moves), target):
                return moves, True
        return [], False

    def plan_and_verify(self, initial, target) -> dict:
        moves, _ = self.plan(initial, target)
        sim = self.simulate(initial, moves)
        return {
            "moves": moves,
            "verified_success": bool(np.array_equal(sim, target)),
            "n_moves": len(moves),
            "final_grid": sim,
            "target_grid": target,
        }
