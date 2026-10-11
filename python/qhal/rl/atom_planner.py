"""
Individual atom move planner.

Action space: (source cell, direction) -> move the atom at source
one cell in direction. Only legal if the destination is empty.
Never drops atoms, never collides.

Works at any occupancy < 100%. Slower per atom than row/column
shifts, but strictly more powerful.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment


DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


@dataclass
class AtomMove:
    row: int
    col: int
    direction: int  # 0=up, 1=down, 2=left, 3=right

    def __repr__(self):
        names = ["up", "down", "left", "right"]
        return f"AtomMove(({self.row},{self.col}) {names[self.direction]})"

    def to_dict(self):
        return {"row": self.row, "col": self.col,
                "direction": self.direction}


class AtomPlanner:

    def __init__(self, H: int, W: int,
                 max_moves: int = 500,
                 n_restarts: int = 5):
        self.H = H
        self.W = W
        self.max_moves = max_moves
        self.n_restarts = n_restarts
        self.n_actions = H * W * 4

    # ─────────────────────────────────────────────────────────────
    def decode(self, action: int) -> AtomMove:
        a = int(action)
        d = a % 4
        cell = a // 4
        r, c = divmod(cell, self.W)
        return AtomMove(r, c, d)

    def is_valid(self, grid: np.ndarray, mv: AtomMove) -> bool:
        H, W = grid.shape
        r, c = mv.row, mv.col
        if not (0 <= r < H and 0 <= c < W):
            return False
        if grid[r, c] == 0:
            return False
        dr, dc = DIRS[mv.direction]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < H and 0 <= nc < W):
            return False
        return grid[nr, nc] == 0

    @staticmethod
    def apply_move(grid: np.ndarray, mv: AtomMove) -> np.ndarray:
        g = grid.copy()
        dr, dc = DIRS[mv.direction]
        g[mv.row + dr, mv.col + dc] = 1
        g[mv.row, mv.col] = 0
        return g

    def simulate(self, initial, moves) -> np.ndarray:
        g = initial.copy()
        for m in moves:
            g = self.apply_move(g, m)
        return g

    # ─────────────────────────────────────────────────────────────
    @staticmethod
    def distance(grid: np.ndarray, target: np.ndarray) -> float:
        a = np.argwhere(grid > 0)
        t = np.argwhere(target > 0)
        if len(a) == 0:
            return float("inf")
        if len(a) == len(t):
            D = np.abs(a[:, None, :] - t[None, :, :]).sum(axis=-1)
            r, c = linear_sum_assignment(D)
            return float(D[r, c].sum())
        # Partial matching (fewer current atoms than targets)
        M, N = len(a), len(t)
        cost = np.zeros((M, N + M))
        cost[:, :N] = np.abs(a[:, None, :] - t[None, :, :]).sum(axis=-1)
        r, c = linear_sum_assignment(cost)
        return float(cost[r, c].sum())

    # ─────────────────────────────────────────────────────────────
    def _greedy_attempt(self, initial, target, rng) -> List[AtomMove]:
        grid = initial.copy()
        moves: List[AtomMove] = []

        for _ in range(self.max_moves):
            if np.array_equal(grid, target):
                return moves
            cur = self.distance(grid, target)
            best_d = cur
            best_mv = None
            best_g = None

            for action in range(self.n_actions):
                mv = self.decode(action)
                if not self.is_valid(grid, mv):
                    continue
                ng = self.apply_move(grid, mv)
                d = self.distance(ng, target)
                if d < best_d:
                    best_d = d
                    best_mv = mv
                    best_g = ng

            if best_mv is not None:
                grid = best_g
                moves.append(best_mv)
                continue

            # No improving move. Try an escape: pick a random legal move
            # that doesn't increase distance by more than 2.
            candidates = []
            for action in range(self.n_actions):
                mv = self.decode(action)
                if not self.is_valid(grid, mv):
                    continue
                ng = self.apply_move(grid, mv)
                d = self.distance(ng, target)
                if d <= cur + 2:
                    candidates.append((mv, ng))
            if not candidates:
                return moves
            idx = int(rng.integers(len(candidates)))
            mv, ng = candidates[idx]
            grid = ng
            moves.append(mv)

        return moves

    def plan(self, initial, target):
        for restart in range(self.n_restarts):
            rng = np.random.default_rng(restart)
            moves = self._greedy_attempt(initial, target, rng)
            if np.array_equal(self.simulate(initial, moves), target):
                return moves, True
        return [], False
