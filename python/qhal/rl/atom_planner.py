"""
Individual atom move planner (optimized).

Action space: (source cell, direction) -> move atom one cell.
Only legal if the destination is empty.

Optimization: compute Hungarian matching ONCE per step. Only
consider moves for atoms that are not at their assigned target.
Cuts per-step work from H*W*4 candidates to ~80.
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
    direction: int

    def __repr__(self):
        names = ["up", "down", "left", "right"]
        return f"AtomMove(({self.row},{self.col}) {names[self.direction]})"

    def to_dict(self):
        return {"row": self.row, "col": self.col,
                "direction": self.direction}


class AtomPlanner:

    def __init__(self, H: int, W: int,
                 max_moves: int = 500,
                 n_restarts: int = 5,
                 top_k: int = 8):
        self.H = H
        self.W = W
        self.max_moves = max_moves
        self.n_restarts = n_restarts
        self.top_k = top_k
        self.n_actions = H * W * 4

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
        M, N = len(a), len(t)
        cost = np.zeros((M, N + M))
        cost[:, :N] = np.abs(a[:, None, :] - t[None, :, :]).sum(axis=-1)
        r, c = linear_sum_assignment(cost)
        return float(cost[r, c].sum())

    def _matching(self, grid, target):
        """Return (assignments, current_dist). assignments is list
        of (atom_pos, target_pos)."""
        a = np.argwhere(grid > 0)
        t = np.argwhere(target > 0)
        if len(a) == 0 or len(a) != len(t):
            return None, float("inf")
        D = np.abs(a[:, None, :] - t[None, :, :]).sum(axis=-1)
        r, c = linear_sum_assignment(D)
        dist = float(D[r, c].sum())
        pairs = [(tuple(a[i]), tuple(t[j])) for i, j in zip(r, c)]
        return pairs, dist

    def _greedy_attempt(self, initial, target, rng) -> List[AtomMove]:
        grid = initial.copy()
        moves: List[AtomMove] = []

        for _ in range(self.max_moves):
            if np.array_equal(grid, target):
                return moves

            pairs, cur_dist = self._matching(grid, target)
            if pairs is None:
                return moves

            # Only consider atoms not already at their assigned target
            mismatched = [(ar, ac, tr, tc)
                          for (ar, ac), (tr, tc) in pairs
                          if (ar, ac) != (tr, tc)]

            if not mismatched:
                # Shouldn't happen if not at target, but be safe
                return moves

            # Build candidate list with cheap heuristic
            candidates = []
            for (ar, ac, tr, tc) in mismatched:
                for d in range(4):
                    mv = AtomMove(ar, ac, d)
                    if not self.is_valid(grid, mv):
                        continue
                    dr, dc = DIRS[d]
                    nr, nc = ar + dr, ac + dc
                    old_d = abs(ar - tr) + abs(ac - tc)
                    new_d = abs(nr - tr) + abs(nc - tc)
                    if new_d < old_d:
                        candidates.append((new_d - old_d, mv))

            # Sort by how much the moving atom improves; try top_k
            candidates.sort(key=lambda x: x[0])

            best_mv = None
            best_dist = cur_dist
            for delta, mv in candidates[: self.top_k]:
                ng = self.apply_move(grid, mv)
                d = self.distance(ng, target)
                if d < best_dist:
                    best_dist = d
                    best_mv = mv

            if best_mv is not None:
                grid = self.apply_move(grid, best_mv)
                moves.append(best_mv)
                continue

            # No progress; random escape that doesn't increase distance
            escape_candidates = []
            for (ar, ac, _, _) in mismatched:
                for d in range(4):
                    mv = AtomMove(ar, ac, d)
                    if not self.is_valid(grid, mv):
                        continue
                    ng = self.apply_move(grid, mv)
                    d_after = self.distance(ng, target)
                    if d_after <= cur_dist + 2:
                        escape_candidates.append((mv, ng))
            if not escape_candidates:
                return moves
            idx = int(rng.integers(len(escape_candidates)))
            mv, ng = escape_candidates[idx]
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
