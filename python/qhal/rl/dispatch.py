"""Solver dispatcher: A* for dense small grids, greedy elsewhere."""
from __future__ import annotations
from typing import List, Tuple
import numpy as np

from .planner import ClassicalPlanner, Move
from .exact import astar


def solve(initial: np.ndarray,
          target: np.ndarray,
          max_astar_nodes: int = 300_000
          ) -> Tuple[List[Move], bool, str]:
    H, W = initial.shape
    cells = H * W
    n_atoms = int(initial.sum())
    occupancy = n_atoms / cells

    if occupancy > 0.25 and cells <= 100:
        moves, ok = astar(initial, target, max_nodes=max_astar_nodes)
        if ok:
            return moves, True, "astar"

    planner = ClassicalPlanner(H, W, n_restarts=8)
    moves, ok = planner.plan(initial, target)
    if ok:
        return moves, True, "greedy"

    if cells <= 400:
        moves, ok = astar(initial, target, max_nodes=max_astar_nodes * 4)
        if ok:
            return moves, True, "astar_fallback"

    return [], False, "failed"
