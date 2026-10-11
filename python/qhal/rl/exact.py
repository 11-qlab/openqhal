"""Exact A* solver for small atom rearrangement instances."""
from __future__ import annotations
from heapq import heappush, heappop
from typing import List, Tuple
import numpy as np
from .planner import ClassicalPlanner, Move


def astar(initial: np.ndarray,
          target: np.ndarray,
          max_nodes: int = 500_000) -> Tuple[List[Move], bool]:
    H, W = initial.shape
    planner = ClassicalPlanner(H, W)

    if initial.sum() != target.sum():
        return [], False

    start_key = initial.tobytes()
    h0 = planner.distance(initial, target)
    heap = [(h0, 0, start_key, initial.copy(), [])]
    visited = {start_key: 0}

    nodes = 0
    while heap and nodes < max_nodes:
        f, g, key, grid, path = heappop(heap)
        nodes += 1

        if np.array_equal(grid, target):
            return path, True

        for action in range(planner.n_actions):
            mv = planner.decode(action)
            if not planner.is_valid(mv):
                continue
            ng = planner.apply_move(grid, mv)
            if ng.sum() != grid.sum():
                continue
            nk = ng.tobytes()
            ng_g = g + 1
            if nk in visited and visited[nk] <= ng_g:
                continue
            visited[nk] = ng_g
            h = planner.distance(ng, target)
            heappush(heap, (ng_g + h, ng_g, nk, ng, path + [mv]))

    return [], False
