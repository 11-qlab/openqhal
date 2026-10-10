"""Handcrafted atom rearrangement puzzle suite."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class Puzzle:
    name: str
    difficulty: str
    initial: np.ndarray
    target: np.ndarray
    min_moves_estimate: int


def _empty(H, W):
    return np.zeros((H, W), dtype=np.int8)


def corner_to_corner(H, W):
    n = min(H, W, 3)
    i = _empty(H, W); i[0, :n] = 1
    t = _empty(H, W); t[-1, -n:] = 1
    return Puzzle("corner_to_corner", "easy", i, t, 2 * H // 3)


def row_to_col(H, W):
    n = min(H, W, 5)
    i = _empty(H, W); i[H // 2, :n] = 1
    t = _empty(H, W); t[:n, W // 2] = 1
    return Puzzle("row_to_col", "medium", i, t, n + 2)


def compress_to_corner(H, W):
    n = min(H, W, 5)
    i = _empty(H, W)
    for k in range(n): i[k, k] = 1
    t = _empty(H, W); t[0, :n] = 1
    return Puzzle("compress_to_corner", "medium", i, t, n + 2)


def spread_from_corner(H, W):
    n = min(H, W, 5)
    i = _empty(H, W)
    for k in range(n): i[k, 0] = 1
    t = _empty(H, W); t[0, :n] = 1
    return Puzzle("spread_from_corner", "medium", i, t, n)


def mirror_horizontal(H, W):
    n = min(H, W, 4)
    i = _empty(H, W)
    for k in range(n):
        i[k, 0] = 1
        if k > 0: i[k, 1] = 1
    t = _empty(H, W)
    for k in range(n):
        t[k, -1] = 1
        if k > 0: t[k, -2] = 1
    return Puzzle("mirror_horizontal", "hard", i, t, 2 * n)


def mirror_vertical(H, W):
    n = min(H, W, 4)
    i = _empty(H, W)
    for k in range(n):
        i[0, k] = 1
        if k > 0: i[1, k] = 1
    t = _empty(H, W)
    for k in range(n):
        t[-1, k] = 1
        if k > 0: t[-2, k] = 1
    return Puzzle("mirror_vertical", "hard", i, t, 2 * n)


def diagonal_formation(H, W):
    n = min(H, W, 5)
    i = _empty(H, W); i[0, :n] = 1
    t = _empty(H, W)
    for k in range(n): t[k, k] = 1
    return Puzzle("diagonal_formation", "hard", i, t, n + 3)


def cross_pattern(H, W):
    i = _empty(H, W)
    i[0,0] = i[0,1] = i[1,0] = i[1,1] = 1
    t = _empty(H, W)
    cH, cW = H // 2, W // 2
    t[cH, cW] = 1; t[cH-1, cW] = 1; t[cH+1, cW] = 1; t[cH, cW-1] = 1
    return Puzzle("cross_pattern", "hard", i, t, H + W)


def swap_halves(H, W):
    i = _empty(H, W)
    i[0,0] = i[0,1] = i[-1,-1] = i[-1,-2] = 1
    t = _empty(H, W)
    t[0,-1] = t[0,-2] = t[-1,0] = t[-1,1] = 1
    return Puzzle("swap_halves", "hard", i, t, H + W)


def ring_pattern(H, W):
    i = _empty(H, W)
    i[0,0] = i[0,-1] = i[-1,0] = i[-1,-1] = 1
    t = _empty(H, W)
    t[0, W//2] = 1; t[H-1, W//2] = 1
    t[H//2, 0] = 1; t[H//2, W-1] = 1
    return Puzzle("ring_pattern", "hard", i, t, H + W)


PUZZLE_GENERATORS = [
    corner_to_corner, row_to_col, compress_to_corner, spread_from_corner,
    diagonal_formation, mirror_horizontal, mirror_vertical,
    cross_pattern, swap_halves, ring_pattern,
]


def make_puzzles(H, W=None) -> List[Puzzle]:
    if W is None: W = H
    return [gen(H, W) for gen in PUZZLE_GENERATORS]


def run_handcrafted_eval(rearranger, H, W=None, verbose=False) -> dict:
    puzzles = make_puzzles(H, W)
    results = {}
    for p in puzzles:
        r = rearranger.solve(p.initial, p.target)
        results[p.name] = {"success": r.success, "steps": r.steps,
                            "difficulty": p.difficulty}
        if verbose:
            mark = "OK  " if r.success else "FAIL"
            print(f"    [{mark}] {p.name:<22} {r.steps:2d} steps "
                  f"({p.difficulty})")
    solved = sum(1 for x in results.values() if x["success"])
    results["_summary"] = {"solved": solved, "total": len(puzzles),
                            "success_rate": solved / len(puzzles)}
    return results
