"""
Handcrafted atom rearrangement puzzles.

Unlike random scramble targets, these puzzles test specific skills:
translation, spreading, compression, rotation, mirroring, diagonal
formation, and cross patterns. Each puzzle is parameterized by grid
size and atom count, so the same suite works at 5x5, 10x10, 20x20.

Used to measure generalization beyond the training distribution.
Random scramble=N targets test "can the model undo N random moves."
Handcrafted puzzles test "can the model execute a specific plan."
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Callable
import numpy as np


# ─────────────────────────────────────────────────────────────────
#  Puzzle dataclass
# ─────────────────────────────────────────────────────────────────

@dataclass
class Puzzle:
    name: str
    difficulty: str        # "easy" | "medium" | "hard"
    initial: np.ndarray
    target: np.ndarray
    min_moves_estimate: int    # rough lower bound on solution length

    def render(self) -> str:
        out = [f"{self.name} [{self.difficulty}]"]
        out.append(f"  estimated moves: ≥{self.min_moves_estimate}")
        out.append("  Initial:")
        for row in self.initial:
            out.append("    " + "".join("O" if v else "." for v in row))
        out.append("  Target:")
        for row in self.target:
            out.append("    " + "".join("O" if v else "." for v in row))
        return "\n".join(out)


# ─────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────

def _empty(H, W):
    return np.zeros((H, W), dtype=np.int8)


def _pattern_at(H, W, pattern, top, left):
    """Place a small binary pattern into a larger grid at (top, left)."""
    g = _empty(H, W)
    ph, pw = pattern.shape
    g[top:top+ph, left:left+pw] = pattern
    return g


# ─────────────────────────────────────────────────────────────────
#  Individual puzzle generators (all scale with grid size)
# ─────────────────────────────────────────────────────────────────

def corner_to_corner(H, W):
    """3 atoms in top-left corner → 3 atoms in bottom-right corner."""
    n = 3
    initial = _empty(H, W)
    initial[0, :n] = 1
    target = _empty(H, W)
    target[-1, -n:] = 1
    return Puzzle(
        name="corner_to_corner",
        difficulty="easy",
        initial=initial, target=target,
        min_moves_estimate=2 * H // 3,
    )


def row_to_col(H, W):
    """A row of atoms → a column of atoms."""
    n = min(H, W, 5)
    initial = _empty(H, W)
    initial[H // 2, :n] = 1
    target = _empty(H, W)
    target[:n, W // 2] = 1
    return Puzzle(
        name="row_to_col",
        difficulty="medium",
        initial=initial, target=target,
        min_moves_estimate=n + 2,
    )


def compress_to_corner(H, W):
    """Atoms scattered along the diagonal → all in the top-left."""
    n = min(H, W, 5)
    initial = _empty(H, W)
    for i in range(n):
        initial[i, i] = 1
    target = _empty(H, W)
    target[0, :n] = 1
    return Puzzle(
        name="compress_to_corner",
        difficulty="medium",
        initial=initial, target=target,
        min_moves_estimate=n + 2,
    )


def spread_from_corner(H, W):
    """Atoms clustered at top-left → spread along top row."""
    n = min(H, W, 5)
    initial = _empty(H, W)
    for i in range(n):
        initial[i, 0] = 1
    target = _empty(H, W)
    target[0, :n] = 1
    return Puzzle(
        name="spread_from_corner",
        difficulty="medium",
        initial=initial, target=target,
        min_moves_estimate=n,
    )


def mirror_horizontal(H, W):
    """Asymmetric cluster → its mirror image across vertical axis."""
    n = min(H, W, 4)
    initial = _empty(H, W)
    for i in range(n):
        initial[i, 0] = 1
        if i > 0:
            initial[i, 1] = 1
    target = _empty(H, W)
    for i in range(n):
        target[i, -1] = 1
        if i > 0:
            target[i, -2] = 1
    return Puzzle(
        name="mirror_horizontal",
        difficulty="hard",
        initial=initial, target=target,
        min_moves_estimate=2 * n,
    )


def mirror_vertical(H, W):
    """Asymmetric cluster → its mirror across horizontal axis."""
    n = min(H, W, 4)
    initial = _empty(H, W)
    for j in range(n):
        initial[0, j] = 1
        if j > 0:
            initial[1, j] = 1
    target = _empty(H, W)
    for j in range(n):
        target[-1, j] = 1
        if j > 0:
            target[-2, j] = 1
    return Puzzle(
        name="mirror_vertical",
        difficulty="hard",
        initial=initial, target=target,
        min_moves_estimate=2 * n,
    )


def diagonal_formation(H, W):
    """Row of atoms → diagonal formation."""
    n = min(H, W, 5)
    initial = _empty(H, W)
    initial[0, :n] = 1
    target = _empty(H, W)
    for i in range(n):
        target[i, i] = 1
    return Puzzle(
        name="diagonal_formation",
        difficulty="hard",
        initial=initial, target=target,
        min_moves_estimate=n + 3,
    )


def cross_pattern(H, W):
    """Atoms in a 2x2 block → plus/cross pattern at center."""
    initial = _empty(H, W)
    initial[0, 0] = 1
    initial[0, 1] = 1
    initial[1, 0] = 1
    initial[1, 1] = 1

    target = _empty(H, W)
    cH, cW = H // 2, W // 2
    target[cH, cW] = 1
    target[cH - 1, cW] = 1
    target[cH + 1, cW] = 1
    target[cH, cW - 1] = 1
    # Drop one atom (4 atoms in cross)
    return Puzzle(
        name="cross_pattern",
        difficulty="hard",
        initial=initial, target=target,
        min_moves_estimate=H + W,
    )


def swap_halves(H, W):
    """Two clusters swap positions."""
    initial = _empty(H, W)
    initial[0, 0] = 1
    initial[0, 1] = 1
    initial[-1, -1] = 1
    initial[-1, -2] = 1

    target = _empty(H, W)
    target[0, -1] = 1
    target[0, -2] = 1
    target[-1, 0] = 1
    target[-1, 1] = 1

    return Puzzle(
        name="swap_halves",
        difficulty="hard",
        initial=initial, target=target,
        min_moves_estimate=H + W,
    )


def ring_pattern(H, W):
    """4 corner atoms → 4 mid-edge atoms (approximate ring)."""
    initial = _empty(H, W)
    initial[0, 0] = 1
    initial[0, -1] = 1
    initial[-1, 0] = 1
    initial[-1, -1] = 1

    target = _empty(H, W)
    target[0, W // 2] = 1
    target[H - 1, W // 2] = 1
    target[H // 2, 0] = 1
    target[H // 2, W - 1] = 1

    return Puzzle(
        name="ring_pattern",
        difficulty="hard",
        initial=initial, target=target,
        min_moves_estimate=H + W,
    )


# ─────────────────────────────────────────────────────────────────
#  Full puzzle suite
# ─────────────────────────────────────────────────────────────────

PUZZLE_GENERATORS: List[Callable[[int, int], Puzzle]] = [
    corner_to_corner,
    row_to_col,
    compress_to_corner,
    spread_from_corner,
    diagonal_formation,
    mirror_horizontal,
    mirror_vertical,
    cross_pattern,
    swap_halves,
    ring_pattern,
]


def make_puzzles(H: int, W: int = None) -> List[Puzzle]:
    """Build the full puzzle suite for a given grid size."""
    if W is None:
        W = H
    return [gen(H, W) for gen in PUZZLE_GENERATORS]


# ─────────────────────────────────────────────────────────────────
#  Evaluation
# ─────────────────────────────────────────────────────────────────

def run_handcrafted_eval(rearranger, H: int, W: int = None,
                         max_steps: int = 50,
                         verbose: bool = False) -> dict:
    """
    Run the model on every handcrafted puzzle. Returns per-puzzle results.

    Args:
        rearranger: AtomRearranger instance (already loaded)
        H, W: grid size
        max_steps: maximum moves allowed per puzzle
        verbose: print each puzzle's outcome

    Returns:
        dict: {puzzle_name: {"success": bool, "steps": int, "difficulty": str}}
    """
    puzzles = make_puzzles(H, W)
    results = {}

    for p in puzzles:
        result = rearranger.solve(p.initial, p.target)
        results[p.name] = {
            "success": result.success,
            "steps": result.steps,
            "difficulty": p.difficulty,
            "min_moves_estimate": p.min_moves_estimate,
        }
        if verbose:
            mark = "OK  " if result.success else "FAIL"
            print(f"    [{mark}] {p.name:<22} "
                  f"{result.steps:2d}/{p.min_moves_estimate} steps  "
                  f"({p.difficulty})")

    # Summary
    n_total = len(results)
    n_solved = sum(1 for r in results.values() if r["success"])
    results["_summary"] = {
        "solved": n_solved,
        "total": n_total,
        "success_rate": n_solved / n_total if n_total else 0.0,
    }
    return results
