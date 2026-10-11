"""
Hybrid planner test with proper high-density scrambles.

At high density, row/column moves get blocked. Use atom moves to
scramble instead — they work at any density < 100%.
"""
import sys
sys.path.insert(0, "python")

import numpy as np
from qhal.rl.hybrid import HybridPlanner
from qhal.rl.atom_planner import AtomPlanner, AtomMove


def random_solvable_atom(H, W, n_atoms, n_moves, rng):
    """
    Generate a solvable pair via atom moves.

    The initial state is random. The target is initial scrambled
    by N legal atom moves. This guarantees solvability because
    every atom move is reversible.
    """
    planner = AtomPlanner(H, W)

    initial = np.zeros((H, W), dtype=np.int8)
    idx = rng.choice(H * W, n_atoms, replace=False)
    initial.flat[idx] = 1

    grid = initial.copy()
    applied = 0
    attempts = 0
    max_attempts = n_moves * 200

    while applied < n_moves and attempts < max_attempts:
        attempts += 1
        # collect all valid atom moves
        legal = []
        for a in range(planner.n_actions):
            mv = planner.decode(a)
            if planner.is_valid(grid, mv):
                legal.append(mv)
        if not legal:
            break
        mv = legal[rng.integers(len(legal))]
        grid = planner.apply_move(grid, mv)
        applied += 1

    return initial, grid, applied


def test_grid(H, W, n_atoms, n_moves, n_trials=20, seed=42):
    planner = HybridPlanner(H, W)
    rng = np.random.default_rng(seed)

    successes = 0
    trivial = 0
    solver_counts = {"trivial": 0, "rowcol": 0, "hybrid": 0,
                     "atom_only": 0, "failed": 0}
    scramble_stats = []

    for _ in range(n_trials):
        initial, target, applied = random_solvable_atom(
            H, W, n_atoms, n_moves, rng)
        scramble_stats.append(applied)

        if np.array_equal(initial, target):
            trivial += 1
            solver_counts["trivial"] += 1
            continue

        moves, ok, solver = planner.plan(initial, target)
        if ok:
            successes += 1
        solver_counts[solver] = solver_counts.get(solver, 0) + 1

    occ = n_atoms / (H * W)
    real_trials = n_trials - trivial
    real_rate = successes / real_trials if real_trials > 0 else 0.0
    avg_scramble = np.mean(scramble_stats)

    print(f"  {H:2d}x{W:<2d} {n_atoms:3d} ({occ:5.1%})  "
          f"real {successes:2d}/{real_trials:<2d} = {real_rate:5.1%}  "
          f"trivial {trivial:2d}/{n_trials}  "
          f"avg_scramble={avg_scramble:.1f}/{n_moves}")
    return real_rate


def main():
    print("=" * 78)
    print("  Hybrid planner — real high-density test")
    print("=" * 78)

    print("\n[5x5]")
    for n in [5, 10, 15, 20, 23, 24]:
        test_grid(5, 5, n, n_moves=5)

    print("\n[10x10]")
    for n in [10, 20, 40, 60, 80, 90, 95, 99]:
        test_grid(10, 10, n, n_moves=10)

    print("\n[20x20]")
    for n in [20, 40, 80, 160, 300, 380, 399]:
        test_grid(20, 20, n, n_moves=15, n_trials=10)


if __name__ == "__main__":
    main()
