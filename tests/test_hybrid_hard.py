"""
Hard-scrambled hybrid planner test.

Scrambler only accepts moves that INCREASE displacement from the
initial state. This forces atoms to walk away from home, producing
genuinely hard rearrangement instances (displacement 40-80 instead
of 9-14).
"""
import sys
sys.path.insert(0, "python")

import numpy as np
from scipy.optimize import linear_sum_assignment

from qhal.rl.hybrid import HybridPlanner
from qhal.rl.atom_planner import AtomPlanner


def hard_scramble(H, W, n_atoms, n_moves, rng):
    """
    Scramble by moving atoms AWAY from their initial positions.
    Only accepts moves that increase total Hungarian displacement.
    Returns (initial, target, applied_moves).
    """
    planner = AtomPlanner(H, W)

    initial = np.zeros((H, W), dtype=np.int8)
    idx = rng.choice(H * W, n_atoms, replace=False)
    initial.flat[idx] = 1

    def disp(g):
        a = np.argwhere(initial > 0)
        t = np.argwhere(g > 0)
        if len(a) == 0 or len(a) != len(t):
            return 0.0
        D = np.abs(a[:, None, :] - t[None, :, :]).sum(axis=-1)
        r, c = linear_sum_assignment(D)
        return float(D[r, c].sum())

    grid = initial.copy()
    applied = 0
    for _ in range(n_moves):
        legal = [planner.decode(a)
                 for a in range(planner.n_actions)
                 if planner.is_valid(grid, planner.decode(a))]
        if not legal:
            break
        candidates = []
        for mv in legal:
            ng = planner.apply_move(grid, mv)
            if disp(ng) >= disp(grid):
                candidates.append((mv, ng))
        if not candidates:
            break
        idx = int(rng.integers(len(candidates)))
        mv, ng = candidates[idx]
        grid = ng
        applied += 1

    return initial, grid, applied


def measure_displacement(initial, target):
    a = np.argwhere(initial > 0)
    t = np.argwhere(target > 0)
    if len(a) == 0 or len(a) != len(t):
        return 0.0
    D = np.abs(a[:, None, :] - t[None, :, :]).sum(axis=-1)
    r, c = linear_sum_assignment(D)
    return float(D[r, c].sum())


def test_grid(H, W, n_atoms, n_moves, n_trials=20, seed=42):
    planner = HybridPlanner(H, W)
    rng = np.random.default_rng(seed)

    successes = 0
    trivial = 0
    displacements = []
    solver_counts = {"trivial": 0, "rowcol": 0, "hybrid": 0,
                     "atom_only": 0, "failed": 0}

    for _ in range(n_trials):
        initial, target, applied = hard_scramble(
            H, W, n_atoms, n_moves, rng)

        if np.array_equal(initial, target) or applied == 0:
            trivial += 1
            solver_counts["trivial"] += 1
            continue

        d = measure_displacement(initial, target)
        displacements.append(d)

        moves, ok, solver = planner.plan(initial, target)
        if ok:
            successes += 1
        solver_counts[solver] = solver_counts.get(solver, 0) + 1

    real_trials = n_trials - trivial
    rate = successes / real_trials if real_trials > 0 else 0.0
    avg_d = np.mean(displacements) if displacements else 0.0

    occ = n_atoms / (H * W)
    print(f"  {H:2d}x{W:<2d}  {n_atoms:3d} atoms ({occ:5.1%})  "
          f"success {successes:2d}/{real_trials:<2d} = {rate:5.1%}  "
          f"avg_disp={avg_d:5.1f}  "
          f"({solver_counts['rowcol']} rc, "
          f"{solver_counts['hybrid']} hyb, "
          f"{solver_counts['atom_only']} atom, "
          f"{solver_counts['failed']} fail)")
    return rate


def main():
    print("=" * 84)
    print("  Hybrid planner — HARD scrambler (atoms walk away from home)")
    print("=" * 84)

    print("\n[5x5]")
    for n in [5, 10, 15, 20]:
        test_grid(5, 5, n, n_moves=50)

    print("\n[10x10]")
    for n in [10, 20, 40, 60, 80]:
        test_grid(10, 10, n, n_moves=100)

    print("\n[20x20]")
    for n in [20, 40, 80, 160]:
        test_grid(20, 20, n, n_moves=200, n_trials=10)


if __name__ == "__main__":
    main()
