"""
Inference on the best 5x5 model (Colab float32, 91.2% on 1000 eps).
"""
import sys, os
sys.path.insert(0, "python")
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from qhal.rl.infer import AtomRearranger, evaluate_checkpoint


CHECKPOINTS = [
    "rl_checkpoints/sb3_5x5_final.zip",
    os.path.expanduser("~/qhal_models/best_5x5/sb3_5x5_final.zip"),
]


def find_checkpoint():
    for p in CHECKPOINTS:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "No 5x5 checkpoint found. Looked in:\n  " +
        "\n  ".join(CHECKPOINTS))


def main():
    ckpt = find_checkpoint()
    print(f"Using: {ckpt}")
    print()

    # ── Single puzzle ──
    print("=" * 60)
    print("  Puzzle: move 5 atoms from top row to diagonal")
    print("=" * 60)
    print()

    r = AtomRearranger(ckpt, grid_size=5, n_atoms=5)

    initial = np.zeros((5, 5), dtype=np.int8)
    initial[0, :] = 1

    target = np.zeros((5, 5), dtype=np.int8)
    for i in range(5):
        target[i, i] = 1

    result = r.solve(initial, target)
    print(result.render())
    print()

    # ── Random problems ──
    print("=" * 60)
    print("  20 random 5x5 problems")
    print("=" * 60)
    print()

    wins = 0
    for i in range(20):
        r2 = r.random_problem(seed=300 + i, scramble=3)
        wins += int(r2.success)
        mark = "OK " if r2.success else "FAIL"
        print(f"  [{mark}] seed={300+i}  steps={r2.steps:2d}  "
              f"reward={r2.total_reward:6.2f}")

    print()
    print(f"  Success: {wins}/20 = {wins*5}%")
    print()

    # ── Batch stats ──
    print("=" * 60)
    print("  Batch evaluation (200 episodes per scramble)")
    print("=" * 60)
    print()

    for scramble in [1, 2, 3]:
        stats = evaluate_checkpoint(
            ckpt, grid_size=5, n_atoms=5,
            episodes=200, scramble=scramble)
        print(f"  scramble={scramble}  "
              f"success={stats['success_rate']*100:5.1f}%  "
              f"steps={stats['mean_steps']:.2f}±{stats['std_steps']:.2f}")


if __name__ == "__main__":
    main()
