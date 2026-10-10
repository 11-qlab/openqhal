"""
AOD rearrangement simulator using the trained 5x5 RL model.

Applies the model's move sequence to a realistic tweezer array
with configurable atom loss per move, measures final match.
"""
import sys, os
sys.path.insert(0, "python")
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


class TweezerArray:
    def __init__(self, H, W, seed=42):
        self.H = H
        self.W = W
        self.rng = np.random.default_rng(seed)
        self.occupation = np.zeros((H, W), dtype=bool)
        self.loss_log = []

    def move_row(self, index, direction, magnitude, loss_rate):
        if not (0 <= index < self.H):
            return
        new_index = index + direction * magnitude
        if not (0 <= new_index < self.H):
            return

        row = self.occupation[index].copy()
        for j in range(self.W):
            if row[j] and self.rng.random() < loss_rate:
                row[j] = False
                self.loss_log.append((index, j))

        self.occupation[index] = False
        self.occupation[new_index] |= row

    def move_col(self, index, direction, magnitude, loss_rate):
        if not (0 <= index < self.W):
            return
        new_index = index + direction * magnitude
        if not (0 <= new_index < self.W):
            return

        col = self.occupation[:, index].copy()
        for i in range(self.H):
            if col[i] and self.rng.random() < loss_rate:
                col[i] = False
                self.loss_log.append((i, index))

        self.occupation[:, index] = False
        self.occupation[:, new_index] |= col


def apply_moves(arr, moves, loss_rate):
    for m in moves:
        if m.axis == 0:
            arr.move_row(m.index, m.direction, m.magnitude, loss_rate)
        else:
            arr.move_col(m.index, m.direction, m.magnitude, loss_rate)
    return arr


def run_trial(initial, target, moves, loss_rate, seed):
    arr = TweezerArray(5, 5, seed=seed)
    arr.occupation = initial.astype(bool)
    apply_moves(arr, moves, loss_rate)

    target_bool = target.astype(bool)
    tp = (arr.occupation & target_bool).sum()
    fp = (arr.occupation & ~target_bool).sum()
    fn = (~arr.occupation & target_bool).sum()
    if tp == 0:
        return 0.0, len(arr.loss_log)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    if precision + recall == 0:
        return 0.0, len(arr.loss_log)
    f1 = 2 * precision * recall / (precision + recall)
    return f1, len(arr.loss_log)


def main():
    from qhal.rl.infer import AtomRearranger

    r = AtomRearranger(
        "models/best_5x5_float64/sb3_5x5_final.zip",
        grid_size=5, n_atoms=5)

    print("=" * 60)
    print("  AOD simulation with realistic loss")
    print("=" * 60)

    loss_rates = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10]
    trials = 100

    print(f"\n{trials} problems per loss rate\n")
    print(f"{'loss/step':>10}  {'mean match':>12}  {'std':>8}  {'atoms lost':>12}")
    print("-" * 50)

    for lr in loss_rates:
        matches, losses = [], []
        for trial in range(trials):
            result = r.random_problem(seed=trial, scramble=6)
            if not result.success:
                continue
            m, l = run_trial(
                result.initial, result.target,
                result.moves, lr, seed=1000 + trial)
            matches.append(m)
            losses.append(l)
        if matches:
            print(f"{lr:>10.3f}  {np.mean(matches):>12.4f}  "
                  f"{np.std(matches):>8.4f}  {np.mean(losses):>12.2f}")


if __name__ == "__main__":
    main()
