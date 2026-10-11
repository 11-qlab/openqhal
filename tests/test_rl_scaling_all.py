"""
Scale the RL compiler across grid sizes: 2x2 through 13x13.

Features:
- Curriculum-scaled horizon (stage_max_steps) to prevent rollout dilution.
- Manhattan distance reward shaping via bipartite matching.
- Dynamic shaping annealing: shape_scale drops to 0 by the final stage.
- Warm-start between stages: model checkpoint saved/loaded per stage.
- Farsighted discount factor (gamma=0.98).
- __setattr__ delegation to ensure horizon limits reach the core env.
"""
import sys, time, argparse, os
sys.path.insert(0, "python")

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from qhal.rl import (
    AtomArrangementEnv, PolicyValueNet,
    PPOTrainer, PPOConfig,
)


CHECKPOINT_DIR = "/tmp/rl_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────
#  Potential-Based Reward Shaping Wrapper
# ─────────────────────────────────────────────────────────────────

class ShapedAtomEnv:
    def __init__(self, env, H, W, shape_scale=0.15, step_penalty=0.01):
        self.env = env
        self.H = H
        self.W = W
        self.shape_scale = shape_scale
        self.step_penalty = step_penalty
        self.last_potential = 0.0
        self._warned = False

    def __setattr__(self, name, value):
        if name in ('env', 'H', 'W', 'shape_scale', 'step_penalty',
                    'last_potential', '_warned'):
            super().__setattr__(name, value)
        else:
            setattr(self.env, name, value)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _compute_potential(self, state):
        try:
            if torch.is_tensor(state):
                state = state.detach().cpu().numpy()

            area = self.H * self.W
            curr_flat = state[:area]
            targ_flat = state[area:2 * area]

            curr_idx = np.argwhere(curr_flat.reshape(self.H, self.W) > 0.5)
            targ_idx = np.argwhere(targ_flat.reshape(self.H, self.W) > 0.5)

            if len(curr_idx) > 0 and len(curr_idx) == len(targ_idx):
                dists = np.abs(curr_idx[:, None, :] -
                               targ_idx[None, :, :]).sum(axis=-1)
                row_ind, col_ind = linear_sum_assignment(dists)
                return -float(dists[row_ind, col_ind].sum())
        except Exception as e:
            if not self._warned:
                print(f"\n[Warning] State extraction failed: {e}. "
                      "Shaping disabled!")
                self._warned = True
        return None   # sentinel: extraction failed this call

    def reset(self, *args, **kwargs):
        state = self.env.reset(*args, **kwargs)
        p = self._compute_potential(state)
        self.last_potential = p if p is not None else 0.0
        return state

    def step(self, action):
        r = self.env.step(action)
        new_potential = self._compute_potential(r.state)

        # Symmetrize: only apply shaping if both endpoints succeeded
        if new_potential is not None and self.last_potential is not None:
            potential_delta = new_potential - self.last_potential
            self.last_potential = new_potential
            bonus = float(self.shape_scale * potential_delta) - self.step_penalty
        else:
            # One of the two extractions failed — skip shaping this step
            self.last_potential = new_potential
            bonus = -self.step_penalty

        try:
            r.reward = r.reward + bonus
        except AttributeError:
            r = r._replace(reward=r.reward + bonus)

        return r


# ─────────────────────────────────────────────────────────────────
#  Per-size configuration
# ─────────────────────────────────────────────────────────────────

SIZE_CONFIGS = {
    2:  dict(n_atoms=2,  max_steps=6,    scramble=(1, 2, 3),
             steps_per_stage=(30_000, 30_000, 30_000),
             hidden=128, lr=5e-4, entropy=0.04,
             threshold=0.98),

    3:  dict(n_atoms=3,  max_steps=12,   scramble=(1, 2, 3),
             steps_per_stage=(80_000, 100_000, 120_000),
             hidden=256, lr=5e-4, entropy=0.04,
             threshold=0.98),

    4:  dict(n_atoms=4,  max_steps=15,   scramble=(1, 2, 3),
             steps_per_stage=(120_000, 150_000, 180_000),
             hidden=384, lr=4e-4, entropy=0.03,
             threshold=0.90),

    5:  dict(n_atoms=5,  max_steps=20,   scramble=(1, 2, 3),
             steps_per_stage=(150_000, 200_000, 300_000),
             hidden=512, lr=3e-4, entropy=0.05,
             threshold=0.85),

    6:  dict(n_atoms=6,  max_steps=24,   scramble=(1, 2, 3),
             steps_per_stage=(200_000, 250_000, 300_000),
             hidden=512, lr=2.5e-4, entropy=0.02,
             threshold=0.75),

    7:  dict(n_atoms=8,  max_steps=28,   scramble=(1, 2, 3),
             steps_per_stage=(250_000, 300_000, 350_000),
             hidden=512, lr=2e-4, entropy=0.02,
             threshold=0.70),

    8:  dict(n_atoms=10, max_steps=32,   scramble=(1, 2, 3),
             steps_per_stage=(300_000, 350_000, 400_000),
             hidden=512, lr=1.5e-4, entropy=0.02,
             threshold=0.60),

    9:  dict(n_atoms=12, max_steps=38,   scramble=(1, 2, 3),
             steps_per_stage=(400_000, 450_000, 500_000),
             hidden=512, lr=1e-4, entropy=0.01,
             threshold=0.50),

    13: dict(n_atoms=20, max_steps=50,   scramble=(1, 2, 3),
             steps_per_stage=(800_000, 800_000, 800_000),
             hidden=768, lr=1e-4, entropy=0.01,
             threshold=0.40),
}


def quick_eval(model, H, W, n_atoms, scramble, max_steps,
               episodes=100, seed=1234):
    raw_env = AtomArrangementEnv(grid_size=H, n_atoms=n_atoms,
                                 max_steps=max_steps,
                                 target_scramble=scramble, seed=seed)
    env = ShapedAtomEnv(raw_env, H, W)
    model.eval()
    wins = 0
    with torch.no_grad():
        for _ in range(episodes):
            state = env.reset()
            done = False
            info = None
            while not done:
                logits, _ = model(state)
                mask = env.action_mask()
                if mask.shape[0] == logits.shape[0]:
                    logits = logits.masked_fill(~mask, -1e9)
                action = int(logits.argmax().item())
                r = env.step(action)
                state = r.state
                done = r.done
                info = r.info
            if info and info.get("success", False):
                wins += 1
    return wins / episodes


def train_one_size(H, W, cfg, seed=42):
    print("\n" + "=" * 68)
    print(f"  Training: {H}x{W}  ({cfg['n_atoms']} atoms)")
    print("=" * 68)

    raw_env = AtomArrangementEnv(
        grid_size=H, n_atoms=cfg["n_atoms"],
        max_steps=cfg["max_steps"],
        target_scramble=cfg["scramble"][0],
        seed=seed,
    )
    env = ShapedAtomEnv(raw_env, H, W, shape_scale=0.15, step_penalty=0.01)
    env.reset()

    model = PolicyValueNet(env.state_dim, env.n_actions,
                           hidden=cfg["hidden"])

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} params, hidden={cfg['hidden']}")
    print(f"  Actions: {env.n_actions}  State: {env.state_dim}")

    ppo_cfg = PPOConfig(
        total_steps=sum(cfg["steps_per_stage"]),
        rollout_len=2048,
        epochs_per_rollout=4,
        mini_batch=256,
        entropy_coef=cfg["entropy"],
        value_coef=0.5,
        lr=cfg["lr"],
        gamma=0.98,
        log_every=20_000,
        seed=seed,
    )
    trainer = PPOTrainer(env, model, ppo_cfg)

    t0 = time.time()
    best = 0.0
    for stage, (scramble, steps) in enumerate(
            zip(cfg["scramble"], cfg["steps_per_stage"]), 1):

        stage_max_steps = min(cfg["max_steps"], scramble * 4 + 1)
        env.target_scramble = scramble
        env.max_steps = stage_max_steps

        # Anneal shape_scale to 0.0 by the final stage
        stage_fraction = (stage - 1) / max(1, len(cfg["scramble"]) - 1)
        current_shape_scale = 0.15 * (1.0 - stage_fraction)
        env.shape_scale = current_shape_scale

        print(f"\n  --- stage {stage}: scramble={scramble}, "
              f"max_steps={stage_max_steps}, "
              f"shape_scale={current_shape_scale:.3f}, "
              f"steps={steps:,} ---")

        stage_t0 = time.time()
        trainer.train(total_steps=steps)
        stage_dt = time.time() - stage_t0

        print(f"  Eval after stage {stage}:")
        for s in cfg["scramble"]:
            acc = quick_eval(model, H, W, cfg["n_atoms"], s,
                             cfg["max_steps"], episodes=50,
                             seed=1000 + s)
            print(f"    scramble={s}: {acc:.1%}")

        hardest = quick_eval(model, H, W, cfg["n_atoms"],
                             cfg["scramble"][-1], cfg["max_steps"],
                             episodes=100, seed=9999)
        best = max(best, hardest)
        print(f"    [hardest eval, 100 episodes]: {hardest:.1%}")
        print(f"    [stage wall-clock]: {stage_dt:.1f}s")

        # Warm-start: save checkpoint after every stage
        ckpt_path = os.path.join(CHECKPOINT_DIR,
                                  f"rl_{H}x{W}_stage{stage}.pt")
        torch.save({"state_dict": model.state_dict(),
                    "size": H, "stage": stage, "scramble": scramble},
                   ckpt_path)
        print(f"    [checkpoint saved]: {ckpt_path}")

    dt = time.time() - t0
    print(f"\n  Total training time: {dt:.1f}s  "
          f"Best hardest-eval: {best:.1%}")
    return model, best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=str, default="2,3,4,5,6,7,8,9,13",
                        help="Comma-separated grid sizes")
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]

    print("=" * 68)
    print(f"  RL compiler scaling: {sizes}")
    print(f"  Hardware: CPU")
    print(f"  Checkpoints: {CHECKPOINT_DIR}")
    print("=" * 68)

    results = {}
    for size in sizes:
        if size not in SIZE_CONFIGS:
            print(f"\n  Skipping {size}x{size}: no config")
            continue
        cfg = SIZE_CONFIGS[size]
        model, best = train_one_size(size, size, cfg)
        results[size] = best

        torch.save({"state_dict": model.state_dict(), "size": size},
                   f"/tmp/rl_compiler_{size}x{size}.pt")

        if best < cfg["threshold"]:
            print(f"\n  !! {size}x{size} below threshold "
                  f"({best:.1%} < {cfg['threshold']:.0%})")
            print(f"  Stopping before larger sizes.")
            break
        else:
            print(f"\n  ++ {size}x{size} passed threshold "
                  f"({best:.1%} >= {cfg['threshold']:.0%})")
            print(f"  Proceeding to next size.")

    print("\n" + "=" * 68)
    print("  SUMMARY")
    print("=" * 68)
    for size in sizes:
        if size in results:
            ok = "PASS" if results[size] >= SIZE_CONFIGS[size]["threshold"] else "FAIL"
            print(f"  {size:3d}x{size:<3d}  {results[size]:5.1%}  [{ok}]")
        else:
            print(f"  {size:3d}x{size:<3d}  (not attempted)")


if __name__ == "__main__":
    main()
