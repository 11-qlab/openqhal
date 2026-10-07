"""
RL compiler scaling with Stable-Baselines3.

Features:
  - SB3 PPO with a custom CNN feature extractor
  - Competency-gated curriculum: advance only when success_rate >= target
  - Retains a small baseline shape_scale (0.010) at the final stage
  - Dynamic entropy coefficient per stage

Usage:
    python tests/train_sb3_scaling.py --size 5
    python tests/train_sb3_scaling.py --sizes 3,5
"""
import sys, time, argparse, os
sys.path.insert(0, "python")

import numpy as np
import torch

from qhal.rl.gym_wrapper import ShapedAtomGym
from qhal.rl.features import AtomGridCNN

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:
    print("ERROR: stable-baselines3 not installed.")
    print("Run: pip install stable-baselines3 gymnasium")
    sys.exit(1)


CHECKPOINT_DIR = "/tmp/rl_checkpoints_sb3"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────
#  Per-size configuration
# ─────────────────────────────────────────────────────────────────

SIZE_CONFIGS = {
    3:  dict(n_atoms=3,  grid=3,
             stages=[dict(scramble=1, max_steps=5,  shape_scale=0.500, target=0.90),
                     dict(scramble=2, max_steps=9,  shape_scale=0.200, target=0.90),
                     dict(scramble=3, max_steps=13, shape_scale=0.050, target=0.95)],
             features_dim=128, net_arch=dict(pi=[128], vf=[128]),
             threshold=0.95),

    5:  dict(n_atoms=5,  grid=5,
             stages=[dict(scramble=1, max_steps=5,  shape_scale=0.500, target=0.90),
                     dict(scramble=2, max_steps=9,  shape_scale=0.200, target=0.90),
                     dict(scramble=3, max_steps=13, shape_scale=0.050, target=0.95)],
             features_dim=256, net_arch=dict(pi=[128], vf=[128]),
             threshold=0.85),

    7:  dict(n_atoms=8,  grid=7,
             stages=[dict(scramble=1, max_steps=8,  shape_scale=0.150, target=0.85),
                     dict(scramble=2, max_steps=14, shape_scale=0.075, target=0.85),
                     dict(scramble=3, max_steps=20, shape_scale=0.010, target=0.90)],
             features_dim=256, net_arch=dict(pi=[256], vf=[256]),
             threshold=0.70),

    9:  dict(n_atoms=12, grid=9,
             stages=[dict(scramble=1, max_steps=10, shape_scale=0.150, target=0.80),
                     dict(scramble=2, max_steps=18, shape_scale=0.075, target=0.80),
                     dict(scramble=3, max_steps=26, shape_scale=0.010, target=0.85)],
             features_dim=256, net_arch=dict(pi=[256], vf=[256]),
             threshold=0.50),
}


# ─────────────────────────────────────────────────────────────────
#  Evaluation
# ─────────────────────────────────────────────────────────────────

def evaluate_policy_success(model, eval_env, episodes=100):
    wins = 0
    for ep in range(episodes):
        obs, _ = eval_env.reset(seed=10_000 + ep)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            if info.get("is_success", False):
                wins += 1
                break
    return wins / episodes


# ─────────────────────────────────────────────────────────────────
#  Curriculum training
# ─────────────────────────────────────────────────────────────────

def train_curriculum(grid, cfg, seed=42):
    print(f"\n{'='*68}")
    print(f"  SB3 training: {grid}x{grid} ({cfg['n_atoms']} atoms)")
    print(f"{'='*68}")

    env = ShapedAtomGym(
        grid_size=grid, n_atoms=cfg["n_atoms"],
        max_steps=cfg["stages"][0]["max_steps"],
        scramble=cfg["stages"][0]["scramble"],
        shape_scale=cfg["stages"][0]["shape_scale"],
        seed=seed)

    eval_env = ShapedAtomGym(
        grid_size=grid, n_atoms=cfg["n_atoms"],
        max_steps=cfg["stages"][-1]["max_steps"],
        scramble=cfg["stages"][-1]["scramble"],
        shape_scale=0.0, seed=seed + 9999)

    policy_kwargs = dict(
        features_extractor_class=AtomGridCNN,
        features_extractor_kwargs=dict(
            features_dim=cfg["features_dim"], grid_size=grid),
        net_arch=cfg["net_arch"])

    model = PPO(
        "MlpPolicy", env, verbose=0,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=4,
        gamma=0.98,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        vf_coef=0.5,
        max_grad_norm=1.0,
        policy_kwargs=policy_kwargs,
        seed=seed)

    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"  Model: {n_params:,} params, features_dim={cfg['features_dim']}")

    eval_interval = 20_480
    t0 = time.time()
    best_hardest = 0.0

    for stage_idx, stage in enumerate(cfg["stages"]):
        print(f"\n  --- Stage {stage_idx+1}: scramble={stage['scramble']}, "
              f"max_steps={stage['max_steps']}, "
              f"shape_scale={stage['shape_scale']:.3f}, "
              f"target={stage['target']:.0%} ---")

        env.set_curriculum(
            scramble=stage["scramble"],
            max_steps=stage["max_steps"],
            shape_scale=stage["shape_scale"])

        # Decay entropy across stages
        model.ent_coef = max(0.01, 0.05 - stage_idx * 0.015)
        print(f"  ent_coef = {model.ent_coef:.3f}")

        success = 0.0
        total = 0
        while success < stage["target"]:
            model.learn(total_timesteps=eval_interval,
                        reset_num_timesteps=False)
            total += eval_interval

            # Eval on the stage's scramble
            eval_env.set_curriculum(scramble=stage["scramble"],
                                     max_steps=stage["max_steps"],
                                     shape_scale=0.0)
            success = evaluate_policy_success(model, eval_env, episodes=100)
            print(f"  step {total:7d}  success {success*100:5.1f}%  "
                  f"(target {stage['target']*100:.0f}%)")

            if total > 500_000:
                print(f"  !! Plateau. Breaking stage {stage_idx+1}.")
                break

        # Final eval at all scrambles
        print(f"  Eval after stage {stage_idx+1}:")
        for s in [1, 2, 3]:
            eval_env.set_curriculum(scramble=s,
                                     max_steps=stage["max_steps"],
                                     shape_scale=0.0)
            acc = evaluate_policy_success(model, eval_env, episodes=50)
            print(f"    scramble={s}: {acc:.1%}")

        eval_env.set_curriculum(scramble=cfg["stages"][-1]["scramble"],
                                 max_steps=cfg["stages"][-1]["max_steps"],
                                 shape_scale=0.0)
        hardest = evaluate_policy_success(model, eval_env, episodes=100)
        best_hardest = max(best_hardest, hardest)
        print(f"    [hardest, 100 eps]: {hardest:.1%}")

        ckpt = os.path.join(CHECKPOINT_DIR, f"sb3_{grid}x{grid}_stage{stage_idx+1}")
        model.save(ckpt)
        print(f"    [saved] {ckpt}")

        if success < stage["target"] and stage_idx < len(cfg["stages"]) - 1:
            print(f"  Stage {stage_idx+1} plateau — not advancing.")
            break

    dt = time.time() - t0
    print(f"\n  Total time: {dt:.1f}s  Best: {best_hardest:.1%}")
    return model, best_hardest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=None)
    p.add_argument("--sizes", type=str, default=None)
    args = p.parse_args()

    if args.sizes:
        sizes = [int(s) for s in args.sizes.split(",")]
    elif args.size:
        sizes = [args.size]
    else:
        sizes = [3, 5]

    print(f"\n{'#'*68}")
    print(f"  SB3 RL compiler scaling: {sizes}")
    print(f"{'#'*68}")

    results = {}
    for size in sizes:
        if size not in SIZE_CONFIGS:
            print(f"\n  Skipping {size}x{size}: no config")
            continue
        cfg = SIZE_CONFIGS[size]
        model, best = train_curriculum(size, cfg)
        results[size] = best

        if best < cfg["threshold"]:
            print(f"\n  !! {size}x{size} below threshold "
                  f"({best:.1%} < {cfg['threshold']:.0%}). Stopping.")
            break
        print(f"\n  ++ {size}x{size} passed ({best:.1%})")

    print(f"\n{'='*68}\n  SUMMARY\n{'='*68}")
    for s in sizes:
        if s in results:
            ok = "PASS" if results[s] >= SIZE_CONFIGS[s]["threshold"] else "FAIL"
            print(f"  {s:3d}x{s:<3d}  {results[s]:5.1%}  [{ok}]")
        else:
            print(f"  {s:3d}x{s:<3d}  (not attempted)")


if __name__ == "__main__":
    main()
