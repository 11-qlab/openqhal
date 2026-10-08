"""
5x5 CPU training with determinism fixes for cross-platform consistency.

Key changes vs train_5x5_kali.py:
  1. Full RNG seeding (torch, numpy, random, pythonhash)
  2. Deterministic algorithm mode (torch.use_deterministic_algorithms)
  3. MKLDNN disabled by default (major CPU/GPU divergence source)
  4. Float32 enforced (or float64 via --dtype flag)
  5. Single-thread option for full determinism (--threads 1)
  6. Structured logging to JSON for cross-platform comparison

References:
  - PyTorch reproducibility guide: https://pytorch.org/docs/stable/notes/randomness.html
  - OneDNN determinism: https://oneapi-src.github.io/oneDNN/dev_guide_deterministic.html
  - SB3 reproducibility: https://stable-baselines3.readthedocs.io/en/master/guide/algos.html

Usage:
    python tests/train_5x5_cpu_fixed.py
    python tests/train_5x5_cpu_fixed.py --threads 1
    python tests/train_5x5_cpu_fixed.py --dtype float64
    python tests/train_5x5_cpu_fixed.py --mkldnn on
    python tests/train_5x5_cpu_fixed.py --tag cpu_fixed_v1
"""
import os
import sys
import time
import json
import random
import argparse
import platform
from pathlib import Path

# ─────────────────────────────────────────────────────────────────
#  ARGUMENT PARSING (before torch import)
# ─────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--threads", type=int, default=8,
                    help="Torch CPU thread count. Set to 1 for full determinism.")
parser.add_argument("--dtype", type=str, default="float32",
                    choices=["float32", "float64"])
parser.add_argument("--mkldnn", type=str, default="off",
                    choices=["on", "off"],
                    help="Enable/disable MKLDNN. 'off' improves cross-platform parity.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--steps", type=int, default=500_000,
                    help="Max steps per stage (cap, not budget)")
parser.add_argument("--tag", type=str, default="cpu_fixed",
                    help="Checkpoint directory suffix")
parser.add_argument("--eval-interval", type=int, default=20_480)
parser.add_argument("--eval-episodes", type=int, default=100)
args = parser.parse_args()

# ─────────────────────────────────────────────────────────────────
#  DETERMINISM SETUP (before any torch imports)
# ─────────────────────────────────────────────────────────────────

# 1. Single-thread determinism option
os.environ["OMP_NUM_THREADS"] = str(args.threads)
os.environ["MKL_NUM_THREADS"] = str(args.threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(args.threads)
os.environ["PYTHONHASHSEED"] = str(args.seed)

# 2. Disable oneDNN at the env level (before torch loads its symbols)
if args.mkldnn == "off":
    os.environ["ONEDNN_MAX_CPU_ISA"] = "AVX2"     # cap ISA to a safe baseline
    os.environ["DNNL_MAX_CPU_ISA"] = "AVX2"       # oneDNN legacy name
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"     # defensive

# ─────────────────────────────────────────────────────────────────
#  Now import torch and the rest
# ─────────────────────────────────────────────────────────────────

sys.path.insert(0, "python")
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from qhal.rl.gym_wrapper import ShapedAtomGym
from qhal.rl.features import AtomGridCNN


# 3. Seed everything
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# 4. Force deterministic algorithms
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 5. MKLDNN flag on the torch object (belt and braces)
if args.mkldnn == "off":
    try:
        torch.backends.mkldnn.enabled = False
    except AttributeError:
        pass

# 6. Set torch thread count after import
torch.set_num_threads(args.threads)


# ─────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────

GRID = 5
N_ATOMS = 5
FEATURES_DIM = 256
NET_ARCH = dict(pi=[128], vf=[128])

STAGES = [
    dict(scramble=1, max_steps=5,  shape_scale=0.500, target=0.90),
    dict(scramble=2, max_steps=9,  shape_scale=0.200, target=0.90),
    dict(scramble=3, max_steps=13, shape_scale=0.050, target=0.95),
]

CKPT_DIR = Path.home() / "projects" / "openqhal" / f"rl_checkpoints_{args.tag}"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

dtype_torch = torch.float64 if args.dtype == "float64" else torch.float32


# ─────────────────────────────────────────────────────────────────
#  Environment info
# ─────────────────────────────────────────────────────────────────

def print_config():
    print("=" * 68)
    print(f"  5x5 CPU training — {args.tag}")
    print("=" * 68)
    print(f"  Platform:     {platform.system()} {platform.machine()}")
    print(f"  Python:       {sys.version.split()[0]}")
    print(f"  Torch:        {torch.__version__}")
    print(f"  Threads:      {torch.get_num_threads()}")
    print(f"  Dtype:        {args.dtype}")
    print(f"  MKLDNN:       {args.mkldnn}")
    print(f"  Seed:         {args.seed}")
    print(f"  Deterministic: {torch.are_deterministic_algorithms_enabled()}")
    print(f"  Checkpoints:  {CKPT_DIR}")
    print(f"  Steps cap:    {args.steps:,} per stage")
    print()


print_config()


# ─────────────────────────────────────────────────────────────────
#  Build environments
# ─────────────────────────────────────────────────────────────────

env = ShapedAtomGym(grid_size=GRID, n_atoms=N_ATOMS,
                    max_steps=STAGES[0]["max_steps"],
                    scramble=STAGES[0]["scramble"],
                    shape_scale=STAGES[0]["shape_scale"],
                    seed=args.seed)

eval_env = ShapedAtomGym(grid_size=GRID, n_atoms=N_ATOMS,
                         max_steps=STAGES[-1]["max_steps"],
                         scramble=STAGES[-1]["scramble"],
                         shape_scale=0.0, seed=args.seed + 9999)


# ─────────────────────────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────────────────────────

policy_kwargs = dict(
    features_extractor_class=AtomGridCNN,
    features_extractor_kwargs=dict(
        features_dim=FEATURES_DIM, grid_size=GRID),
    net_arch=NET_ARCH)

device = "cpu"

model = PPO(
    "MlpPolicy", env, verbose=0,
    learning_rate=3e-4, n_steps=2048, batch_size=256, n_epochs=4,
    gamma=0.98, gae_lambda=0.95, clip_range=0.2,
    ent_coef=0.05, vf_coef=0.5, max_grad_norm=1.0,
    policy_kwargs=policy_kwargs,
    device=device, seed=args.seed)

# Cast to chosen precision if requested
if args.dtype == "float64":
    model.policy = model.policy.double()

n_params = sum(p.numel() for p in model.policy.parameters())
print(f"Model params: {n_params:,}\n")


# ─────────────────────────────────────────────────────────────────
#  Evaluation
# ─────────────────────────────────────────────────────────────────

def evaluate(model, env, episodes=100, seed_base=10_000):
    """Greedy eval, deterministic. Uses fixed eval seed sequence."""
    wins = 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed_base + ep)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
            if info.get("is_success", False):
                wins += 1
                break
    return wins / episodes


# ─────────────────────────────────────────────────────────────────
#  Training loop
# ─────────────────────────────────────────────────────────────────

history = []
t_start = time.time()

for stage_idx, stage in enumerate(STAGES):
    print(f"\n{'='*66}")
    print(f"Stage {stage_idx+1}: scramble={stage['scramble']}, "
          f"max_steps={stage['max_steps']}, "
          f"shape_scale={stage['shape_scale']:.3f}, "
          f"target={stage['target']:.0%}")
    print(f"{'='*66}")

    env.set_curriculum(scramble=stage["scramble"],
                       max_steps=stage["max_steps"],
                       shape_scale=stage["shape_scale"])

    model.ent_coef = max(0.01, 0.05 - stage_idx * 0.015)
    print(f"ent_coef = {model.ent_coef:.3f}")

    success = 0.0
    stage_steps = 0
    stage_start = time.time()

    while success < stage["target"] and stage_steps < args.steps:
        model.learn(total_timesteps=args.eval_interval,
                    reset_num_timesteps=False)
        stage_steps += args.eval_interval

        eval_env.set_curriculum(scramble=stage["scramble"],
                                max_steps=stage["max_steps"],
                                shape_scale=0.0)
        success = evaluate(model, eval_env, episodes=args.eval_episodes)
        elapsed = time.time() - stage_start
        print(f"  step {stage_steps:7d}  "
              f"success {success*100:5.1f}%  "
              f"(target {stage['target']*100:.0f}%)  "
              f"[{elapsed:.0f}s]")

        history.append({
            "stage": stage_idx + 1,
            "steps": stage_steps,
            "success": success,
            "wall_clock_s": elapsed,
        })

        # Save intermediate checkpoints
        if stage_steps % 100_000 < args.eval_interval:
            ckpt = CKPT_DIR / f"sb3_5x5_stage{stage_idx+1}_step{stage_steps}"
            model.save(str(ckpt))
            print(f"  [checkpoint] {ckpt.name}")

    # Eval across scramble levels
    print(f"\n  Eval after stage {stage_idx+1}:")
    for s in [1, 2, 3]:
        eval_env.set_curriculum(scramble=s,
                                max_steps=stage["max_steps"],
                                shape_scale=0.0)
        acc = evaluate(model, eval_env, episodes=50)
        print(f"    scramble={s}: {acc:.1%}")

    # Save stage checkpoint
    ckpt = CKPT_DIR / f"sb3_5x5_stage{stage_idx+1}_final"
    model.save(str(ckpt))
    print(f"  [saved] {ckpt.name}.zip")

    # Save history
    with open(CKPT_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    if success < stage["target"] and stage_idx < len(STAGES) - 1:
        print(f"\n  Stage {stage_idx+1} below target. Stopping.")
        break


# ─────────────────────────────────────────────────────────────────
#  Final
# ─────────────────────────────────────────────────────────────────

total_time = time.time() - t_start
print(f"\n{'='*66}")
print(f"Training complete. Total: {total_time/60:.1f} min")
print(f"{'='*66}")

eval_env.set_curriculum(scramble=3, max_steps=13, shape_scale=0.0)
final = evaluate(model, eval_env, episodes=200)
print(f"\nFinal eval (scramble=3, 200 episodes): {final:.1%}")

model.save(str(CKPT_DIR / "sb3_5x5_final"))
print(f"Saved {CKPT_DIR}/sb3_5x5_final.zip")

# Save run metadata
meta = {
    "tag": args.tag,
    "platform": f"{platform.system()} {platform.machine()}",
    "torch_version": torch.__version__,
    "threads": args.threads,
    "dtype": args.dtype,
    "mkldnn": args.mkldnn,
    "seed": args.seed,
    "steps_cap_per_stage": args.steps,
    "eval_interval": args.eval_interval,
    "eval_episodes": args.eval_episodes,
    "final_success": final,
    "total_wall_clock_s": total_time,
    "history": history,
}
with open(CKPT_DIR / "run_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print(f"Saved {CKPT_DIR}/run_meta.json")
