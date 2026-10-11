"""
Local 5x5 training — same config as the Colab run.

No GPU on this machine. Expect 5-10x slower than Colab T4.
Checkpoints save per stage; safe to Ctrl+C and resume.
"""
import os, sys, time, json
sys.path.insert(0, "python")
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from qhal.rl.gym_wrapper import ShapedAtomGym
from qhal.rl.features import AtomGridCNN
from stable_baselines3 import PPO


GRID = 5
N_ATOMS = 5
FEATURES_DIM = 256
NET_ARCH = dict(pi=[128], vf=[128])

STAGES = [
    dict(scramble=1, max_steps=5,  shape_scale=0.500, target=0.90),
    dict(scramble=2, max_steps=9,  shape_scale=0.200, target=0.90),
    dict(scramble=3, max_steps=13, shape_scale=0.050, target=0.95),
]

EVAL_INTERVAL = 20_480
MAX_STEPS_PER_STAGE = 500_000

CKPT_DIR = os.path.expanduser("~/projects/openqhal/rl_checkpoints_local")
os.makedirs(CKPT_DIR, exist_ok=True)

env = ShapedAtomGym(grid_size=GRID, n_atoms=N_ATOMS,
                    max_steps=STAGES[0]["max_steps"],
                    scramble=STAGES[0]["scramble"],
                    shape_scale=STAGES[0]["shape_scale"],
                    seed=42)

eval_env = ShapedAtomGym(grid_size=GRID, n_atoms=N_ATOMS,
                         max_steps=STAGES[-1]["max_steps"],
                         scramble=STAGES[-1]["scramble"],
                         shape_scale=0.0, seed=9999)

policy_kwargs = dict(
    features_extractor_class=AtomGridCNN,
    features_extractor_kwargs=dict(features_dim=FEATURES_DIM, grid_size=GRID),
    net_arch=NET_ARCH)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on: {device}")
print(f"Checkpoints: {CKPT_DIR}\n")

model = PPO(
    "MlpPolicy", env, verbose=0,
    learning_rate=3e-4, n_steps=2048, batch_size=256, n_epochs=4,
    gamma=0.98, gae_lambda=0.95, clip_range=0.2,
    ent_coef=0.05, vf_coef=0.5, max_grad_norm=1.0,
    policy_kwargs=policy_kwargs,
    device=device, seed=42)

n_params = sum(p.numel() for p in model.policy.parameters())
print(f"Model params: {n_params:,}\n")


def evaluate(model, env, episodes=100, seed_base=10_000):
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


history = []
t_start = time.time()

for stage_idx, stage in enumerate(STAGES):
    print(f"\n{'='*66}")
    print(f"Stage {stage_idx+1}: scramble={stage['scramble']}, "
          f"max_steps={stage['max_steps']}, "
          f"shape_scale={stage['shape_scale']}, target={stage['target']:.0%}")
    print(f"{'='*66}")

    env.set_curriculum(scramble=stage["scramble"],
                       max_steps=stage["max_steps"],
                       shape_scale=stage["shape_scale"])

    model.ent_coef = max(0.01, 0.05 - stage_idx * 0.015)
    print(f"ent_coef = {model.ent_coef:.3f}")

    success = 0.0
    stage_steps = 0
    stage_start = time.time()

    while success < stage["target"]:
        model.learn(total_timesteps=EVAL_INTERVAL,
                    reset_num_timesteps=False)
        stage_steps += EVAL_INTERVAL

        eval_env.set_curriculum(scramble=stage["scramble"],
                                max_steps=stage["max_steps"],
                                shape_scale=0.0)
        success = evaluate(model, eval_env, episodes=100)
        elapsed = time.time() - stage_start
        print(f"  step {stage_steps:7d}  "
              f"success {success*100:5.1f}%  "
              f"(target {stage['target']*100:.0f}%)  [{elapsed:.0f}s]")

        history.append({"stage": stage_idx + 1, "steps": stage_steps,
                        "success": success, "wall_clock_s": elapsed})

        if stage_steps > MAX_STEPS_PER_STAGE:
            print(f"  !! Plateau at stage {stage_idx+1}. Breaking.")
            break

    print(f"\n  Eval after stage {stage_idx+1}:")
    for s in [1, 2, 3]:
        eval_env.set_curriculum(scramble=s,
                                max_steps=stage["max_steps"],
                                shape_scale=0.0)
        acc = evaluate(model, eval_env, episodes=50)
        print(f"    scramble={s}: {acc:.1%}")

    ckpt = os.path.join(CKPT_DIR, f"sb3_{GRID}x{GRID}_stage{stage_idx+1}")
    model.save(ckpt)
    print(f"  [saved] {ckpt}.zip")

    with open(os.path.join(CKPT_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    if success < stage["target"] and stage_idx < len(STAGES) - 1:
        print(f"\n  Stage {stage_idx+1} below target. Stopping.")
        break

total_time = time.time() - t_start
print(f"\n{'='*66}")
print(f"Training complete. Total: {total_time/60:.1f} min")
print(f"{'='*66}")

eval_env.set_curriculum(scramble=3, max_steps=13, shape_scale=0.0)
final = evaluate(model, eval_env, episodes=200)
print(f"\nFinal eval (scramble=3, 200 episodes): {final:.1%}")
