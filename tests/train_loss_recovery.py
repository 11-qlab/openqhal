"""
Curriculum training for loss recovery.

Stage ladder:
  0.00  -> learn basic rearrangement (matches classical planner)
  0.01  -> light loss
  0.02  -> realistic AOD loss
  0.05  -> pessimistic
  0.10  -> worst case
"""
import sys, time, os, glob
sys.path.insert(0, "python")

import numpy as np
import torch
from stable_baselines3 import PPO
from qhal.rl.loss_env import LossyAtomEnv
from qhal.rl.features import AtomGridCNN

GRID = 5
N_ATOMS = 5
FEATURES = 256
NET_ARCH = dict(pi=[256, 256], vf=[256, 256])

STAGES = [
    dict(p_loss=0.00, max_steps=15, target=0.90),
    dict(p_loss=0.01, max_steps=15, target=0.90),
    dict(p_loss=0.02, max_steps=20, target=0.85),
    dict(p_loss=0.05, max_steps=25, target=0.80),
    dict(p_loss=0.10, max_steps=30, target=0.70),
]
EVAL_INTERVAL = 20_480
MAX_PER_STAGE = 400_000
CKPT_DIR = os.path.expanduser("~/projects/openqhal/rl_loss_recovery")
os.makedirs(CKPT_DIR, exist_ok=True)

env = LossyAtomEnv(grid_size=GRID, n_atoms=N_ATOMS,
                    max_steps=STAGES[0]["max_steps"],
                    scramble=1, p_loss=STAGES[0]["p_loss"], seed=42)

policy_kwargs = dict(
    features_extractor_class=AtomGridCNN,
    features_extractor_kwargs=dict(features_dim=FEATURES, grid_size=GRID),
    net_arch=NET_ARCH)

model = PPO("CnnPolicy", env, verbose=0,
            learning_rate=3e-4, n_steps=2048, batch_size=256, n_epochs=4,
            gamma=0.98, gae_lambda=0.95, clip_range=0.15,
            ent_coef=0.02, vf_coef=0.5, max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            device="cuda" if torch.cuda.is_available() else "cpu",
            seed=42)

print(f"Params: {sum(p.numel() for p in model.policy.parameters()):,}\n")


def evaluate(model, env, episodes=50):
    wins = 0
    for i in range(episodes):
        obs, _ = env.reset(seed=20_000 + i)
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(int(a))
            done = term or trunc
            if info.get("is_success", False):
                wins += 1
                break
    return wins / episodes


t_start = time.time()
for stage_idx, stage in enumerate(STAGES):
    print(f"\n{'='*60}")
    print(f"  Stage {stage_idx+1}:  p_loss={stage['p_loss']}  "
          f"target={stage['target']:.0%}")
    print(f"{'='*60}")

    env.set_curriculum(p_loss=stage["p_loss"],
                       scramble=1,
                       max_steps=stage["max_steps"])

    success = 0.0
    stage_steps = 0
    eval_env = LossyAtomEnv(grid_size=GRID, n_atoms=N_ATOMS,
                             max_steps=stage["max_steps"],
                             scramble=1, p_loss=stage["p_loss"],
                             seed=9999)

    while success < stage["target"] and stage_steps < MAX_PER_STAGE:
        model.learn(total_timesteps=EVAL_INTERVAL,
                    reset_num_timesteps=False)
        stage_steps += EVAL_INTERVAL
        success = evaluate(model, eval_env, episodes=50)
        elapsed = time.time() - t_start
        print(f"  step {stage_steps:7d}  success {success*100:5.1f}%  "
              f"[{elapsed/60:.1f} min]")

    model.save(f"{CKPT_DIR}/stage{stage_idx+1}_p{stage['p_loss']}.zip")

    if success < stage["target"]:
        print(f"\n  Stage capped at {success:.1%}. Stopping.")
        break

model.save(f"{CKPT_DIR}/final.zip")
print(f"\nDone in {(time.time()-t_start)/60:.1f} min")
print(f"Model: {CKPT_DIR}/final.zip")
