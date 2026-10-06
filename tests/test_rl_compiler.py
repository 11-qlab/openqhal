"""
Train an RL compiler with curriculum learning.
"""
import sys, time
sys.path.insert(0, "python")

import numpy as np
import torch

from qhal.rl import (
    AtomArrangementEnv, PolicyValueNet, RolloutBuffer,
    evaluate, save_model, TrainConfig,
)
from torch.distributions import Categorical
import torch.nn.functional as F


def compute_returns(rewards, gamma, dones):
    returns = []
    G = 0.0
    for r, d in zip(reversed(rewards), reversed(dones)):
        G = r + gamma * G * (1.0 - d)
        returns.insert(0, G)
    return returns


def train_curriculum(episodes=4000, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Phase 1: 2x2 grid, 2 atoms, 1-move targets — trivially learnable
    env = AtomArrangementEnv(grid_size=2, n_atoms=2,
                             max_steps=6, target_scramble=1, seed=seed)
    model = PolicyValueNet(env.state_dim, env.n_actions, hidden=64)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    gamma = 0.95
    entropy_coef = 0.02
    value_coef = 0.5

    # Full curriculum: 1-move targets for 1/3 of episodes,
    # 2-move for the next 1/3, 3-move for the last 1/3
    curriculum = [
        (1, episodes // 3),
        (2, 2 * episodes // 3),
        (3, episodes),
    ]

    history = []
    for ep in range(1, episodes + 1):
        # Update difficulty
        for scramble, until_ep in curriculum:
            if ep <= until_ep:
                env.target_scramble = scramble
                break

        buf = RolloutBuffer()
        state = env.reset()
        done = False
        last_info = None

        while not done:
            logits, value = model(state)
            if not torch.isfinite(logits).all():
                logits = torch.zeros_like(logits)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            result = env.step(action.item())
            buf.add(state, action.item(), result.reward,
                    log_prob, value, float(result.done))
            state = result.state
            done = result.done
            last_info = result.info

        # Update
        t = buf.tensors()
        returns = torch.tensor(
            compute_returns(t["rewards"].tolist(), gamma, t["dones"].tolist()),
            dtype=torch.float32)
        values = t["values"]

        # No normalization — the raw return already has the right sign.
        # Normalization was inverting the advantage on failed episodes.
        advantage = returns - values.detach()
        advantage = torch.nan_to_num(advantage, nan=0.0)

        policy_loss = -(t["log_probs"] * advantage).mean()
        value_loss = F.mse_loss(values, returns)
        logits_all, _ = model(t["states"])
        entropy = Categorical(logits=logits_all).entropy().mean()
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        history.append({"episode": ep, "success": last_info["success"],
                        "steps": last_info["steps"]})

        if ep % 250 == 0:
            recent = history[-250:]
            succ = np.mean([h["success"] for h in recent])
            steps = np.mean([h["steps"] for h in recent])
            print(f"ep {ep:5d}  scramble={env.target_scramble}  "
                  f"success {succ:5.1%}  steps {steps:4.1f}")

        if ep % 1000 == 0:
            # Eval env must match the training env's dimensions
            eval_env = AtomArrangementEnv(grid_size=env.H, n_atoms=env.n_atoms,
                                           max_steps=env.max_steps,
                                           target_scramble=env.target_scramble,
                                           seed=ep + 999)
            eval_succ = evaluate(model, eval_env, n_episodes=100)
            print(f"  [eval@{ep}]  greedy success = {eval_succ:.1%}")

    return model, env, history


def main():
    print("=" * 68)
    print("  RL Compiler with Curriculum Learning")
    print("=" * 68)

    env = AtomArrangementEnv(grid_size=2, n_atoms=2, target_scramble=1, seed=0)
    env.reset()
    print(f"\n[1] Env: {env.H}x{env.W}, {env.n_atoms} atoms, "
          f"{env.n_actions} actions, state_dim={env.state_dim}")

    print(f"\n[2] Training with curriculum (1->2->3 move targets)")
    t0 = time.time()
    model, trained_env, hist = train_curriculum(episodes=4000)
    print(f"\n    training time: {time.time() - t0:.1f}s")

    save_model(model, "/tmp/rl_compiler.pt")

    # Final eval at each scramble level
    print(f"\n[3] Final eval (500 episodes each)")
    for scramble in [1, 2, 3]:
        test_env = AtomArrangementEnv(grid_size=2, n_atoms=2,
                                       max_steps=6,
                                       target_scramble=scramble,
                                       seed=7777)
        succ = evaluate(model, test_env, n_episodes=500)
        print(f"    scramble={scramble}  greedy success = {succ:.1%}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
