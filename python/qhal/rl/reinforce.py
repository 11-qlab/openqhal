"""
REINFORCE with value baseline for the atom rearrangement compiler.

The policy network is trained to maximize expected episode return.
The value head serves as a learned baseline to reduce variance.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .env import AtomArrangementEnv
from .policy import PolicyValueNet, RolloutBuffer


@dataclass
class TrainConfig:
    episodes: int = 3000
    gamma: float = 0.98
    lr: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    log_every: int = 250
    seed: int = 42
    eval_every: int = 500
    eval_episodes: int = 50


def compute_returns(rewards, gamma, dones):
    """Monte Carlo returns with terminal masking."""
    returns = []
    G = 0.0
    for r, d in zip(reversed(rewards), reversed(dones)):
        G = r + gamma * G * (1.0 - d)
        returns.insert(0, G)
    return returns


def train(cfg: TrainConfig = None):
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    env = AtomArrangementEnv(grid_size=4, n_atoms=6,
                             max_steps=20, seed=cfg.seed)

    model = PolicyValueNet(env.state_dim, env.n_actions, hidden=128)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # Metrics
    history = []
    for ep in range(1, cfg.episodes + 1):
        buf = RolloutBuffer()
        state = env.reset()
        done = False
        while not done:
            logits, value = model(state)
            # Guard: if weights blew up, reset this step to uniform
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

        # Compute returns
        t = buf.tensors()
        returns = compute_returns(t["rewards"].tolist(),
                                  cfg.gamma, t["dones"].tolist())
        returns = torch.tensor(returns, dtype=torch.float32)
        values = t["values"]

        # Advantage = returns - values (detached baseline)
        advantage = returns - values.detach()
        # Normalize only when there is more than one sample in the batch
        if advantage.numel() > 1:
            std = advantage.std()
            if torch.isfinite(std) and std > 1e-8:
                advantage = (advantage - advantage.mean()) / std
            else:
                advantage = advantage - advantage.mean()
        # Guard against NaN from any source
        advantage = torch.nan_to_num(advantage, nan=0.0, posinf=0.0, neginf=0.0)

        # Policy loss: -E[log_pi(a|s) * A]
        policy_loss = -(t["log_probs"] * advantage).mean()

        # Value loss: MSE to returns
        value_loss = F.mse_loss(values, returns)

        # Entropy bonus
        logits_all, _ = model(t["states"])
        dist_all = Categorical(logits=logits_all)
        entropy = dist_all.entropy().mean()

        loss = policy_loss + cfg.value_coef * value_loss - \
               cfg.entropy_coef * entropy

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # Logging
        info = result.info
        success = info["success"]
        history.append({"episode": ep,
                        "reward": float(t["rewards"].sum().item()),
                        "steps": info["steps"],
                        "success": success})

        if ep % cfg.log_every == 0:
            recent = history[-cfg.log_every:]
            succ_rate = np.mean([h["success"] for h in recent])
            avg_steps = np.mean([h["steps"] for h in recent])
            avg_ret = np.mean([h["reward"] for h in recent])
            print(f"ep {ep:5d}  "
                  f"success {succ_rate:5.2%}  "
                  f"steps {avg_steps:5.1f}  "
                  f"return {avg_ret:7.2f}")

        # Periodic eval
        if ep % cfg.eval_every == 0:
            eval_success = evaluate(model, env, cfg.eval_episodes)
            print(f"  [eval@{ep}]  greedy success = {eval_success:.1%}")

    return model, history


@torch.no_grad()
def evaluate(model, env, n_episodes=50) -> float:
    """Greedy rollout; return success rate."""
    model.eval()
    successes = 0
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        last_info = None
        while not done:
            logits, _ = model(state)
            action = logits.argmax().item()
            result = env.step(action)
            state = result.state
            done = result.done
            last_info = result.info
        if last_info and last_info["success"]:
            successes += 1
    model.train()
    return successes / n_episodes


def save_model(model, path: str):
    torch.save({"state_dict": model.state_dict()}, path)
    print(f"saved {path}")


def load_model(path: str, env: AtomArrangementEnv):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = PolicyValueNet(env.state_dim, env.n_actions, hidden=128)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model
