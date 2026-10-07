"""
PPO training for the RL atom rearrangement compiler.

Clipped surrogate objective, GAE advantage, multiple epochs per rollout.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


@dataclass
class PPOConfig:
    total_steps: int = 60_000
    rollout_len: int = 512
    epochs_per_rollout: int = 4
    mini_batch: int = 128
    gamma: float = 0.97
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    lr: float = 3e-4
    entropy_coef: float = 0.02
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    log_every: int = 2000
    seed: int = 42


class PPOTrainer:
    def __init__(self, env, policy, cfg: PPOConfig = None):
        self.env = env
        self.policy = policy
        self.cfg = cfg or PPOConfig()
        self.opt = torch.optim.Adam(policy.parameters(), lr=self.cfg.lr)

        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

    # ── rollout collection ──
    @torch.no_grad()
    def collect_rollout(self):
        cfg = self.cfg
        states, actions, log_probs, values, rewards, dones = [], [], [], [], [], []

        state = self.env.reset()
        episode_success = []
        episode_steps = []

        for _ in range(cfg.rollout_len):
            logits, value = self.policy(state)
            mask = self.env.action_mask()
            if mask.shape[0] == logits.shape[0]:
                logits = logits.masked_fill(~mask, -1e9)
            dist = Categorical(logits=logits)
            action = dist.sample()
            lp = dist.log_prob(action)

            result = self.env.step(action.item())

            states.append(state)
            actions.append(action.item())
            log_probs.append(lp.item())
            values.append(value.item())
            rewards.append(result.reward)
            dones.append(float(result.done))

            state = result.state

            if result.done:
                episode_success.append(result.info["success"])
                episode_steps.append(result.info["steps"])
                state = self.env.reset()

        return {
            "states": states, "actions": actions, "log_probs": log_probs,
            "values": values, "rewards": rewards, "dones": dones,
            "success": episode_success, "steps": episode_steps,
        }

    # ── GAE ──
    def compute_gae(self, rollout):
        rewards = np.array(rollout["rewards"], dtype=np.float32)
        values = np.array(rollout["values"], dtype=np.float32)
        dones = np.array(rollout["dones"], dtype=np.float32)

        # Bootstrap from last state
        with torch.no_grad():
            _, last_value = self.policy(self.env._obs())
        last_value = float(last_value)

        advantages = np.zeros_like(rewards)
        gae = 0.0
        next_value = last_value
        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0.0
                gae = 0.0
            delta = rewards[t] + self.cfg.gamma * next_value - values[t]
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * gae
            advantages[t] = gae
            next_value = values[t]

        returns = advantages + values
        return torch.tensor(advantages), torch.tensor(returns)

    # ── PPO update ──
    def update(self, rollout):
        cfg = self.cfg
        states = torch.stack(rollout["states"])
        actions = torch.tensor(rollout["actions"], dtype=torch.long)
        old_log_probs = torch.tensor(rollout["log_probs"], dtype=torch.float32)
        advantages, returns = self.compute_gae(rollout)

        # Normalize advantage
        if advantages.numel() > 1:
            adv_std = advantages.std()
            if torch.isfinite(adv_std) and adv_std > 1e-8:
                advantages = (advantages - advantages.mean()) / adv_std

        N = len(states)
        for _ in range(cfg.epochs_per_rollout):
            perm = torch.randperm(N)
            for start in range(0, N, cfg.mini_batch):
                idx = perm[start:start + cfg.mini_batch]
                logits, value = self.policy(states[idx])
                mask = self.env.action_mask()
                if mask.shape[0] == logits.shape[1]:
                    logits = logits.masked_fill(~mask, -1e9)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions[idx])
                entropy = dist.entropy().mean()

                ratio = (new_log_probs - old_log_probs[idx]).exp()
                adv_b = advantages[idx]
                surr1 = ratio * adv_b
                surr2 = ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(value, returns[idx])
                loss = policy_loss + cfg.value_coef * value_loss \
                       - cfg.entropy_coef * entropy

                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(),
                                                cfg.max_grad_norm)
                self.opt.step()

    # ── training loop ──
    def train(self, total_steps: Optional[int] = None):
        cfg = self.cfg
        total = total_steps or cfg.total_steps
        history = []
        collected = 0
        while collected < total:
            rollout = self.collect_rollout()
            self.update(rollout)
            collected += cfg.rollout_len

            if rollout["success"]:
                history.append({
                    "step": collected,
                    "success": np.mean(rollout["success"]),
                    "steps": np.mean(rollout["steps"]),
                })

            if collected % cfg.log_every < cfg.rollout_len and history:
                recent = history[-max(1, len(history) // 5):]
                succ = np.mean([h["success"] for h in recent])
                steps = np.mean([h["steps"] for h in recent])
                print(f"  step {collected:6d}  success {succ:5.1%}  "
                      f"steps {steps:4.1f}")
        return history
