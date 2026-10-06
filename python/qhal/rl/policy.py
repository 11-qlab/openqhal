"""
Policy and value networks for the atom rearrangement RL compiler.

Two heads on a shared trunk:
  - policy head:  action logits
  - value head:   state-value estimate V(s)

The trunk is a small MLP (grid is 4x4 -> 32 features; MLP is enough).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyValueNet(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, n_actions)
        self.value_head  = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        return self.policy_head(h), self.value_head(h).squeeze(-1)


class RolloutBuffer:
    """Storage for REINFORCE / actor-critic rollouts."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.states: list = []
        self.actions: list = []
        self.rewards: list = []
        self.log_probs: list = []
        self.values: list = []
        self.dones: list = []

    def add(self, state, action, reward, log_prob, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def tensors(self):
        return {
            "states":    torch.stack(self.states),
            "actions":   torch.tensor(self.actions, dtype=torch.long),
            "rewards":   torch.tensor(self.rewards, dtype=torch.float32),
            "log_probs": torch.stack(self.log_probs),
            "values":    torch.stack(self.values),
            "dones":     torch.tensor(self.dones, dtype=torch.float32),
        }

    def __len__(self):
        return len(self.states)
