"""
Hindsight Experience Replay (Andrychowicz et al. 2017) for the
atom rearrangement environment.

Every failed episode is relabeled with the achieved final grid as a
new goal. The trajectory that failed to reach target T1 becomes a
successful trajectory toward goal G_achieved. This turns sparse-reward
failures into dense, useful training signal.
"""
from __future__ import annotations
from typing import List, Tuple
import numpy as np
import torch


class HERBuffer:
    def __init__(self, k_future: int = 4, rng_seed: int = 0):
        self.k_future = k_future
        self.rng = np.random.default_rng(rng_seed)
        self.episodes: List[list] = []   # list of episodes, each is list of transitions

    def add_episode(self, transitions: list):
        """Store one full episode: list of (state, action, log_prob, value)."""
        self.episodes.append(transitions)

    def sample_batch(self, batch_size: int, gs: int):
        """Sample a mixed batch of original + relabeled transitions."""
        all_transitions = []
        for ep in self.episodes:
            all_transitions.extend(self._relabel_episode(ep, gs))

        if len(all_transitions) < batch_size:
            idxs = self.rng.integers(0, len(all_transitions), size=batch_size)
        else:
            idxs = self.rng.choice(len(all_transitions), size=batch_size,
                                    replace=False)

        batch = [all_transitions[i] for i in idxs]
        states = torch.stack([b[0] for b in batch])
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        next_states = torch.stack([b[3] for b in batch])
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32)
        return states, actions, rewards, next_states, dones

    def _relabel_episode(self, ep: list, gs: int):
        """Return original + K relabeled transitions per step."""
        out = []
        T = len(ep)
        for t, (s, a, lp, v) in enumerate(ep):
            # Original transition
            s_np = s.detach().cpu().numpy()
            # We need the NEXT state which is the next transition's state,
            # or if this is the last step, the state after the step.
            # For simplicity, we assume the transitions include both s and s'.
            # (Adjust the caller to pass (s, a, lp, v, s_next, done).)
            out.append((s, a, 0.0, s, 0.0))  # placeholder; see patch below
        return out
