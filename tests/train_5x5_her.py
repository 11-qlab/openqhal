"""
Train 5x5 with Hindsight Experience Replay on top of PPO.

The key change: after every episode, we relabel the trajectory with
K random future states as new goals, and add those relabeled
transitions to the PPO buffer. Every failure becomes a useful signal.
"""
import sys, time
sys.path.insert(0, "python")

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from qhal.rl import AtomArrangementEnv, PolicyValueNet, PPOConfig
from qhal.rl.ppo import PPOTrainer


def relabel_reward(s_grid, s_next_grid, goal_grid, gs):
    """Recompute reward for a transition under a new goal."""
    d_prev = float((s_grid != goal_grid).sum())
    d_next = float((s_next_grid != goal_grid).sum())
    delta = d_prev - d_next
    reward = 0.5 * delta
    done = bool((s_next_grid == goal_grid).all())
    if done:
        reward += 5.0
    return reward, done


def make_her_episode_transitions(ep_transitions, gs, k_future=4, rng=None):
    """
    ep_transitions: list of (state_tensor, action, log_prob, value,
                             next_state_tensor, done)
    Returns list of augmented transitions (state, action, log_prob, value,
                                             next_state, done, reward_override)
    """
    if rng is None:
        rng = np.random.default_rng()
    T = len(ep_transitions)
    augmented = []
    for t in range(T):
        s, a, lp, v, s_next, d = ep_transitions[t]
        s_np = s.detach().cpu().numpy()
        sn_np = s_next.detach().cpu().numpy()

        # Original reward from the actual env
        augmented.append((s, a, lp, v, s_next, float(d), None))

        # K relabeled versions
        for _ in range(k_future):
            # Sample a future grid as the virtual goal
            j = int(rng.integers(t, T))
            future_sn = ep_transitions[j][4].detach().cpu().numpy()
            goal_grid = future_sn[:gs]

            # Rebuild state with the new goal
            new_s = np.concatenate([
                s_np[:gs],                          # current grid
                goal_grid,                          # new target
                (s_np[:gs] != goal_grid).astype(np.float32),
                s_np[3*gs:3*gs+1],
            ])
            new_sn = np.concatenate([
                sn_np[:gs],
                goal_grid,
                (sn_np[:gs] != goal_grid).astype(np.float32),
                sn_np[3*gs:3*gs+1],
            ])

            new_r, new_done = relabel_reward(s_np[:gs], sn_np[:gs],
                                              goal_grid, gs)
            augmented.append((
                torch.tensor(new_s, dtype=torch.float32),
                a, lp, v,
                torch.tensor(new_sn, dtype=torch.float32),
                float(new_done),
                new_r,
            ))
    return augmented


def ppo_her_train(env, model, cfg: PPOConfig,
                  total_steps: int, gs: int, k_future: int = 4):
    """PPO with HER-augmented buffer."""
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    rng = np.random.default_rng(cfg.seed + 1)

    env.reset()
    episode_buffer = []
    state = env.reset()
    total = 0
    while total < total_steps:
        # Collect one full episode
        state = env.reset()
        ep = []
        done = False
        while not done:
            with torch.no_grad():
                logits, value = model(state)
                mask = env.action_mask()
                if mask.shape[0] == logits.shape[0]:
                    logits = logits.masked_fill(~mask, -1e9)
                dist = Categorical(logits=logits)
                action = dist.sample()
                lp = dist.log_prob(action)

            result = env.step(action.item())
            ep.append((state, action.item(), lp.item(), value.item(),
                        result.state, float(result.done)))
            state = result.state
            done = result.done
            total += 1
            if total >= total_steps:
                break

        # HER augmentation
        aug = make_her_episode_transitions(ep, gs, k_future=k_future, rng=rng)

        # PPO update on the augmented buffer
        states = torch.stack([x[0] for x in aug])
        actions = torch.tensor([x[1] for x in aug], dtype=torch.long)
        old_lp = torch.tensor([x[2] for x in aug], dtype=torch.float32)
        rewards = torch.tensor([x[6] if x[6] is not None else 0.0
                                 for x in aug], dtype=torch.float32)
        dones = torch.tensor([x[5] for x in aug], dtype=torch.float32)

        # Recompute reward for original transitions from the actual env step
        # (the env already returned it via result.reward; for simplicity use
        # the delta-based reward here)
        for i, x in enumerate(aug):
            if x[6] is None:
                s_np = x[0].detach().cpu().numpy()
                sn_np = x[4].detach().cpu().numpy()
                r, _ = relabel_reward(s_np[:gs], sn_np[:gs], sn_np[gs:2*gs], gs)
                rewards[i] = r

        # GAE (simplified single-episode version)
        with torch.no_grad():
            _, next_v = model(state)
        next_v = float(next_v)
        adv = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_val = next_v if t == len(rewards) - 1 else 0.0
            delta = rewards[t] + cfg.gamma * next_val * (1 - dones[t]) - 0.0
            gae = delta + cfg.gamma * cfg.gae_lambda * (1 - dones[t]) * gae
            adv[t] = gae
        returns = adv + 0.0
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # PPO epochs
        for _ in range(cfg.epochs_per_rollout):
            perm = torch.randperm(len(aug))
            for s_i in range(0, len(aug), cfg.mini_batch):
                idx = perm[s_i:s_i + cfg.mini_batch]
                logits, value = model(states[idx])
                mask = env.action_mask()
                if mask.shape[0] == logits.shape[1]:
                    logits = logits.masked_fill(~mask, -1e9)
                dist = Categorical(logits=logits)
                new_lp = dist.log_prob(actions[idx])
                entropy = dist.entropy().mean()
                ratio = (new_lp - old_lp[idx]).exp()
                a = adv[idx]
                s1 = ratio * a
                s2 = ratio.clamp(1-cfg.clip_eps, 1+cfg.clip_eps) * a
                pol_loss = -torch.min(s1, s2).mean()
                val_loss = F.mse_loss(value, returns[idx])
                loss = pol_loss + cfg.value_coef * val_loss - cfg.entropy_coef * entropy
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

        if total % 20_000 < 1_000:
            print(f"  step {total:7d}")


def main():
    H = W = 5
    gs = H * W
    cfg = dict(n_atoms=5, max_steps=13, hidden=512)

    env = AtomArrangementEnv(grid_size=H, n_atoms=cfg["n_atoms"],
                             max_steps=cfg["max_steps"],
                             target_scramble=3, seed=42)
    env.reset()
    model = PolicyValueNet(env.state_dim, env.n_actions, hidden=cfg["hidden"])

    # Warm start from stage 2
    import os
    ckpt_path = "/tmp/rl_checkpoints/rl_5x5_stage2.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        print(f"warm-started from {ckpt_path}")

    pcfg = PPOConfig(
        total_steps=400_000, rollout_len=2048,
        epochs_per_rollout=4, mini_batch=256,
        gamma=0.98, gae_lambda=0.95, clip_eps=0.2,
        lr=1.5e-4, entropy_coef=0.05, value_coef=0.5,
        log_every=20_000, seed=42,
    )

    print(f"\nTraining 5x5 with HER (k_future=4)")
    t0 = time.time()
    ppo_her_train(env, model, pcfg, total_steps=400_000, gs=gs, k_future=4)
    print(f"\ntime: {time.time()-t0:.1f}s")

    # Eval
    from tests.test_rl_scaling_all import quick_eval
    for s in [1, 2, 3]:
        acc = quick_eval(model, H, W, cfg["n_atoms"], s, 13,
                         episodes=200, seed=8000+s)
        print(f"  scramble={s}: {acc:.1%}")

    torch.save({"state_dict": model.state_dict(), "size": 5},
               "/tmp/rl_5x5_her.pt")


if __name__ == "__main__":
    main()
