"""
Scale-up test: 3x3 grid, PPO, Transformer policy, Schedule integration.
"""
import sys, time
sys.path.insert(0, "python")

import numpy as np
import torch

from qhal.rl import (
    AtomArrangementEnv, PolicyValueNet, TransformerAdapter,
    PPOTrainer, PPOConfig,
    RLCompiler, moves_to_program, moves_to_schedule,
)


def eval_success(model, H, W, n_atoms, scramble, episodes=200, seed=1234):
    env = AtomArrangementEnv(grid_size=H, n_atoms=n_atoms,
                             max_steps=8, target_scramble=scramble,
                             seed=seed)
    model.eval()
    wins = 0
    with torch.no_grad():
        for _ in range(episodes):
            state = env.reset()
            done = False
            info = None
            while not done:
                logits, _ = model(state)
                action = int(logits.argmax().item())
                r = env.step(action)
                state = r.state
                done = r.done
                info = r.info
            if info and info["success"]:
                wins += 1
    return wins / episodes


def main():
    print("=" * 68)
    print("  RL compiler scale-up: 3x3 + PPO + Transformer + Schedule")
    print("=" * 68)

    H = W = 3
    n_atoms = 3
    max_steps = 8

    # ── MLP + masking + PPO on 3x3 ──
    print(f"\n[MLP + MASKING] PPO training — 3x3, {n_atoms} atoms")
    env = AtomArrangementEnv(grid_size=H, n_atoms=n_atoms,
                             max_steps=max_steps, target_scramble=1,
                             seed=42)
    mlp = PolicyValueNet(env.state_dim, env.n_actions, hidden=128)
    cfg = PPOConfig(total_steps=60_000, rollout_len=512,
                    epochs_per_rollout=4, mini_batch=128,
                    entropy_coef=0.03, value_coef=0.5,
                    lr=3e-4, gamma=0.95,
                    log_every=4000, seed=42)
    trainer = PPOTrainer(env, mlp, cfg)

    t0 = time.time()
    for stage, (scramble, steps) in enumerate([
            (1, 20_000), (2, 20_000), (3, 20_000)], 1):
        env.target_scramble = scramble
        print(f"\n  --- stage {stage}: scramble={scramble} ---")
        trainer.train(total_steps=steps)
        for s in [1, 2, 3]:
            acc = eval_success(mlp, H, W, n_atoms, s, episodes=200,
                               seed=300 + s)
            print(f"    eval scramble={s}: {acc:.1%}")
    print(f"\n  training time: {time.time()-t0:.1f}s")

    torch.save({"state_dict": mlp.state_dict()},
               "/tmp/rl_mlp_3x3.pt")

    # ── D: Schedule integration ──
    print(f"\n[D] Schedule integration")
    compiler = RLCompiler(model_path="/tmp/rl_mlp_3x3.pt",
                          grid_size=H, n_atoms=n_atoms,
                          use_transformer=False)

    # A specific rearrangement
    initial = np.array([[1,1,0],[0,0,0],[0,0,1]], dtype=np.int8)
    target  = np.array([[0,1,0],[0,1,0],[0,1,0]], dtype=np.int8)
    moves, ok = compiler.compile(initial, target, max_steps=8)
    print(f"  compiled {len(moves)} moves, success={ok}")
    for m in moves[:6]:
        print(f"    {m}")

    # Convert to Program
    prog = moves_to_program(moves, n_atoms=n_atoms,
                             duration_per_move_ns=200.0)
    print(f"  Program: {len(prog.pulses)} pulses, "
          f"{prog.total_duration_ns:.0f} ns")

    # Convert to Schedule
    sched = moves_to_schedule(moves)
    if sched is not None:
        print(f"  Schedule: depth={sched.depth}, "
              f"gates={sched.num_gates}, method={sched.method}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
