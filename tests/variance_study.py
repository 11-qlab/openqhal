"""
Variance study for the SB3 + CNN RL compiler on 5x5.

Runs the same config N times with different seeds. Reports:
  - per-seed final success
  - mean, std, min, max
  - comparison to the Colab T4 baseline

Usage:
    python tests/variance_study.py --size 5 --seeds 3
    python tests/variance_study.py --size 5 --seeds 5 --steps 400000
    python tests/variance_study.py --size 3 --seeds 5    # fast sanity check

Resumes: if a seed's result JSON already exists, it's skipped.
"""
import os, sys, time, json, argparse, random
sys.path.insert(0, "python")
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from qhal.rl.gym_wrapper import ShapedAtomGym
from qhal.rl.features import AtomGridCNN
from stable_baselines3 import PPO


# ─────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────

CONFIGS = {
    3: dict(n_atoms=3, grid=3,
            stages=[dict(scramble=1, max_steps=5,  shape_scale=0.500),
                    dict(scramble=2, max_steps=9,  shape_scale=0.200),
                    dict(scramble=3, max_steps=13, shape_scale=0.050)],
            features_dim=128, hidden=128,
            threshold=0.90),

    5: dict(n_atoms=5, grid=5,
            stages=[dict(scramble=1, max_steps=5,  shape_scale=0.500),
                    dict(scramble=2, max_steps=9,  shape_scale=0.200),
                    dict(scramble=3, max_steps=13, shape_scale=0.050)],
            features_dim=256, hidden=512,
            threshold=0.80),
}


# ─────────────────────────────────────────────────────────────────
#  Single-seed training
# ─────────────────────────────────────────────────────────────────

def train_seed(seed: int, cfg: dict, total_steps: int,
                results_dir: str, verbose: bool = True):
    """Train one seed through all stages. Return dict of results."""

    grid = cfg["grid"]
    env = ShapedAtomGym(grid_size=grid, n_atoms=cfg["n_atoms"],
                        max_steps=cfg["stages"][0]["max_steps"],
                        scramble=cfg["stages"][0]["scramble"],
                        shape_scale=cfg["stages"][0]["shape_scale"],
                        seed=seed)

    eval_env = ShapedAtomGym(grid_size=grid, n_atoms=cfg["n_atoms"],
                             max_steps=cfg["stages"][-1]["max_steps"],
                             scramble=cfg["stages"][-1]["scramble"],
                             shape_scale=0.0, seed=seed + 9999)

    policy_kwargs = dict(
        features_extractor_class=AtomGridCNN,
        features_extractor_kwargs=dict(
            features_dim=cfg["features_dim"], grid_size=grid),
        net_arch=dict(pi=[128], vf=[128]))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO(
        "MlpPolicy", env, verbose=0,
        learning_rate=3e-4, n_steps=2048, batch_size=256, n_epochs=4,
        gamma=0.98, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.05, vf_coef=0.5, max_grad_norm=1.0,
        policy_kwargs=policy_kwargs,
        device=device, seed=seed)

    def evaluate(model, env, episodes=100, seed_base=10_000 + seed * 1000):
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

    result = {"seed": seed, "stages": [], "device": device}

    for stage_idx, stage in enumerate(cfg["stages"]):
        env.set_curriculum(scramble=stage["scramble"],
                            max_steps=stage["max_steps"],
                            shape_scale=stage["shape_scale"])
        model.ent_coef = max(0.01, 0.05 - stage_idx * 0.015)

        stage_steps = 0
        success = 0.0
        stage_start = time.time()

        # Per-stage cap: allow up to `max_per_stage` steps before giving up.
        # Stages advance on SUCCESS THRESHOLD, not step budget, so different
        # seeds can spend different amounts of time.
        max_per_stage = total_steps  # total_steps is now the CAP PER STAGE
        EVAL_INTERVAL = 20_480

        while (stage_steps < max_per_stage
                and success < cfg["threshold"]):
            model.learn(total_timesteps=EVAL_INTERVAL,
                        reset_num_timesteps=False)
            stage_steps += EVAL_INTERVAL

            eval_env.set_curriculum(scramble=stage["scramble"],
                                     max_steps=stage["max_steps"],
                                     shape_scale=0.0)
            success = evaluate(model, eval_env, episodes=100)

            if verbose:
                print(f"    seed={seed} stage={stage_idx+1} "
                      f"step={stage_steps:7d} succ={success*100:5.1f}%")

        # Post-stage eval at all scramble levels
        stage_result = {
            "stage": stage_idx + 1,
            "scramble": stage["scramble"],
            "steps": stage_steps,
            "success_at_stage": success,
            "wall_clock_s": time.time() - stage_start,
            "success_scramble_1": 0.0,
            "success_scramble_2": 0.0,
            "success_scramble_3": 0.0,
        }
        for s in [1, 2, 3]:
            eval_env.set_curriculum(scramble=s,
                                     max_steps=stage["max_steps"],
                                     shape_scale=0.0)
            acc = evaluate(model, eval_env, episodes=50)
            stage_result[f"success_scramble_{s}"] = acc

        result["stages"].append(stage_result)

        # Save the checkpoint
        ckpt = os.path.join(results_dir,
                             f"seed{seed}_stage{stage_idx+1}")
        model.save(ckpt)

        if success < cfg["threshold"] and stage_idx < len(cfg["stages"]) - 1:
            # Stalled on this stage; can still try next
            pass

    # Final eval at hardest scramble, 200 episodes
    eval_env.set_curriculum(scramble=cfg["stages"][-1]["scramble"],
                             max_steps=cfg["stages"][-1]["max_steps"],
                             shape_scale=0.0)
    result["final_success_scramble_3"] = evaluate(
        model, eval_env, episodes=200, seed_base=50_000 + seed)

    return result


# ─────────────────────────────────────────────────────────────────
#  Driver
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=3,
                         help="Number of different seeds to run")
    parser.add_argument("--steps", type=int, default=800_000,
                         help="Max steps PER STAGE (cap, not fixed budget)")
    parser.add_argument("--seed-start", type=int, default=42)
    args = parser.parse_args()

    if args.size not in CONFIGS:
        print(f"No config for {args.size}x{args.size}")
        return

    cfg = CONFIGS[args.size]
    results_dir = os.path.expanduser(
        f"~/projects/openqhal/rl_variance_{args.size}x{args.size}")
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n{'#'*68}")
    print(f"  Variance study: {args.size}x{args.size}, "
          f"{args.seeds} seeds, {args.steps:,} steps/seed")
    print(f"  Checkpoints: {results_dir}")
    print(f"{'#'*68}\n")

    all_results = []
    t_start = time.time()

    for i in range(args.seeds):
        seed = args.seed_start + i
        result_path = os.path.join(results_dir, f"seed{seed}_result.json")

        if os.path.exists(result_path):
            print(f"[seed {seed}] already done, loading...")
            with open(result_path) as f:
                all_results.append(json.load(f))
            continue

        print(f"\n{'='*68}")
        print(f"[seed {seed}]  ({i+1}/{args.seeds})")
        print(f"{'='*68}")

        t_seed = time.time()
        try:
            result = train_seed(seed, cfg, args.steps, results_dir,
                                 verbose=True)
        except KeyboardInterrupt:
            print(f"\n[seed {seed}] interrupted by user")
            break

        result["total_wall_clock_s"] = time.time() - t_seed

        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)

        all_results.append(result)

        print(f"\n[seed {seed}] final success on scramble=3: "
              f"{result['final_success_scramble_3']:.1%}")
        print(f"[seed {seed}] wall clock: {result['total_wall_clock_s']:.0f}s")

    # ── Summary ──
    if not all_results:
        print("\nNo seeds completed.")
        return

    finals = [r["final_success_scramble_3"] for r in all_results]
    stage1 = [r["stages"][0]["success_at_stage"] for r in all_results
              if r.get("stages")]
    stage2 = [r["stages"][1]["success_at_stage"] for r in all_results
              if len(r.get("stages", [])) > 1]
    stage3 = [r["stages"][2]["success_at_stage"] for r in all_results
              if len(r.get("stages", [])) > 2]

    print(f"\n{'='*68}")
    print(f"  SUMMARY: {len(all_results)} seeds")
    print(f"{'='*68}\n")

    def stats(vals, name):
        if not vals:
            print(f"  {name:25s} n/a")
            return
        a = np.array(vals) * 100
        print(f"  {name:25s} mean={a.mean():5.1f}%  "
              f"std={a.std():4.1f}  "
              f"min={a.min():5.1f}  max={a.max():5.1f}")

    stats(stage1, "Stage 1 peak")
    stats(stage2, "Stage 2 peak")
    stats(stage3, "Stage 3 peak")
    stats(finals, "Final scramble=3 (200 ep)")

    print(f"\n  Per-seed final success:")
    for r in all_results:
        print(f"    seed {r['seed']:5d}  "
              f"scramble3 = {r['final_success_scramble_3']*100:5.1f}%")

    total_time = time.time() - t_start
    print(f"\n  Total study time: {total_time/60:.1f} min")

    # Save summary
    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "size": args.size,
            "seeds": args.seeds,
            "steps_per_seed": args.steps,
            "final_success": finals,
            "mean": float(np.mean(finals)),
            "std": float(np.std(finals)),
            "min": float(np.min(finals)),
            "max": float(np.max(finals)),
            "per_seed": all_results,
        }, f, indent=2)

    print(f"\n  Saved: {summary_path}")


if __name__ == "__main__":
    main()
