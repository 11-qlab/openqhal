"""
Scan EVERY SB3 checkpoint on the system, infer its grid size from
the observation space, evaluate it, and rank by success rate.

Search paths:
    ~/projects/openqhal/     (main repo)
    ~/qhal_models/           (persistent model dir)
    /tmp/                    (temporary checkpoints)
    /content/openqhal/       (Colab, if mounted)
"""
import os
import sys
import json
import glob
import zipfile
from pathlib import Path

sys.path.insert(0, "python")
import numpy as np
import torch
from stable_baselines3 import PPO

ROOT = Path.home() / "projects" / "openqhal"

SEARCH_PATHS = [
    ROOT,
    Path.home() / "qhal_models",
    Path("/tmp"),
    Path("/content/openqhal"),
]


# ─────────────────────────────────────────────────────────────────
#  Discovery
# ─────────────────────────────────────────────────────────────────

def looks_like_sb3_checkpoint(p: Path) -> bool:
    """A .zip is an SB3 checkpoint if it contains policy.pth + system_info.txt."""
    try:
        with zipfile.ZipFile(p) as z:
            names = set(z.namelist())
            return "policy.pth" in names and "system_info.txt" in names
    except (zipfile.BadZipFile, OSError):
        return False


def find_all_sb3_checkpoints():
    """Return every SB3 checkpoint under SEARCH_PATHS."""
    found = []
    seen_hashes = set()
    for root in SEARCH_PATHS:
        if not root.exists():
            continue
        for p in root.rglob("*.zip"):
            size_mb = p.stat().st_size / (1024 * 1024)
            if size_mb < 1 or size_mb > 200:
                continue
            if not looks_like_sb3_checkpoint(p):
                continue
            # Skip duplicated copies of the same file (same size + name)
            key = (p.name, p.stat().st_size)
            if key in seen_hashes:
                continue
            seen_hashes.add(key)
            found.append(p)
    return sorted(set(found))


# ─────────────────────────────────────────────────────────────────
#  Architecture inference
# ─────────────────────────────────────────────────────────────────

def infer_grid_from_obs(obs_shape):
    """obs = (3*H*W + 1,) -> H (assumes square grid)."""
    n = int(obs_shape[0])
    area = (n - 1) // 3
    side = int(round(area ** 0.5))
    if side * side != area:
        return None
    return side


def infer_n_atoms(path: Path, grid_size: int) -> int:
    """
    Guess the training-time n_atoms.
    Heuristic: parse from path (e.g. 'sb3_5x5', '5x5'), default to grid//1.
    """
    name = path.stem.lower()
    # Common patterns: sb3_5x5, rl_4x4, rl_compiler_3x3
    import re
    m = re.search(r"(\d+)x(\d+)", name)
    if m:
        # If grid matches, atoms is usually same or nearby
        return grid_size  # safe default

    # Fallback: use 20% occupancy (matches most of our training runs)
    cells = grid_size * grid_size
    return max(1, round(0.2 * cells))


# ─────────────────────────────────────────────────────────────────
#  Evaluation
# ─────────────────────────────────────────────────────────────────

def evaluate_model(path: Path, grid_size: int, n_atoms: int,
                   episodes: int = 100, scramble: int = 3):
    """Load model and evaluate. Returns success stats."""
    from qhal.rl.infer import evaluate_checkpoint
    try:
        stats = evaluate_checkpoint(
            str(path), grid_size=grid_size, n_atoms=n_atoms,
            episodes=episodes, scramble=scramble)
        return stats["success_rate"]
    except Exception as e:
        return None


# ─────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 100)
    print("  FULL SB3 CHECKPOINT SCAN")
    print("=" * 100)
    print()

    zips = find_all_sb3_checkpoints()
    print(f"Found {len(zips)} SB3 checkpoints\n")

    if not zips:
        print("No SB3 checkpoints found.")
        return

    rows = []

    for p in zips:
        try:
            model = PPO.load(str(p), device="cpu")
            obs_shape = tuple(model.observation_space.shape)
            n_actions = int(model.action_space.n)
            n_params = sum(q.numel() for q in model.policy.parameters())
            grid = infer_grid_from_obs(obs_shape)

            row = {
                "path": p,
                "rel": str(p.relative_to(ROOT)) if ROOT in p.parents else str(p),
                "obs": obs_shape,
                "grid": grid,
                "n_actions": n_actions,
                "n_params": n_params,
                "success": None,
                "error": None,
                "n_atoms": None,
            }

            # Evaluate if grid is inferable
            if grid is not None:
                n_atoms = infer_n_atoms(p, grid)
                row["n_atoms"] = n_atoms
                s = evaluate_model(p, grid, n_atoms,
                                    episodes=100, scramble=3)
                row["success"] = s

            rows.append(row)

        except Exception as e:
            rows.append({
                "path": p,
                "rel": str(p),
                "error": f"{type(e).__name__}: {e}",
            })

    # ── Table ──
    print(f"{'path':<58}  {'grid':>4}  {'obs':>6}  {'acts':>4}  {'success':>8}")
    print("-" * 90)

    for r in rows:
        if r.get("error"):
            print(f"{r['rel']:<58}  ERR: {r['error']}")
            continue
        grid = r["grid"] or "?"
        obs = r["obs"][0]
        acts = r["n_actions"]
        succ = (f"{r['success']*100:5.1f}%"
                if r["success"] is not None else "  —   ")
        print(f"{r['rel']:<58}  {grid:>4}  {obs:>6}  {acts:>4}  {succ:>8}")

    # ── Ranking ──
    ranked = [r for r in rows
              if r.get("success") is not None]
    ranked.sort(key=lambda r: r["success"], reverse=True)

    print()
    print("=" * 100)
    print("  RANKING BY SUCCESS RATE (scramble=3, 100 episodes)")
    print("=" * 100)
    print()
    print(f"{'rank':>4}  {'grid':>4}  {'params':>10}  {'success':>8}  {'path':<50}")
    print("-" * 90)

    for i, r in enumerate(ranked, 1):
        print(f"{i:>4}  {r['grid']:>4}  {r['n_params']:>10,}  "
              f"{r['success']*100:>7.1f}%  {r['rel']:<50}")

    if ranked:
        print()
        print("=" * 100)
        print(f"  WINNER: {ranked[0]['rel']}")
        print(f"  Grid:   {ranked[0]['grid']}x{ranked[0]['grid']}")
        print(f"  Atoms:  {ranked[0]['n_atoms']}")
        print(f"  Success: {ranked[0]['success']*100:.1f}%")
        print("=" * 100)


if __name__ == "__main__":
    main()
