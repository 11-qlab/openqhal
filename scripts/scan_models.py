"""
Scan all RL checkpoint directories and rank models by success rate.

Looks for run_meta.json and history.json in any rl_checkpoints*/ dir.
Prints a ranked table.
"""
import os
import json
import glob
from pathlib import Path

ROOT = Path.home() / "projects" / "openqhal"

rows = []

for d in sorted(ROOT.glob("rl_checkpoints*")):
    if not d.is_dir():
        continue

    # ── Try run_meta.json first ──
    meta_path = d / "run_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            m = json.load(f)
        rows.append({
            "dir": d.name,
            "source": "run_meta.json",
            "final": m.get("final_success"),
            "dtype": m.get("dtype"),
            "threads": m.get("threads"),
            "mkldnn": m.get("mkldnn"),
            "tag": m.get("tag"),
            "wall_min": m.get("total_wall_clock_s", 0) / 60,
            "platform": m.get("platform"),
        })
        continue

    # ── Fall back to history.json ──
    hist_path = d / "history.json"
    if hist_path.exists():
        with open(hist_path) as f:
            h = json.load(f)
        if isinstance(h, list) and h:
            last = h[-1]
            rows.append({
                "dir": d.name,
                "source": "history.json",
                "final": last.get("success"),
                "stage": last.get("stage"),
                "steps": last.get("steps"),
                "wall_min": last.get("wall_clock_s", 0) / 60,
            })

# ── Also scan variance directories ──
for d in sorted(ROOT.glob("rl_variance*")):
    for sub in d.glob("*"):
        meta = sub / "run_meta.json" if sub.is_dir() else None
        if meta and meta.exists():
            with open(meta) as f:
                m = json.load(f)
            rows.append({
                "dir": f"{d.name}/{sub.name}",
                "source": "run_meta.json",
                "final": m.get("final_success"),
                "dtype": m.get("dtype"),
                "threads": m.get("threads"),
                "tag": m.get("tag"),
                "wall_min": m.get("total_wall_clock_s", 0) / 60,
            })

# ── Print ranked table ──
print()
print("=" * 88)
print("  MODEL RANKING BY SUCCESS RATE")
print("=" * 88)
print()

valid = [r for r in rows if r.get("final") is not None]
valid.sort(key=lambda r: r["final"], reverse=True)

print(f"{'rank':>4}  {'dir':<32}  {'success':>8}  {'source':<18}  {'meta':<30}")
print("-" * 100)

for i, r in enumerate(valid, 1):
    succ = r["final"] * 100
    meta_bits = []
    if r.get("dtype"):    meta_bits.append(r["dtype"])
    if r.get("threads") is not None: meta_bits.append(f"{r['threads']}t")
    if r.get("mkldnn"):   meta_bits.append(f"mkldnn={r['mkldnn']}")
    if r.get("stage"):    meta_bits.append(f"stage{r['stage']}")
    if r.get("wall_min"): meta_bits.append(f"{r['wall_min']:.0f}min")
    meta = " ".join(meta_bits)

    print(f"{i:>4}  {r['dir']:<32}  {succ:>7.1f}%  {r['source']:<18}  {meta:<30}")

print()
print(f"Total models found: {len(valid)}")
print()

# ── Highlight the winner ──
if valid:
    w = valid[0]
    print("=" * 88)
    print(f"  BEST MODEL: {w['dir']}")
    print(f"  Success:    {w['final']*100:.1f}%")
    if w.get("wall_min"):
        print(f"  Wall:       {w['wall_min']:.1f} min")
    print("=" * 88)
    print()
    print("  Checkpoint candidates in that directory:")
    for f in sorted((ROOT / w["dir"]).glob("*.zip")):
        size_mb = f.stat().st_size / 1e6
        print(f"    {f.name}  ({size_mb:.1f} MB)")
