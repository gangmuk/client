#!/usr/bin/env python3
"""
plot_slices.py  <summary.csv>  [out_dir]

Creates one PNG per distinct west_start:
    slice_west_<west>.png
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11,
    "figure.dpi": 120,
    "axes.facecolor": "white"
})

# ───────────────────────────── I/O ──────────────────────────────
if len(sys.argv) < 2:
    print("Usage: plot_slices.py  <summary.csv>  [out_dir]")
    sys.exit(1)

csv_path = Path(sys.argv[1]).resolve()
out_dir  = Path(sys.argv[2] if len(sys.argv) > 2 else csv_path.parent)
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_path)

# ───────────────────── slice & plot helper ─────────────────────
def plot_slice(west_start: int, slice_df: pd.DataFrame, out_dir: Path):
    slice_df = slice_df.sort_values('south_start')

    x  = slice_df['south_start']
    yB = slice_df['latency_baseline_ms']
    yP = slice_df['latency_papa_ms']
    wW = slice_df['weight_C_to_W']

    fig, ax1 = plt.subplots(figsize=(6.0, 4.0))

    ax1.plot(x, yB, label='Baseline', color='slategray', linewidth=2.5)
    ax1.plot(x, yP, label='PAPA',     color='navy',      linewidth=2.5)
    ax1.set_xlabel("South start key")
    ax1.set_ylabel("Mean latency (ms)")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, wW, label='PAPA C→W weight', color='orange',
             linestyle='--', linewidth=2.5)
    ax2.fill_between(x, 0, wW, step='mid', alpha=0.15, color='orange')
    ax2.set_ylabel("Central→West weight")
    ax2.set_ylim(0, 1)

    # legend – combine both axes
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', ncol=3, frameon=False)

    ax1.set_title(f"Slice  west_start={west_start}")

    out_file = out_dir / f"slice_west_{west_start}.png"
    plt.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)
    print(f"  ✓  {out_file.relative_to(out_dir.parent)}")

# ───────────────────────── main loop ────────────────────────────
print(f"Writing PNGs into  {out_dir}")
for west in sorted(df['west_start'].unique()):
    plot_slice(west, df[df['west_start'] == west], out_dir)
