#!/usr/bin/env python3
"""
make_plots.py  <summary.csv>

Outputs:
  heat_gain.pdf          -- Baseline/PAPA latency‐gain heatmap
  heat_splitW.pdf        -- Normalized C→W‐fraction heatmap
  slices_combined.pdf    -- 2×3 grid of per-west_start slice plots
"""

import sys
from pathlib import Path
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ─── Paper‑friendly defaults ───────────────────────
plt.rcParams.update({
    "font.size":       12,
    "axes.titlesize":  12,
    "axes.labelsize":  12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 2.5,
    "axes.linewidth":  1.0,
    "legend.frameon":  False,
    "figure.dpi":      300,
})

DARK_THRESHOLD = 0.5

def pivot_pruned(df, value):
    tbl = df.pivot(index='oS', columns='oW', values=value)
    tbl = tbl.sort_index().sort_index(axis=1)
    tbl = tbl.dropna(how='all', axis=0).dropna(how='all', axis=1)
    return tbl.values, tbl.columns.values, tbl.index.values

def edges_from_centers(c):
    if len(c) > 1:
        d = np.diff(c) / 2
        return np.concatenate([[c[0] - d[0]], c[:-1] + d, [c[-1] + d[-1]]])
    return c + np.array([-0.5, +0.5])

def plot_pcol(mesh, xs, ys, title, cbar_label, out_path, cmap, vmin, vmax):
    xe, ye = edges_from_centers(xs), edges_from_centers(ys)

    fig, ax = plt.subplots(figsize=(4.8, 4.3), constrained_layout=True)
    pcm = ax.pcolormesh(xe, ye, mesh,
                        cmap=cmap, vmin=vmin, vmax=vmax,
                        shading='flat')

    ax.plot([0, 1], [0, 1], '--', color='white', linewidth=1)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box')

    ticks = np.linspace(0.2, 1.0, 5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    fmt = FuncFormatter(lambda v, _: f"{int(v * 100)}%")
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.set_xlabel("$o_W$ (%)")
    ax.set_ylabel("$o_S$ (%)")
    ax.set_title(title, pad=6)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    cbar = fig.colorbar(pcm, ax=ax, pad=0.05, fraction=0.046, aspect=25)
    cbar.set_label(cbar_label, rotation=270, labelpad=14)
    cbar.ax.tick_params(labelsize=10)

    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ {out_path.name}")

def plot_slices_combined(df, west_starts, out_path):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    n = len(west_starts)
    fig, axs = plt.subplots(
        2, n, figsize=(8.5, 4),
        sharex='col', sharey='row',
        gridspec_kw={'height_ratios': [2.8, 1.0]}
    )
    fig.subplots_adjust(top=0.80, hspace=0.25, wspace=0.3)

    handles = []

    for col, w0 in enumerate(west_starts):
        df2 = df[df['west_start'] == w0].sort_values('south_start')
        oS  = (100 - df2['south_start']).clip(0,100) / 100.0
        latB = df2['latency_baseline_ms']
        latP = df2['latency_papa_ms']
        wW   = df2['weight_C_to_W'].clip(upper=0.5)
        wS   = df2['weight_C_to_S'].clip(upper=0.5)

        ax0, ax1 = axs[0, col], axs[1, col]

        # --- Latency ---
        if col == 0:
            hB, = ax0.plot(oS, latB, '--', color='grey', lw=1.8, label="TEASE w/o PAPA")
            hP, = ax0.plot(oS, latP,  '-', color='black', lw=1.8, label="TEASE")
            handles += [hB, hP]
        else:
            ax0.plot(oS, latB, '--', color='grey', lw=1.8)
            ax0.plot(oS, latP,  '-', color='black', lw=1.8)

        ymi, yma = min(latB.min(), latP.min()), max(latB.max(), latP.max())
        pad = 0.05 * (yma - ymi)
        ax0.set_ylim(ymi - pad, yma + pad)
        ax0.set_xlim(0, 1)

        if col == 0:
            ax0.set_ylabel("Latency (ms)")
        ax0.set_title(rf"$o_{{W}}={100-w0}\%$", pad=6)
        ax0.xaxis.set_major_formatter(FuncFormatter(lambda v,_: f"{int(v*100)}%"))
        for sp in ('top','right'):
            ax0.spines[sp].set_visible(False)

        # --- Weights ---
        if col == 0:
            hW, = ax1.step(oS, wW, where='mid', lw=1.8, color='#ff7f0e', label="C→W")
            hS, = ax1.step(oS, wS, where='mid', lw=1.8, color='#1f77b4', label="C→S")
            handles += [hW, hS]
        else:
            ax1.step(oS, wW, where='mid', lw=1.8, color='#ff7f0e')
            ax1.step(oS, wS, where='mid', lw=1.8, color='#1f77b4')

        ax1.fill_between(oS, 0, wW, step='mid', color='#ff7f0e', alpha=0.3)
        ax1.fill_between(oS, 0, wS, step='mid', color='#1f77b4', alpha=0.3)
        ax1.set_ylim(0, 0.5)
        ax1.set_xlim(0, 1)
        if col == 0:
            ax1.set_ylabel("Weight")
        ax1.set_xlabel("$o_S$ (%)")
        ax1.xaxis.set_major_formatter(FuncFormatter(lambda v,_: f"{int(v*100)}%"))
        for sp in ('top','right'):
            ax1.spines[sp].set_visible(False)

    fig.legend(
        handles, ["TEASE w/o PAPA","TEASE","C→W","C→S"],
        loc='upper center', ncol=4,
        frameon=True, framealpha=1.0, edgecolor='lightgray',
        fontsize=10,
        bbox_to_anchor=(0.5, 0.99)
    )

    plt.tight_layout(rect=[0.02,0.02,0.98,0.9])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"✓ {Path(out_path).name}")

if __name__=="__main__":
    if len(sys.argv)!=2:
        print("usage: make_plots.py <summary.csv>")
        sys.exit(1)

    summary_csv = Path(sys.argv[1])
    outdir      = summary_csv.parent
    df          = pd.read_csv(summary_csv)

    total = (df['weight_C_to_W'] + df['weight_C_to_S']).replace(0,1)
    df['normCW']       = df['weight_C_to_W'] / total
    df['gain_clamped'] = df['gain_baseline_div_papa'].clip(lower=1.0)

    gm, xs, ys = pivot_pruned(df, 'gain_clamped')
    plot_pcol(gm, xs, ys,
              "Latency gain", "× imprv.",
              outdir/"heat_gain.pdf",
              cmap='inferno', vmin=1.0, vmax=None)

    fm, xs, ys = pivot_pruned(df, 'normCW')
    plot_pcol(fm, xs, ys,
              "Norm. C→W", "Fraction",
              outdir/"heat_splitW.pdf",
              cmap='plasma', vmin=0, vmax=1)

    plot_slices_combined(df, [0,50,100], outdir/"slices_combined.pdf")

    print("\nDone.")
