#!/usr/bin/env python3
"""
collect_hyper_sweep.py   <sweep_root> [--t TARGET_TIME]

Walk the directory tree from your sweep and sample both latency and
routing weights at the row whose timestamp (or counter) is closest to
TARGET_TIME (default=180). Emits <sweep_root>/summary.csv with:

    west_start, south_start, oW, oS,
    latency_baseline_ms, latency_papa_ms,
    weight_C_to_W, weight_C_to_S,
    gain_baseline_div_papa
"""

import sys, re, csv
from pathlib import Path
import pandas as pd

# ───────────────────────── helper utilities ──────────────────────────
def latency_at(csv_path: Path, target: float) -> float:
    """
    From the 2‑col CSV (time, latency), return the latency at the row
    with time closest to 'target'.
    """
    best = None
    best_dt = float('inf')
    with csv_path.open() as f:
        reader = csv.reader(f)
        for parts in reader:
            if len(parts) < 2:
                continue
            try:
                t = float(parts[0])
                lat = float(parts[1])
            except ValueError:
                continue
            dt = abs(t - target)
            if dt < best_dt:
                best_dt, best = dt, lat
    if best is None:
        raise ValueError(f"No valid latency rows in {csv_path}")
    return best

def weights_at(history_csv: Path, target: float) -> tuple[float, float]:
    """
    From the routing‑history CSV (with 'counter' column), pick the
    counter value nearest to 'target', then extract the weights for
    Central→West and Central→South from that counter block.
    """
    if not history_csv.exists():
        return 0.0, 0.0

    df = pd.read_csv(history_csv)
    # convert counter to numeric (some rows may be non-numeric)
    df['counter'] = pd.to_numeric(df['counter'], errors='coerce')
    df = df.dropna(subset=['counter'])
    if df.empty:
        return 0.0, 0.0

    # find the counter nearest to target
    df['dt'] = (df['counter'] - target).abs()
    closest_ctr = df.loc[df['dt'].idxmin(), 'counter']
    block = df[df['counter'] == closest_ctr]

    def w(dst: str) -> float:
        row = block[
            (block['src_cid'] == 'us-central-1') &
            (block['dst_cid'] == dst)
        ]
        return float(row['weight'].iloc[0]) if not row.empty else 0.0

    return w('us-west-1'), w('us-south-1')

def overlap_fraction(start: int, width: int = 100) -> float:
    """Compute |[0,100] ∩ [start, start+width]| / 100."""
    a0, a1 = 0, 100
    b0, b1 = start, start + width
    inter = max(0, min(a1, b1) - max(a0, b0))
    return inter / 100.0
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # parse args
    if len(sys.argv) not in (2,3):
        print("Usage: collect_hyper_sweep.py <sweep_root> [--t=TARGET_TIME]")
        sys.exit(1)

    root = Path(sys.argv[1]).resolve()
    if not root.is_dir():
        sys.exit(f"Error: {root} is not a directory")

    target = 180.0
    if len(sys.argv) == 3:
        try:
            target = float(sys.argv[2].split('=')[1])
        except:
            pass

    pat = re.compile(r'west_(\d+)_south_(\d+)')
    rows = []

    for sub in root.glob('west_*_south_*'):
        m = pat.fullmatch(sub.name)
        if not m:
            continue

        west_start, south_start = map(int, m.groups())
        exp = sub / 'getRecord-w100c500e1200s100'
        dir_papa = exp / 'SLATE-with-jumping-global'
        dir_base = exp / 'SLATE-without-jumping'

        # sample latencies
        try:
            lat_papa = latency_at(
                dir_papa / 'SLATE-with-jumping-global-jumping_latency.csv',
                target)
            lat_base = latency_at(
                dir_base / 'SLATE-without-jumping-jumping_latency.csv',
                target)
        except Exception as e:
            print(f"[warn] {sub.name}: latency error: {e}")
            continue

        # sample routing weights
        try:
            w_c2w, w_c2s = weights_at(
                dir_papa / 'SLATE-with-jumping-global-jumping_routing_history.csv',
                target)
        except Exception as e:
            print(f"[warn] {sub.name}: weight error: {e}")
            w_c2w, w_c2s = 0.0, 0.0

        oW = overlap_fraction(west_start)
        oS = overlap_fraction(south_start)
        gain = lat_base / lat_papa if lat_papa else float('inf')

        rows.append({
            'west_start': west_start,
            'south_start': south_start,
            'oW': oW, 'oS': oS,
            'latency_baseline_ms': lat_base,
            'latency_papa_ms': lat_papa,
            'weight_C_to_W': w_c2w,
            'weight_C_to_S': w_c2s,
            'gain_baseline_div_papa': gain
        })

    # output
    if not rows:
        sys.exit("No valid runs found.")
    df = pd.DataFrame(rows)
    df = df.sort_values(['west_start','south_start'])
    out_csv = root / 'summary.csv'
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}  ({len(df)} rows)")

if __name__ == "__main__":
    main()
