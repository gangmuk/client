import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def set_style():
    """Set the general style parameters for matplotlib with increased font sizes"""
    plt.rcParams.update({
        'font.size': 20,
        'axes.labelsize': 24,
        'axes.titlesize': 26,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'legend.fontsize': 20,
        'lines.linewidth': 3,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
        'axes.spines.top': True,
        'axes.spines.right': True,
        'axes.spines.left': True,
        'axes.spines.bottom': True
    })

def extract_request_type(endpoint, src_svc, dst_svc):
    """
    Extracts the request type from the endpoint string, including source and destination services.
    Example: 'corecontrast@POST@/singlecore' -> 'corecontrast-frontend-post-singlecore'
    """
    parts = endpoint.split('@')
    if len(parts) < 3:
        return "unknown"
    method = parts[1].lower()
    path = parts[2].strip('/').replace('/', '-')
    return f"{src_svc}-{dst_svc}-{method}-{path}"

def plot_weight_vs_counter(df, latency_df, output_pdf, src_cid, request_type):
    set_style()

    # Filter & clean
    filt = df[(df['src_cid']==src_cid)&(df['request_type']==request_type)]
    if filt.empty:
        print(f"No data for {src_cid} / {request_type}")
        return
    filt['counter']=pd.to_numeric(filt['counter'],errors='coerce')
    filt['weight'] =pd.to_numeric(filt['weight'], errors='coerce').round(2)
    latency_df['counter']=pd.to_numeric(latency_df['counter'],errors='coerce')
    filt.dropna(subset=['counter','weight'], inplace=True)
    latency_df.dropna(subset=['counter','latency'], inplace=True)

    # Align latency to weight counters
    latency_df = (
        latency_df.set_index('counter')
                  .reindex(filt['counter'].unique(), method='nearest')
                  .reset_index()
    )

    # Plot setup
    fig, ax1 = plt.subplots(figsize=(12,7))
    plt.subplots_adjust(left=0.12, right=0.88, top=0.88, bottom=0.15)

    # Weight curves
    colors = ['#1f77b4','#2ca02c','#ff7f0e']
    for (dst,grp),c in zip(filt.groupby('dst_cid'),colors):
        ax1.plot(grp['counter'], grp['weight'],
                 color=c, linestyle='-', linewidth=4,
                 label=dst, zorder=5)

    # Latency on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(latency_df['counter'], latency_df['latency'],
             color='red', linestyle='--',
            #  marker='X', markersize=10,
            #  markerfacecolor='yellow', markeredgecolor='black',
             linewidth=3, label='Latency',
             alpha=1.0, zorder=7)

    # Axes labels & limits
    ax1.set_xlabel('Time (s)', fontsize=26, fontweight='bold')
    ax1.set_ylabel('Traffic Weight', fontsize=26, fontweight='bold')
    ax1.set_ylim(0,1)
    ax1.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
    ax1.tick_params(labelsize=24)

    max_lat = latency_df['latency'].max()*1.1
    ax2.set_ylabel('Global Latency', fontsize=26, color='red', fontweight='bold')
    ax2.set_ylim(0, max_lat)
    ax2.tick_params(labelsize=24, colors='red')
    ax2.spines['right'].set_color('red')

    ax1.grid(True, alpha=0.2)

    # Combined legend INSIDE plot at around t=250, weight~0.8
    handles = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels  = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax1.legend(handles, labels,
               loc='upper left',
               bbox_to_anchor=(0.55, 0.99),
               frameon=True, facecolor='white',
               edgecolor='black', framealpha=0.9)

    # Title
    plt.title(f'Weights and Latency for Ruleset {src_cid}',
              fontsize=26, pad=15, fontweight='bold')

    # Save
    plt.savefig(output_pdf, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved {output_pdf}")

def main():
    if len(sys.argv)!=4:
        print("Usage: python plot_weight_vs_counter.py <input_csv> <latency_csv> <out_dir>")
        sys.exit(1)

    csv_file, latency_file, out_dir = sys.argv[1:]
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_file)
    latency_df = pd.read_csv(latency_file, names=['counter','latency'])

    # Drop repeated headers
    df = df[pd.to_numeric(df['counter'],errors='coerce').notnull()].copy()
    df.reset_index(drop=True, inplace=True)

    # Request types
    df['request_type'] = df.apply(
        lambda r: extract_request_type(r['dst_endpoint'], r['src_svc'], r['dst_svc']),
        axis=1
    )

    regions = df['src_cid'].unique()
    reqs    = [r for r in df['request_type'].unique() if r!="unknown"]

    for region in regions:
        for rt in reqs:
            safe_rt = rt.replace('/','-').replace('@','-')
            out_pdf = os.path.join(out_dir, f"{region}-{safe_rt}.pdf")
            plot_weight_vs_counter(df, latency_df, out_pdf, region, rt)

if __name__=="__main__":
    main()
