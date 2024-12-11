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
        'xtick.labelsize': 24,      # Increased from 20 to 24
        'ytick.labelsize': 24,      # Increased from 20 to 24
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

def abbreviate_traffic_class(traffic_class):
    """
    Abbreviate the traffic class name to make it more concise
    Example: frontend-backend-post-data -> fe-be-post-data
    """
    parts = traffic_class.split('-')
    if len(parts) >= 4:
        # Abbreviate service names
        parts[0] = parts[0][:2]  # First service
        parts[1] = parts[1][:2]  # Second service
        # Keep the HTTP method and path
        return '-'.join(parts)
    return traffic_class

def plot_weight_vs_counter(df, latency_df, output_pdf, src_cid, request_type, title_suffix):
    """
    Plots weight vs counter and latency for a specific region and request type.
    Now with extended, thick time markers crossing the x-axis and increased font sizes.
    """
    set_style()
    
    filtered_df = df[
        (df['src_cid'] == src_cid) &
        (df['request_type'] == request_type)
    ]

    if filtered_df.empty:
        print(f"No data for region '{src_cid}' with request type '{request_type}'. Skipping plot.")
        return

    # Data preparation
    filtered_df['weight'] = pd.to_numeric(filtered_df['weight'], errors='coerce').round(2)
    filtered_df['counter'] = pd.to_numeric(filtered_df['counter'], errors='coerce')
    latency_df['counter'] = pd.to_numeric(latency_df['counter'], errors='coerce')
    filtered_df = filtered_df.dropna(subset=['counter', 'weight'])
    latency_df = latency_df.dropna(subset=['counter', 'latency'])
    latency_df = latency_df.set_index('counter').reindex(filtered_df['counter'].unique(), method='nearest').reset_index()

    # Create figure
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Adjust margins
    plt.subplots_adjust(left=0.12, right=0.88, top=0.88, bottom=0.15)

    # Plot weight vs counter first
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e']
    grouped = filtered_df.groupby('dst_cid')
    
    for (dst_cid, group), color in zip(grouped, colors):
        ax1.plot(group['counter'], group['weight'], 
                linestyle='-', 
                linewidth=4,
                color=color,
                label=dst_cid,
                zorder=5)

    # Plot latency
    ax2 = ax1.twinx()
    ax2.plot(latency_df['counter'], latency_df['latency'], 
             color='red',
             linestyle=':',
             marker='x',
             markersize=6,
             linewidth=2,
             label='Latency',
             alpha=0.7,
             zorder=4)
    
    # Add extended, thick markers at specific times
    marker_times = [(360, 'blue'), (720, 'black'), (900, 'green')]
    
    for time, color in marker_times:
        ax1.vlines(x=time,
                  ymin=0,
                  ymax=0.075,
                  color=color,
                  linewidth=14,
                  zorder=7)
        
        ax1.vlines(x=time,
                  ymin=-0.1,
                  ymax=0,
                  color=color,
                  linewidth=8,
                  zorder=7)
    
    # Set y-axis limits
    ax1.set_ylim(0, 1)
    y2_max = max(latency_df['latency']) * 1.1
    ax2.set_ylim(0, y2_max)

    # Customize axes with increased font sizes
    ax1.set_xlabel('Time (s)', fontsize=26, fontweight='bold')
    ax1.set_ylabel('Traffic Weight', fontsize=26, fontweight='bold')
    ax2.set_ylabel('Global Latency', fontsize=26, color='red', fontweight='bold')
    
    ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.tick_params(axis='both', which='major', labelsize=24)  # Increased to 24
    ax2.tick_params(axis='y', labelsize=24, colors='red')      # Increased to 24
    ax2.spines['right'].set_color('red')
    
    ax1.grid(True, alpha=0.2)

    # Place legend inside the plot at top left with increased font size
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, 
              loc='upper left',
              fontsize=20,
              frameon=True,
              facecolor='white',
              edgecolor='black',
              framealpha=0.8)

    # Set title with increased font size
    plt.title(f'Weights and Latency for Ruleset {src_cid}', 
             fontsize=26, 
             pad=15,
             fontweight='bold')

    plt.savefig(output_pdf, 
                bbox_inches='tight', 
                dpi=300,
                facecolor='white',
                edgecolor='none')
    plt.close()
    print(f"Plot saved to {output_pdf}")

def extract_request_type(endpoint, src_svc, dst_svc):
    """
    Extracts the request type from the endpoint string, including source and destination services.
    Example: 'corecontrast@POST@/singlecore' -> 'corecontrast-frontend-post-singlecore'
    """
    try:
        parts = endpoint.split('@')
        if len(parts) < 3:
            return "unknown"
        method = parts[1].lower()
        path = parts[2].strip('/').replace('/', '-')
        return f"{src_svc}-{dst_svc}-{method}-{path}"
    except Exception as e:
        print(f"Error extracting request type from endpoint '{endpoint}': {e}")
        return "unknown"

def main():
    if len(sys.argv) != 4:
        print("Usage: python plot_weight_vs_counter.py <input_csv> <latency_csv> <output_directory>")
        sys.exit(1)

    csv_file = sys.argv[1]
    latency_csv = sys.argv[2]
    output_directory = sys.argv[3]

    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Read the CSV files
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading input CSV file '{csv_file}': {e}")
        sys.exit(1)

    try:
        latency_df = pd.read_csv(latency_csv, names=['counter', 'latency'])
    except Exception as e:
        print(f"Error reading latency CSV file '{latency_csv}': {e}")
        sys.exit(1)

    # Remove any rows where 'counter' is not numeric to eliminate repeated headers
    df = df[pd.to_numeric(df['counter'], errors='coerce').notnull()].copy()

    # Reset index after filtering
    df.reset_index(drop=True, inplace=True)

    # Extract request types from 'dst_endpoint'
    df['request_type'] = df.apply(lambda row: extract_request_type(row['dst_endpoint'], row['src_svc'], row['dst_svc']), axis=1)

    # Detect unique regions from 'src_cid'
    regions = df['src_cid'].unique()

    # Detect unique request types from 'request_type' column, excluding 'unknown'
    request_types = df['request_type'].unique()
    request_types = [rt for rt in request_types if rt != "unknown"]

    print(f"Detected regions: {regions}")
    print(f"Detected request types: {request_types}")

    # Iterate over each region and request type to generate plots
    for region in regions:
        for request_type in request_types:
            print(f"Generating plot for region '{region}' with request type '{request_type}'")
            safe_request_type = request_type.replace('/', '-').replace('@', '-')
            output_pdf = os.path.join(
                output_directory,
                f"{region}-{safe_request_type}.pdf"
            )

            plot_weight_vs_counter(df, latency_df, output_pdf, region, request_type, safe_request_type)

if __name__ == "__main__":
    main()