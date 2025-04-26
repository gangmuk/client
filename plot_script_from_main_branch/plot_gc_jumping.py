import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def plot_weight_vs_counter(df, latency_df, output_pdf, src_cid, request_type, title_suffix):
    """
    Plots weight vs counter and latency for a specific region and request type.
    """
    # Filter the DataFrame based on the specified src_cid and request_type
    filtered_df = df[
        (df['src_cid'] == src_cid) &
        (df['request_type'] == request_type)
    ]

    if filtered_df.empty:
        print(f"No data for region '{src_cid}' with request type '{request_type}'. Skipping plot.")
        return

    # Convert weight values to numeric and truncate to two decimal places
    filtered_df['weight'] = pd.to_numeric(filtered_df['weight'], errors='coerce').round(2)

    # Ensure 'counter' is numeric in both DataFrames for proper alignment
    filtered_df['counter'] = pd.to_numeric(filtered_df['counter'], errors='coerce')
    latency_df['counter'] = pd.to_numeric(latency_df['counter'], errors='coerce')

    # Drop any NaN values after conversion
    filtered_df = filtered_df.dropna(subset=['counter', 'weight'])
    latency_df = latency_df.dropna(subset=['counter', 'latency'])

    # Create a function to find nearest latency value
    def find_nearest_latency(counter_value):
        return latency_df.iloc[(latency_df['counter'] - counter_value).abs().argsort()[0]]['latency']

    # Create a new column with matched latency values
    counter_values = filtered_df['counter'].unique()
    matched_latencies = pd.DataFrame({
        'counter': counter_values,
        'latency': [find_nearest_latency(c) for c in counter_values]
    })

    # Group by 'dst_cid'
    grouped = filtered_df.groupby('dst_cid')

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot weight vs counter
    for dst_cid, group in grouped:
        ax1.plot(group['counter'], group['weight'], linestyle='-', label=f'dst_cid: {dst_cid}')
    
    ax1.set_xlabel('Counter')
    ax1.set_ylabel('Weight')
    ax1.set_ylim(0, 1)
    ax1.set_yticks([i * 0.1 for i in range(11)])
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True)

    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Plot latency on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(matched_latencies['counter'], matched_latencies['latency'], 
            color='r', linestyle=':', label='Latency')
    ax2.set_ylabel('Latency')
    ax2.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

    plt.title(f'Time vs Weight for {src_cid} [{request_type}] (Ruleset {title_suffix})')
    plt.tight_layout()
    plt.savefig(output_pdf, bbox_inches='tight')
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

            # Use the request_type as the title suffix
            title_suffix = request_type.replace('-', ' ').title()

            plot_weight_vs_counter(df, latency_df, output_pdf, region, request_type, title_suffix)

if __name__ == "__main__":
    main()