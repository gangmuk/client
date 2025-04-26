import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def plot_weight_vs_counter_comparison(df1, df2, output_pdf, src_cid, request_type, title_suffix):
    """
    Plots weight vs counter for a specific region and request type from two different CSV files.
    """
    # Filter DataFrames based on the specified src_cid and request_type
    filtered_df1 = df1[
        (df1['src_cid'] == src_cid) &
        (df1['request_type'] == request_type)
    ]

    filtered_df2 = df2[
        (df2['src_cid'] == src_cid) &
        (df2['request_type'] == request_type)
    ]

    if filtered_df1.empty and filtered_df2.empty:
        print(f"No data for region '{src_cid}' with request type '{request_type}' in either CSV. Skipping plot.")
        return

    # Convert weight values to numeric and truncate to two decimal places
    filtered_df1['weight'] = pd.to_numeric(filtered_df1['weight'], errors='coerce').round(2)
    filtered_df2['weight'] = pd.to_numeric(filtered_df2['weight'], errors='coerce').round(2)

    # Ensure 'counter' is numeric
    filtered_df1['counter'] = pd.to_numeric(filtered_df1['counter'], errors='coerce')
    filtered_df2['counter'] = pd.to_numeric(filtered_df2['counter'], errors='coerce')

    # Drop any NaN values after conversion
    filtered_df1 = filtered_df1.dropna(subset=['counter', 'weight'])
    filtered_df2 = filtered_df2.dropna(subset=['counter', 'weight'])

    # Group by 'dst_cid' for each DataFrame
    grouped1 = filtered_df1.groupby('dst_cid')
    grouped2 = filtered_df2.groupby('dst_cid')

    fig, ax = plt.subplots(figsize=(12, 7))

    # Linestyles and colors for differentiation
    linestyles = ['-', '--', '-.', ':']
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']

    # Plot weight vs counter for the first CSV
    for i, (dst_cid, group) in enumerate(grouped1):
        ax.plot(group['counter'], group['weight'], 
                linestyle=linestyles[i % len(linestyles)], 
                color=colors[i % len(colors)], 
                label=f'CSV1 - dst_cid: {dst_cid}')
    
    # Plot weight vs counter for the second CSV
    for i, (dst_cid, group) in enumerate(grouped2):
        ax.plot(group['counter'], group['weight'], 
                linestyle=linestyles[i % len(linestyles)], 
                color=colors[(i + len(grouped1)) % len(colors)], 
                label=f'CSV2 - dst_cid: {dst_cid}')
    
    ax.set_xlabel('Counter')
    ax.set_ylabel('Weight')
    ax.set_ylim(0, 1)
    ax.set_yticks([i * 0.1 for i in range(11)])
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True)

    plt.title(f'Time vs Weight Comparison for {src_cid} [{request_type}]')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(output_pdf, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to {output_pdf}")

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
        print("Usage: python plot_weight_vs_counter_comparison.py <input_csv1> <input_csv2> <output_directory>")
        sys.exit(1)

    csv_file1 = sys.argv[1]
    csv_file2 = sys.argv[2]
    output_directory = sys.argv[3]

    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Read the CSV files
    try:
        df1 = pd.read_csv(csv_file1)
    except Exception as e:
        print(f"Error reading first input CSV file '{csv_file1}': {e}")
        sys.exit(1)

    try:
        df2 = pd.read_csv(csv_file2)
    except Exception as e:
        print(f"Error reading second input CSV file '{csv_file2}': {e}")
        sys.exit(1)

    # Remove any rows where 'counter' is not numeric to eliminate repeated headers
    df1 = df1[pd.to_numeric(df1['counter'], errors='coerce').notnull()].copy()
    df2 = df2[pd.to_numeric(df2['counter'], errors='coerce').notnull()].copy()

    # Reset indices after filtering
    df1.reset_index(drop=True, inplace=True)
    df2.reset_index(drop=True, inplace=True)

    # Extract request types from 'dst_endpoint'
    df1['request_type'] = df1.apply(lambda row: extract_request_type(row['dst_endpoint'], row['src_svc'], row['dst_svc']), axis=1)
    df2['request_type'] = df2.apply(lambda row: extract_request_type(row['dst_endpoint'], row['src_svc'], row['dst_svc']), axis=1)

    # Get all unique regions across both CSVs
    regions = set(df1['src_cid'].unique()).union(set(df2['src_cid'].unique()))

    # Get unique request types across both CSVs, excluding 'unknown'
    request_types1 = set(df1['request_type'].unique()) - {"unknown"}
    request_types2 = set(df2['request_type'].unique()) - {"unknown"}
    request_types = request_types1.intersection(request_types2)

    print(f"Detected regions: {regions}")
    print(f"Detected common request types: {request_types}")

    # Iterate over each region and request type to generate comparison plots
    for region in regions:
        for request_type in request_types:
            # Generate a safe filename
            safe_request_type = request_type.replace('/', '-').replace('@', '-')
            output_pdf = os.path.join(
                output_directory,
                f"comparison-{region}-{safe_request_type}.pdf"
            )

            plot_weight_vs_counter_comparison(df1, df2, output_pdf, region, request_type, request_type)

if __name__ == "__main__":
    main()