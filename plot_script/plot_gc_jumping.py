import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_weight_vs_counter(csv_file, latency_csv, output_pdf, src_cid, title_suffix):
    # Read the CSV files
    df = pd.read_csv(csv_file)
    latency_df = pd.read_csv(latency_csv, names=['counter', 'latency'])

    # Filter the DataFrame based on the specified conditions
    filtered_df = df[(df['src_svc'] == 'sslateingress') &
                     (df['dst_svc'] == 'frontend') &
                     (df['src_cid'] == src_cid)]

    # Convert weight values to numeric and truncate to two decimal places
    filtered_df['weight'] = pd.to_numeric(filtered_df['weight'], errors='coerce').round(2)

    # Ensure 'counter' is numeric in both DataFrames for proper alignment
    filtered_df['counter'] = pd.to_numeric(filtered_df['counter'], errors='coerce')
    latency_df['counter'] = pd.to_numeric(latency_df['counter'], errors='coerce')

    # Drop any NaN values after conversion
    filtered_df = filtered_df.dropna(subset=['counter', 'weight'])
    latency_df = latency_df.dropna(subset=['counter', 'latency'])

    # Resample the latency data to match the density of the counter values in the filtered data
    latency_df = latency_df.set_index('counter').reindex(filtered_df['counter'].unique(), method='nearest').reset_index()

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
    ax2.plot(latency_df['counter'], latency_df['latency'], color='r', linestyle=':', label='Latency')
    ax2.set_ylabel('Latency')
    ax2.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

    plt.title(f'Time vs Weight for different clusters and Latency (Ruleset {title_suffix})')
    plt.tight_layout()
    plt.savefig(output_pdf, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python plot_weight_vs_counter.py <input_csv> <latency_csv> <output_pdf_us_central_1> <output_pdf_us_east_1>")
        sys.exit(1)

    csv_file = sys.argv[1]
    latency_csv = sys.argv[2]
    output_pdf_us_central_1 = sys.argv[3]
    output_pdf_us_east_1 = sys.argv[4]

    # Plot for src_cid us-central-1
    plot_weight_vs_counter(csv_file, latency_csv, output_pdf_us_central_1, 'us-central-1', 'us-central-1')

    # Plot for src_cid us-east-1
    plot_weight_vs_counter(csv_file, latency_csv, output_pdf_us_east_1, 'us-east-1', 'us-east-1')