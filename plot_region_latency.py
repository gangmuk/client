import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_latencies(csv_file, output_pdf):
    # Read the CSV into a DataFrame
    df = pd.read_csv(csv_file, header=None, names=['counter', 'region', 'latency'])
    
    # Pivot the data to have one column for each region's latency
    df_pivot = df.pivot(index='counter', columns='region', values='latency')
    
    # Plot latencies for each region over time (counter)
    plt.figure(figsize=(10, 6))
    for region in df_pivot.columns:
        plt.plot(df_pivot.index, df_pivot[region], label=region)

    # Add labels and title
    plt.xlabel('Time (Counter)')
    plt.ylabel('Latency (ms)')
    plt.title('Latency Over Time for Four Regions')
    plt.legend()
    
    # Save the plot to the specified PDF location
    plt.tight_layout()
    plt.savefig(output_pdf)
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python plot_latencies.py <input_csv> <output_pdf>")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_pdf = sys.argv[2]

    plot_latencies(input_csv, output_pdf)
