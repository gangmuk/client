import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_latencies(csv_file, output_pdf_base):
    # Read the CSV into a DataFrame
    df = pd.read_csv(csv_file, header=None, names=['counter', 'region', 'endpoint', 'latency'])
    
    # Filter out rows where the endpoint is missing (only three values provided)
    df = df.dropna(subset=['endpoint'])
    
    # Filter to include only rows with at least four entries per region-endpoint combination
    df['region_endpoint'] = df['region'] + ' - ' + df['endpoint']
    valid_region_endpoints = df['region_endpoint'].value_counts()
    valid_region_endpoints = valid_region_endpoints[valid_region_endpoints >= 4].index
    df_filtered = df[df['region_endpoint'].isin(valid_region_endpoints)]
    
    # Get unique request types from the 'endpoint' column
    request_types = df_filtered['endpoint'].unique()
    
    # Function to plot latencies by request type
    def plot_by_request_type(request_type, output_pdf):
        # Filter data for the specific request type
        df_request = df_filtered[df_filtered['endpoint'] == request_type]
        
        # Pivot the data to have one column for each region's latency
        df_pivot = df_request.pivot(index='counter', columns='region', values='latency')
        
        # Plot latencies for each region over time (counter)
        plt.figure(figsize=(12, 8))
        for region in df_pivot.columns:
            plt.plot(df_pivot.index, df_pivot[region], label=region)

        # Add labels and title
        plt.xlabel('Time (Counter)')
        plt.ylabel('Latency (ms)')
        plt.title(f'Latency Over Time for {request_type}')
        
        # Place the legend outside the plot area
        plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize='small')
        
        # Save the plot to the specified PDF location
        plt.tight_layout()
        plt.savefig(output_pdf, bbox_inches="tight")
        plt.close()

    # Generate plots for each unique request type
    for request_type in request_types:
        # Create an output filename based on the request type
        safe_request_type = request_type.replace("/", "-").replace("@", "-")
        plot_by_request_type(request_type, f"{output_pdf_base}-{safe_request_type}.pdf")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python plot_latencies.py <input_csv> <output_pdf_base>")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_pdf_base = sys.argv[2]

    plot_latencies(input_csv, output_pdf_base)