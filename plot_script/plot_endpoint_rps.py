import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_rps(input_csv, output_pdf, service_filter=None, endpoint_filter=None):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(input_csv)
    
    # Filter by service if service_filter is provided
    if service_filter:
        df = df[df['service'] == service_filter]
    
    # Filter by endpoint if endpoint_filter is provided
    if endpoint_filter:
        df = df[df['endpoint'] == endpoint_filter]
    
    if df.empty:
        print("No data found for the given filters.")
        sys.exit(1)
    
    # Create a unique identifier for each endpoint in a region
    df['region_endpoint'] = df['region'] + " - " + df['endpoint']
    
    # Pivot the data for plotting
    pivot_df = df.pivot(index='counter', columns='region_endpoint', values='rps')
    
    # Plot the data
    plt.figure(figsize=(12, 8))
    pivot_df.plot(ax=plt.gca(), linestyle='-', marker='o', alpha=0.8)
    
    # Set plot title and labels
    title = "Requests Per Second (RPS) Over Time"
    if service_filter:
        title += f" - Service: {service_filter}"
    if endpoint_filter:
        title += f" - Endpoint: {endpoint_filter}"
    plt.title(title, fontsize=16)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("RPS", fontsize=12)
    plt.legend(title="Region - Endpoint", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    
    # Save the plot to the PDF
    plt.savefig(output_pdf, format='pdf')
    print(f"Graph saved to {output_pdf}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python plot_rps.py <input_csv> <output_pdf> [<service_filter>] [<endpoint_filter>]")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    output_pdf = sys.argv[2]
    service_filter = sys.argv[3] if len(sys.argv) > 3 else None
    endpoint_filter = sys.argv[4] if len(sys.argv) > 4 else None
    
    plot_rps(input_csv, output_pdf, service_filter, endpoint_filter)
