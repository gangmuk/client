import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_rps(input_csv, output_pdf):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(input_csv)
    
    # Ensure 'rps' is numeric
    df['rps'] = pd.to_numeric(df['rps'], errors='coerce')
    
    # Drop rows with missing or invalid RPS values
    df = df.dropna(subset=['rps'])
    
    if df.empty:
        print("Error: No valid numeric data in 'rps' column to plot.")
        sys.exit(1)
    
    # Create a unique identifier for each endpoint in a region
    df['region_endpoint'] = df['region'] + " - " + df['endpoint']
    
    # Pivot the data for plotting
    pivot_df = df.pivot(index='counter', columns='region_endpoint', values='rps')
    
    # Check for data to plot
    if pivot_df.empty:
        print("Error: Pivot table contains no data to plot.")
        sys.exit(1)
    
    # Plot the data
    plt.figure(figsize=(12, 8))
    pivot_df.plot(ax=plt.gca(), linestyle='-', marker='o', alpha=0.8)
    
    # Set plot title and labels
    plt.title("Requests Per Second (RPS) Over Time", fontsize=16)
    plt.xlabel("Counter", fontsize=12)
    plt.ylabel("RPS", fontsize=12)
    plt.legend(title="Region - Endpoint", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    
    # Save the plot to the PDF
    plt.savefig(output_pdf, format='pdf')
    print(f"Graph saved to {output_pdf}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python plot_rps.py <input_csv> <output_pdf>")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    output_pdf = sys.argv[2]
    
    plot_rps(input_csv, output_pdf)
