import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

def plot_latency_comparison_on_average(csv_file, output_image):
    """
    Plot latency comparison graphs for each scheme using latency percentiles
    and display the values on the bars with a 45-degree rotation.
    """
    # Read the parsed CSV file
    df = pd.read_csv(csv_file)

    # Define latency columns
    latency_columns = ["LatencyMin", "LatencyMean", "Latency50", "Latency90", "Latency95", "Latency99", "LatencyMax"]

    # Group data by Scheme
    grouped = df.groupby("Scheme")[latency_columns].mean()

    # Plot the latency comparison
    ax = grouped.T.plot(kind="bar", figsize=(10, 6), rot=0)
    plt.title("Latency Comparison by Scheme")
    plt.xlabel("Latency Percentiles")
    plt.ylabel("Latency (ms)")
    plt.legend(title="Scheme")
    plt.grid()
    plt.tight_layout()

    # Annotate values on the bars
    for container in ax.containers:
        labels = ax.bar_label(container, fmt="%.2f", label_type="edge")
        # Rotate each label by 45 degrees
        for label in labels:
            label.set_rotation(45)
            label.set_ha("right")  # Adjust alignment for better readability

    # Save the plot to a file
    plt.savefig(output_image)
    print(f"Latency comparison plot saved as {output_image}")


def plot_latency_comparison_individually(csv_file, output_image):
    """
    Plot latency comparison graphs for each scheme using latency percentiles.
    """
    # Read the parsed CSV file
    df = pd.read_csv(csv_file)

    # Define latency columns
    latency_columns = ["LatencyMin", "LatencyMean", "Latency50", "Latency90", "Latency95", "Latency99", "LatencyMax"]

    # Melt the DataFrame for scatter plot
    melted_df = df.melt(
        id_vars=["Scheme"], value_vars=latency_columns, 
        var_name="LatencyMetric", value_name="LatencyValue"
    )

    # Scatter plot
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=melted_df, x="LatencyMetric", y="LatencyValue", hue="Scheme", alpha=0.7, markers="o", s=100)
    plt.title("Latency Comparison by Scheme (Scatter Plot)")
    plt.xlabel("Latency Percentiles")
    plt.ylabel("Latency (ms)")
    plt.legend(title="Scheme")
    plt.tight_layout()
    plt.grid()

    # Save the plot to a file
    plt.savefig(output_image)
    print(f"Latency comparison plot saved as {output_image}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot-corecontrast.py <dir>")
        sys.exit(1)
    base_dir = sys.argv[1]
    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        sys.exit(1)
    csv_file = f"{base_dir}/latency_comparison_results.csv"  # Replace with your CSV file
    output_image_on_average = f"{base_dir}/latency_comparison_on_average.png"
    output_image_individually = f"{base_dir}/latency_comparison_individually.png"
    plot_latency_comparison_on_average(csv_file, output_image_on_average)
    plot_latency_comparison_individually(csv_file, output_image_individually)

if __name__ == "__main__":
    main()
