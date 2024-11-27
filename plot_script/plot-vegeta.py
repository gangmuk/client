import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

color = {"west": "blue", "central": "green", "east": "brown", "south": "orange"}


def encode_binary_to_csv(input_bin, output_csv):
    """Convert a binary file to CSV using Vegeta, if not already converted."""
    # if os.path.exists(output_csv):
    #     print(f"CSV file {output_csv} already exists. Skipping conversion.")
    #     return
    print(f"Converting {input_bin} to {output_csv}...")
    try:
        with open(output_csv, "w") as outfile:
            subprocess.run(
                ["vegeta", "encode", "--to", "csv", input_bin],
                stdout=outfile,
                check=True
            )
        print(f"Created CSV file: {output_csv}")
    except subprocess.CalledProcessError:
        print(f"Error converting {input_bin} to CSV. Ensure Vegeta is installed.")
        sys.exit(1)

def convert_binaries_to_csv_parallel(input_dir):
    """Convert all binary files in the input directory to CSV in parallel."""
    binary_files = [
        (os.path.join(input_dir, bin_file), f"{input_dir}/{bin_file}.csv")
        for bin_file in os.listdir(input_dir) if bin_file.endswith(".results.bin")
    ]
    print(f"Found {len(binary_files)} binary files to process.")
    with ThreadPoolExecutor() as executor:
        executor.map(lambda args: encode_binary_to_csv(*args), binary_files)


def csv_to_df(csv_file, cluster, usecols=None, dtypes=None):
    """Load a CSV file into a DataFrame with appropriate column names."""
    column_names = [
        "Timestamp", "HTTP Status", "Request Latency", "Bytes Out", "Bytes In", 
        "Error", "Base64 Body", "Attack Name", "Sequence Number", "Method", "URL", "Headers"
    ]
    df = pd.read_csv(csv_file, header=None, names=column_names, usecols=usecols, dtype=dtypes)
    df["Cluster"] = cluster
    return df


def load_csv_parallel(args):
    """Helper function to load a single CSV file (for parallel execution)."""
    csv_file, cluster, usecols, dtypes = args
    return csv_to_df(csv_file, cluster, usecols=usecols, dtypes=dtypes)

def load_and_merge_csvs_parallel(input_dir):
    """Load and merge all CSV files in the input directory using parallel processing."""
    usecols = ["Timestamp", "HTTP Status", "Request Latency"]  # Load only necessary columns
    dtypes = {
        "Timestamp": "int64",
        "HTTP Status": "category",
        "Request Latency": "float64",
    }
    csv_files = []
    for csv_file in [f for f in os.listdir(input_dir) if f.endswith(".results.bin.csv")]:
        cluster = csv_file.split(".")[-4]
        if cluster not in ["west", "central", "east", "south"]:
            print(f"Error: Cluster name {cluster} is not valid.")
            continue
        csv_files.append((f"{input_dir}/{csv_file}", cluster, usecols, dtypes))
    with ThreadPoolExecutor() as executor:
        dfs = list(executor.map(load_csv_parallel, csv_files))
    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"Parallel csv_to_df and concat: {int(time.time() - ts)} seconds")
    return merged_df


def plot_latency_and_load(merged_df, input_dir):
    ts = time.time()
    merged_df["Request Latency (ms)"] = merged_df["Request Latency"] / 1e6
    print(f"1: {int(time.time() - ts)}")
    
    ts = time.time()
    merged_df["Start Time (s)"] = (merged_df["Timestamp"] - merged_df["Timestamp"].min()) / 1e9
    print(f"2: {int(time.time() - ts)}")
    
    ts = time.time()
    merged_df["Time (s)"] = merged_df["Start Time (s)"].astype(int)
    print(f"3: {int(time.time() - ts)}")
    
    ts = time.time()
    merged_df["Request Latency (ms)"] = merged_df["Request Latency"] / 1e6
    print(f"4: {int(time.time() - ts)}")
    
    ts = time.time()
    merged_df["Start Time (s)"] = (merged_df["Timestamp"] - merged_df["Timestamp"].min()) / 1e9
    print(f"5: {int(time.time() - ts)}")
    
    ts = time.time()
    merged_df["Time (s)"] = merged_df["Start Time (s)"].astype(int)
    print(f"6: {int(time.time() - ts)}")
    
    ts = time.time()
    rps_per_second = merged_df.groupby(["Cluster", "Time (s)"]).size()
    print(f"7: {int(time.time() - ts)}")
    
    ts = time.time()
    merged_df["RPS"] = merged_df["Time (s)"].map(rps_per_second)
    print(f"8: {int(time.time() - ts)}")
    
    ## It took 68 seconds
    # ts = time.time()
    # merged_df.to_csv("merged_df.csv")
    # print(f"9: {int(time.time() - ts)}")
    
    ts = time.time()
    max_average_latency = 0
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    for cluster in merged_df["Cluster"].unique():
        df_cluster = merged_df[merged_df["Cluster"] == cluster]
        print(f"Cluster: {cluster}")
        aggregated = df_cluster.groupby(["Time (s)"]).agg(RPS=("Time (s)", "count"), AvgLatency=("Request Latency (ms)", "mean")).reset_index()
        ax1.plot(aggregated["Time (s)"], aggregated["RPS"], label=f"RPS-{cluster}", color=color[cluster], linewidth=1, linestyle=":")
        # ax2.scatter(df_cluster["Start Time (s)"], df_cluster["Request Latency (ms)"], label=f"Individual Request Latency-{cluster}", color=color[cluster], alpha=0.05, zorder=1)
        ax2.plot(aggregated["Time (s)"], aggregated["AvgLatency"], label=f"Average Latency (ms)-{cluster}",color=color[cluster], linewidth=1.5, marker="^", markersize=0)
        max_average_latency = max(max_average_latency, aggregated["AvgLatency"].max())
    
    total_rps = merged_df.groupby(["Time (s)"]).size()
    total_avg_latency = merged_df.groupby(["Time (s)"]).agg(AvgLatency=("Request Latency (ms)", "mean")).reset_index()
    ax1.plot(total_rps, color="red", linewidth=1.5, label="Total RPS")
    ax2.plot(total_avg_latency["Time (s)"], total_avg_latency["AvgLatency"], color="black", linewidth=1.5, label="Total Average Latency")
    ax1.set_xlabel("Time (s)")
    ax1.set_xlim(left=0)
    ax1.set_ylabel("Requests Per Second (RPS)", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax2.set_ylabel("Average Latency (ms)", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax1.set_ylim(0, 1.1 * total_rps.max())
    ax2.set_ylim(0, 1.1 * max_average_latency)
    ax1.legend(loc="upper left", fontsize="small")
    ax2.legend(loc="upper right", fontsize="small")
    plt.title("Individual Latency, Load (RPS), and Average Latency Over Time")
    plt.grid()
    plt.tight_layout()
    output_plot = f"{input_dir}/vegeta-latency.png"
    plt.savefig(output_plot, dpi=300)
    print("*" * 30)
    print(f"**** Saving plot to {output_plot}")
    print("*" * 30)
    print(f"10: {int(time.time() - ts)}")


if __name__ == "__main__":
    # input_dir = "/users/gangmuk/projects/client/gangmuk-test/exp-checkoutcart/bg30/SLATE-with-jumping-global-13/"
    input_dir = sys.argv[1]
    
    if not os.path.isdir(input_dir):
        print(f"Error: {input_dir} is not a valid directory.")
        sys.exit(1)

    ts = time.time()
    convert_binaries_to_csv_parallel(input_dir)
    print(f"parallel encode_binary_to_csv: {int(time.time() - ts)}")

    ts = time.time()
    merged_df = load_and_merge_csvs_parallel(input_dir)
    print(f"parellel csv_to_df and concat: {int(time.time() - ts)}")
    
    ts = time.time()
    plot_latency_and_load(merged_df, input_dir)
    print(f"plot_latency_and_load: {int(time.time() - ts)}")