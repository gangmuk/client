#!/usr/bin/env python3

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from matplotlib.ticker import MultipleLocator

color = {"west": "violet", "central": "green", "east": "brown", "south": "orange"}


#def encode_binary_to_csv(input_bin, output_csv):
#    # if os.path.exists(output_csv):
#    #     print(f"CSV file {output_csv} already exists. Skipping conversion.")
#    #     return
#    print(f"Converting {input_bin} to {output_csv}...")
#    try:
#        # Step 1: Encode binary to CSV using Vegeta
#        with open(output_csv, "w") as outfile:
#            subprocess.run(["vegeta", "encode", "--to", "csv", input_bin], stdout=outfile, check=True)
#        print(f"Created CSV file: {output_csv}")
#        
#        # Step 2: Remove the last column from the CSV
#        temp_csv = f"{output_csv}.tmp"
#        with open(output_csv, "r") as infile, open(temp_csv, "w", newline="") as outfile:
#            reader = csv.reader(infile)
#            writer = csv.writer(outfile)
#            for row in reader:
#                # Exclude the last column
#                writer.writerow(row[:-1])
#        
#        # Replace the original CSV with the modified version
#        os.replace(temp_csv, output_csv)
#        print(f"Removed the last column and updated CSV file: {output_csv}")
#
#        # Step 3: Delete the original binary file
#        subprocess.run(["rm", input_bin])
#        print(f"Deleted original vegeta binary file: {input_bin}")
#    except subprocess.CalledProcessError:
#        print(f"Error converting {input_bin} to CSV. Ensure Vegeta is installed.")
#        sys.exit(1)
#    except Exception as e:
#        print(f"Unexpected error: {e}")
#        sys.exit(1)
#


def encode_binary_to_csv(input_bin, output_csv):
    print(f"Converting {input_bin} to {output_csv}...")
    try:
        # Step 1: Encode binary to CSV using Vegeta
        with open(output_csv, "w") as outfile:
            subprocess.run(["/users/gangmuk/projects/client/vegeta", "encode", "--to", "csv", input_bin], stdout=outfile, check=True)
        print(f"Created CSV file: {output_csv}")

        # Step 2: Remove the last column only if the number of columns is twelve
        # Read the CSV file with pandas
        df = pd.read_csv(output_csv, header=None)

        # Check and remove the last column if there are exactly 12 columns
        if df.shape[1] == 12:
            df = df.iloc[:, :-1]  # Drop the last column

        # Save the updated DataFrame back to the same file
        df.to_csv(output_csv, index=False, header=False)
        print(f"Processed CSV file with conditional column removal: {output_csv}")

        # Step 3: Delete the original binary file
        os.remove(input_bin)
        print(f"Deleted original vegeta binary file: {input_bin}")
    except Exception as e:
        print(f"Unexpected error: {e}, {input_bin}")
        sys.exit(1)



def convert_binaries_to_csv_parallel(input_dir):
    binary_files = [
        (os.path.join(input_dir, bin_file), f"{input_dir}/{bin_file}.csv")
        for bin_file in os.listdir(input_dir) if bin_file.endswith(".results.bin")
    ]
    print(f"Found {len(binary_files)} binary files to process.")
    with ThreadPoolExecutor() as executor:
        executor.map(lambda args: encode_binary_to_csv(*args), binary_files)


def csv_to_df(csv_file, cluster, usecols=None, dtypes=None):
    column_names = [
        "Timestamp", "HTTP Status", "Request Latency", "Bytes Out", "Bytes In", 
        "Error", "Base64 Body", "Attack Name", "Sequence Number", "Method", "URL"
    ]
    try:
        df = pd.read_csv(csv_file, header=None, names=column_names, usecols=usecols, dtype=dtypes)
    except (FileNotFoundError, pd.errors.EmptyDataError) as e:
        print(f"Error reading CSV file {csv_file}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}, {csv_file}")
        return None
    df["Cluster"] = cluster
    return df


def load_csv_parallel(args):
    csv_file, cluster, usecols, dtypes = args
    return csv_to_df(csv_file, cluster, usecols=usecols, dtypes=dtypes)

def load_and_merge_csvs_parallel(input_dir):
    usecols = ["Timestamp", "HTTP Status", "Request Latency"]
    dtypes = {
        "Timestamp": "int64",
        "HTTP Status": "category",
        "Request Latency": "float64",
    }
    csv_files = []
    print(f"input_dir: {input_dir}")
    for csv_file in [f for f in os.listdir(input_dir) if f.endswith(".results.bin.csv")]:
        cluster = csv_file.split(".")[-4]
        if cluster not in ["west", "central", "east", "south"]:
            print(f"Error: Cluster name {cluster} is not valid.")
            continue
        csv_files.append((f"{input_dir}/{csv_file}", cluster, usecols, dtypes))
    with ThreadPoolExecutor() as executor:
        dfs = list(executor.map(load_csv_parallel, csv_files))
    merged_df = pd.concat(dfs, ignore_index=True)
    return merged_df

def plot_per_cluster_latency_and_load(merged_df, cluster, input_dir):
    df_cluster = merged_df[merged_df["Cluster"] == cluster]
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    print(f"Cluster: {cluster}")
    aggregated = df_cluster.groupby(["Time (s)"]).agg(RPS=("Time (s)", "count"), AvgLatency=("Request Latency (ms)", "mean")).reset_index()
    ax1.plot(aggregated["Time (s)"], aggregated["RPS"], label=f"RPS-{cluster}", color=color[cluster], linewidth=2, linestyle="--")
    
    # ax2.plot(aggregated["Time (s)"], aggregated["AvgLatency"], label=f"Latency-{cluster}",color=color[cluster], linewidth=1.5, marker="^", markersize=0)
    ax2.scatter(aggregated["Time (s)"], aggregated["AvgLatency"], label=f"Latency-{cluster}", color=color[cluster], marker="x", s=10, alpha=0.5)
    
    avg = df_cluster["Request Latency (ms)"].mean()
    p50 = df_cluster["Request Latency (ms)"].quantile(0.50)
    p75 = df_cluster["Request Latency (ms)"].quantile(0.75)
    p90 = df_cluster["Request Latency (ms)"].quantile(0.90)
    p95 = df_cluster["Request Latency (ms)"].quantile(0.95)
    p99 = df_cluster["Request Latency (ms)"].quantile(0.99)
    print(f"{cluster} avg lat: {avg:.2f} ms")
    print(f"{cluster} 99th percentile latency: {p99:.2f} ms")
    print(f"{cluster} 95th percentile latency: {p95:.2f} ms")
    print(f"{cluster} 90th percentile latency: {p90:.2f} ms")
    print(f"{cluster} 75th percentile latency: {p75:.2f} ms")
    print(f"{cluster} 50th percentile latency: {p50:.2f} ms")
    
    
    ax1.set_xlabel("Time (s)", fontsize=16)
    ax1.set_xlim(left=0)
    ax1.set_ylabel("Requests Per Second (RPS)", color="blue")
    ax1.tick_params(axis="y", labelcolor="red")
    ax2.set_ylabel("Average Latency (ms)", color="blue")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.set_ylim(bottom=0)
    ax1.legend(loc="upper left", fontsize=12)
    ax2.legend(loc="upper right", fontsize=12)
    plt.title("Individual Latency, Load (RPS), and Average Latency Over Time")
    plt.grid()
    plt.tight_layout()
    output_plot = f"{input_dir}/vegeta-latency-{cluster}.pdf"
    plt.savefig(output_plot)
    print("*" * 30)
    print(f"Saving plot to {output_plot}")
    print("*" * 30)




def plot_latency_and_load(merged_df, input_dir):    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    total_rps = merged_df.groupby(["Time (s)"]).size()
    total_avg_latency = merged_df.groupby(["Time (s)"]).agg(AvgLatency=("Request Latency (ms)", "mean")).reset_index()
    ax1.plot(total_rps, color="red", linewidth=1.5, label="Total RPS")
    ax2.plot(total_avg_latency["Time (s)"], total_avg_latency["AvgLatency"], color="blue", linewidth=1.5, label="Total Average Latency")
    
    ax1.set_xlabel("Time (s)", fontsize=16)
    ax1.set_xlim(left=0)
    ax1.set_ylabel("Requests Per Second (RPS)", color="blue")
    ax1.tick_params(axis="y", labelcolor="red")
    ax2.set_ylabel("Average Latency (ms)", color="blue")
    ax2.tick_params(axis="y", labelcolor="red")
    ax1.set_ylim(0, 1.1 * total_rps.max())
    ax2.set_ylim(bottom=0)
    ax1.legend(loc="upper left", fontsize=12)
    ax2.legend(loc="upper right", fontsize=12)
    plt.title("Individual Latency, Load (RPS), and Average Latency Over Time")
    plt.grid()
    plt.tight_layout()
    output_plot = f"{input_dir}/vegeta-latency.pdf"
    plt.savefig(output_plot)
    print("*" * 30)
    print(f"Saving plot to {output_plot}")
    print("*" * 30)
    
def plot_latency_and_load_for_all_subdir(merged_df_list, input_dir):
    """Plot latency and load (RPS) for all clusters from multiple subdirectories on the same plot, with twin y-axes."""
    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax2 = ax1.twinx()
    max_average_latency = 0
    total_rps_max = 0
    avg_latencies = {}
    for subdir_name, merged_df in merged_df_list.items():
        # west only
        # merged_df = merged_df.loc[merged_df["Cluster"] == "west"]
        
        df_cluster = dict()
        for cluster in merged_df["Cluster"].unique():
            df_cluster[cluster] = merged_df[merged_df["Cluster"] == cluster]
        
        for cluster in df_cluster:
            aggregated = df_cluster[cluster].groupby(["Time (s)"]).agg(RPS=("Time (s)", "count"), AvgLatency=("Request Latency (ms)", "mean")).reset_index()
            # ax2.plot(aggregated["Time (s)"], aggregated["AvgLatency"], label=f"Latency (ms)-{cluster}-{subdir_name}", color=color[cluster], linewidth=1.5, marker="^", markersize=0)
            ax2.scatter(aggregated["Time (s)"], aggregated["AvgLatency"], label=f"Latency-{cluster}", color=color[cluster], marker="x", s=10, alpha=0.5)
            max_average_latency = max(max_average_latency, aggregated["AvgLatency"].max())
            
        for cluster in df_cluster:
            aggregated = df_cluster[cluster].groupby(["Time (s)"]).agg(RPS=("Time (s)", "count"), AvgLatency=("Request Latency (ms)", "mean")).reset_index()
            ax1.plot(aggregated["Time (s)"], aggregated["RPS"], label=f"RPS-{cluster}", color=color[cluster], linewidth=1.5, linestyle="-")
            
        merged_df["Request Latency (ms)"] = merged_df["Request Latency"] / 1e6
        merged_df["Start Time (s)"] = (merged_df["Timestamp"] - merged_df["Timestamp"].min()) / 1e9
        merged_df["Time (s)"] = merged_df["Start Time (s)"].astype(int)
        rps_per_second = merged_df.groupby(["Cluster", "Time (s)"]).size()
        merged_df["RPS"] = merged_df["Time (s)"].map(rps_per_second)

        aggregated = merged_df.groupby(["Time (s)"]).agg(
            RPS=("Time (s)", "count"),
            AvgLatency=("Request Latency (ms)", "mean")
        ).reset_index()
        

        total_rps_max = max(total_rps_max, aggregated["RPS"].max())
        max_average_latency = max(max_average_latency, aggregated["AvgLatency"].max())
        avg_latencies[subdir_name] = aggregated["AvgLatency"].mean()

        # 
        # ax1.plot(
        #     aggregated["Time (s)"],
        #     aggregated["RPS"],
        #     label=f"Average RPS",
        #     linewidth=1,
        #     linestyle="--"
        # )

        # # Plot average latency on the second y-axis
        # ax2.plot(
        #     aggregated["Time (s)"],
        #     aggregated["AvgLatency"],
        #     label=f"Average Latency",
        #     linewidth=1,
        #     alpha=0.8,
        # )

    # Add labels and legends
    ax1.set_xlabel("Time (s)", fontsize=16)
    ax1.set_ylabel("Requests Per Second (RPS)", fontsize=16, color="blue")
    ax2.set_ylabel("Average Latency (ms)", fontsize=16, color="orange")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax2.tick_params(axis="y", labelcolor="orange")

    # Add legends for both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=12, ncol=2)
    
    ax1.set_ylim(bottom=0)
    ax1.set_xlim(left=0)
    ax2.set_ylim(bottom=0)
    ax2.set_ylim(top=1000)

    # Add title and grid
    # ax2.yaxis.set_major_locator(MultipleLocator(100))
    ax2.grid(axis="y", which="major", linestyle="-", alpha=0.5)
    plt.title(f"{input_dir.split('/')[-2:]}", fontsize=20)
    # plt.grid()
    plt.tight_layout()

    # Save the plot
    output_plot = f"{input_dir}/all_clusters_latency_and_load_twins.pdf"
    plt.savefig(output_plot)
    print("*" * 30)
    print(f"Saving combined plot to {output_plot}")
    print("*" * 30)
    plt.close()



def plot_latency_cdf(merged_df_list, input_dir):
    plt.figure(figsize=(8, 7))
    avg_latencies = {}
    for subdir_name, merged_df in merged_df_list.items():
        latencies = merged_df["Request Latency (ms)"].sort_values()
        cdf = latencies.rank(method='max').div(len(latencies))
        avg_latencies[subdir_name] = latencies.mean()
        plt.plot(latencies, cdf, label=f"{subdir_name}", linewidth=2)
    
    for subdir_name, avg_latency in avg_latencies.items():
        print(f"{subdir_name} avg lat: {avg_latency:.2f} ms")
        print(f"{subdir_name} 99th percentile latency: {merged_df['Request Latency (ms)'].quantile(0.99):.2f} ms")
        print(f"{subdir_name} 95th percentile latency: {merged_df['Request Latency (ms)'].quantile(0.95):.2f} ms")
        print(f"{subdir_name} 90th percentile latency: {merged_df['Request Latency (ms)'].quantile(0.90):.2f} ms")
        print(f"{subdir_name} 75th percentile latency: {merged_df['Request Latency (ms)'].quantile(0.75):.2f} ms")
        print(f"{subdir_name} 50th percentile latency: {merged_df['Request Latency (ms)'].quantile(0.50):.2f} ms")
        
        # plt.text(avg_latency, f"{subdir_name} avg lat: {avg_latency:.2f} ms", fontsize=12)
    plt.xlabel("Latency (ms)", fontsize=16)
    plt.ylabel("CDF", fontsize=16)
    plt.title("CDF of Average Latency", fontsize=20)
    plt.grid()
    plt.legend(loc="lower right", fontsize=14)
    plt.xlim(left=0)
    # plt.xlim(right=1000)
    # print(f"cropped x-axis to 0-1000 ms")
    plt.ylim(bottom=0, top=1.01)
    plt.tight_layout()
    
    output_plot = f"{input_dir}/latency_cdf.pdf"
    plt.savefig(output_plot)
    plt.close()
    
    print("*" * 30)
    print(f"Saving CDF plot to {output_plot}")
    print("*" * 30)

if __name__ == "__main__":
    input_dir = sys.argv[1]
    if not os.path.isdir(input_dir):
        print(f"Error: {input_dir} is not a valid directory.")
        sys.exit(1)

    convert_binaries_to_csv_parallel(input_dir)
    merged_df_list = {}
    if os.path.isdir(input_dir):
        merged_df = load_and_merge_csvs_parallel(input_dir)
        merged_df.loc[:, "Request Latency (ms)"] = merged_df["Request Latency"] / 1e6
        merged_df.loc[:, "Start Time (s)"] = (merged_df["Timestamp"] - merged_df["Timestamp"].min()) / 1e9
        merged_df.loc[:, "Time (s)"] = merged_df["Start Time (s)"].astype(int)
        merged_df.groupby(["Cluster", "Time (s)"]).size()
        rps_per_second = merged_df.groupby(["Cluster", "Time (s)"]).size()
        merged_df["RPS"] = merged_df["Time (s)"].map(rps_per_second) 
        # for cluster in merged_df["Cluster"].unique():
        #     plot_per_cluster_latency_and_load(merged_df, cluster, input_dir)
        # plot_latency_and_load(merged_df, input_dir)
        temp = input_dir.split("/")[-1]
        merged_df_list[temp] = merged_df
    
    
    # plot_all_latency_and_load(merged_df, subdir_path, per_cluster=False)
    plot_latency_cdf(merged_df_list, input_dir)
    plot_latency_and_load_for_all_subdir(merged_df_list, input_dir)
