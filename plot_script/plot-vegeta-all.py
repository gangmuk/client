#!/usr/bin/env python3

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
import csv
import re
import numpy as np

color = {"west": "violet", "central": "green", "east": "brown", "south": "orange"}


def parse_latency_file(file_path):
    metrics = {}
    try:
        with open(file_path, "r") as file:
            for line in file:
                if line.startswith("Requests"):
                    parts = line.split()
                    if len(parts) >= 6:
                        metrics["RequestsTotal"] = int(parts[-3].strip(","))
                        metrics["RequestsRate"] = float(parts[-2].strip(","))
                        metrics["RequestsThroughput"] = float(parts[-1])
                elif line.startswith("Latencies"):
                    parts = line.split()
                    if len(parts) > 2:
                        latencies = parts[-7:]  # Extract the last 7 items
                        metrics["LatencyMin"] = convert_to_ms(latencies[0])
                        metrics["LatencyMean"] = convert_to_ms(latencies[1])
                        metrics["Latency50"] = convert_to_ms(latencies[2])
                        metrics["Latency90"] = convert_to_ms(latencies[3])
                        metrics["Latency95"] = convert_to_ms(latencies[4])
                        metrics["Latency99"] = convert_to_ms(latencies[5])
                        metrics["LatencyMax"] = convert_to_ms(latencies[6])
                elif line.startswith("Success"):
                    success_match = re.search(r"Success\s+\[ratio\]\s+([\d.]+)%", line)
                    if success_match:
                        metrics["SuccessRatio"] = float(success_match.group(1))
        return metrics
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None

def convert_to_ms(value):
    value = value.strip(",")
    
    # Handle microseconds (µ or µs)
    if 'µ' in value:
        # Extract the numeric part before the symbol
        numeric_part = value.split('µ')[0].strip()
        return float(numeric_part) / 1000  # Convert microseconds to milliseconds
    
    # Handle milliseconds
    elif value.endswith("ms"):
        return float(value.replace("ms", ""))
    
    # Handle seconds
    elif value.endswith("s"):
        return float(value.replace("s", "")) * 1000
    
    else:
        raise ValueError(f"Unexpected latency value: {value}")
def parse_vegeta_stats_file(directory, failure_latency=10000.0):
    total_success = 0
    total_failure = 0
    total_throughput = 0.0
    total_requests_sum = 0
    weighted_success_sum = 0
    weighted_latency_sum = 0.0
    weighted_latency_99_sum = 0.0
    total_latency_including_failures = 0.0
    for file_name in os.listdir(directory):
        if file_name.endswith(".stats.txt"):
            file_path = os.path.join(directory, file_name)
            metrics = parse_latency_file(file_path)
            if metrics:
                total_requests = metrics.get("RequestsTotal", 0)
                success_ratio = metrics.get("SuccessRatio", 0.0)
                throughput = metrics.get("RequestsThroughput", 0.0)
                latency_mean = metrics.get("LatencyMean", 0.0)
                latency_99 = metrics.get("Latency99", 0.0)
                successes = int((success_ratio / 100) * total_requests)
                failures = total_requests - successes
                total_success += successes
                total_failure += failures
                total_throughput += throughput
                total_requests_sum += total_requests
                weighted_success_sum += (success_ratio / 100) * total_requests
                weighted_latency_sum += latency_mean * successes
                weighted_latency_99_sum += latency_99 * successes
                total_latency_including_failures += (latency_mean * successes) + (failure_latency * failures)
    success_ratio = (weighted_success_sum / total_requests_sum) * 100
    average_latency_mean = (weighted_latency_sum / total_requests_sum)
    average_latency_99 = (weighted_latency_99_sum / total_requests_sum)
    average_latency_considering_failure = (total_latency_including_failures / total_requests_sum)
    return total_success, total_failure, total_throughput, success_ratio, average_latency_mean, average_latency_99, average_latency_considering_failure

def process_subdirectories(base_dir):
    results = []
    for sub_dir in [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]:
        total_success, total_failure, total_throughput, weighted_success_ratio, avg_latency, avg_latency_99, average_latency_considering_failure = parse_vegeta_stats_file(sub_dir)
            # "Weighted Success Ratio (%)": weighted_success_ratio,
            # "Average Latency 2 (ms)": avg_latency_2
        results.append({
            "Directory": os.path.basename(sub_dir),
            "# Total Requests": total_success + total_failure,
            "# Success": total_success,
            "# Failure": total_failure,
            'Success ratio(%)': total_success / (total_success + total_failure) * 100,
            "Tput (RPS)": int(total_throughput),
            "Avg (ms)": int(avg_latency),
            # "Avg Lat (ms) considering failures": int(average_latency_considering_failure),
            "P99 (ms)": int(avg_latency_99),
        })
    return results

def encode_binary_to_csv(input_bin, output_csv):
    if os.path.exists(output_csv):
        print(f"CSV file {output_csv} already exists. Skipping conversion.")
        return
    print(f"Converting {input_bin} to {output_csv}...")
    try:
        with open(output_csv, "w") as outfile:
            subprocess.run(["/users/gangmuk/projects/client/vegeta", "encode", "--to", "csv", input_bin], stdout=outfile, check=True)
        print(f"Created CSV file: {output_csv}")
        df = pd.read_csv(output_csv, header=None)
        if df.shape[1] == 12:
            df = df.iloc[:, :-1]  # Drop the last column
        df.to_csv(output_csv, index=False, header=False)
        print(f"Processed CSV file with conditional column removal: {output_csv}")
        os.remove(input_bin)
        print(f"Deleted original vegeta binary file: {input_bin}")
    except subprocess.CalledProcessError:
        print(f"Error converting {input_bin} to CSV. Ensure Vegeta is installed.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

def convert_binaries_to_csv_parallel(input_dir):
    binary_files = [
        (os.path.join(input_dir, bin_file), f"{input_dir}/{bin_file}.csv")
        for bin_file in os.listdir(input_dir) if bin_file.endswith(".results.bin")
    ]
    # print(f"In {input_dir}, Found {len(binary_files)} binary files to process.")
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
        print(f"An unexpected error occurred: {e}")
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
    # print(f"input_dir: {input_dir}")
    for csv_file in [f for f in os.listdir(input_dir) if f.endswith(".results.bin.csv")]:
        # print(f"csv_file: {csv_file}")
        cluster = csv_file.split(".")[-4]
        if cluster not in ["west", "central", "east", "south"]:
            print(f"Error: Cluster name {cluster} is not valid.")
            continue
        csv_files.append((f"{input_dir}/{csv_file}", cluster, usecols, dtypes))
    with ThreadPoolExecutor() as executor:
        dfs = list(executor.map(load_csv_parallel, csv_files))
    merged_df = pd.concat(dfs, ignore_index=True)
    return merged_df


def plot_latency_and_load(merged_df, input_dir, per_cluster=False):
    merged_df["Request Latency (ms)"] = merged_df["Request Latency"] / 1e6
    merged_df["Start Time (s)"] = (merged_df["Timestamp"] - merged_df["Timestamp"].min()) / 1e9
    merged_df["Time (s)"] = merged_df["Start Time (s)"].astype(int)
    merged_df["Request Latency (ms)"] = merged_df["Request Latency"] / 1e6
    merged_df["Start Time (s)"] = (merged_df["Timestamp"] - merged_df["Timestamp"].min()) / 1e9
    merged_df["Time (s)"] = merged_df["Start Time (s)"].astype(int)
    rps_per_second = merged_df.groupby(["Cluster", "Time (s)"]).size()
    merged_df.loc[:, "RPS"] = merged_df["Time (s)"].map(rps_per_second)
    max_average_latency = 0
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    if per_cluster:
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
    ax2.plot(total_avg_latency["Time (s)"], total_avg_latency["AvgLatency"], color="blue", linewidth=1.5, label="Total Average Latency")
    ax1.set_xlabel("Time (s)", fontsize=16)
    ax1.set_xlim(left=0)
    ax1.set_ylabel("Requests Per Second (RPS)", color="blue")
    ax1.tick_params(axis="y", labelcolor="red")
    ax2.set_ylabel("Average Latency (ms)", color="blue")
    ax2.tick_params(axis="y", labelcolor="red")
    ax1.set_ylim(0, 1.1 * total_rps.max())
    # ax2.set_ylim(0, 1.1 * max_average_latency)
    ax2.set_ylim(bottom=0)
    ax1.legend(loc="upper left", fontsize=12)
    ax2.legend(loc="upper right", fontsize=12)
    plt.title("Individual Latency, Load (RPS), and Average Latency Over Time")
    plt.grid()
    plt.tight_layout()
    output_plot = f"{input_dir}/timeseries-avg-latency.pdf"
    plt.savefig(output_plot)
    print(f"**** Saving Time series for avg latency pdf: {output_plot}")
    
    
def plot_latency_and_load_for_all_subdir(merged_df_list, input_dir, warmup_period_seconds):
    """Plot latency and load (RPS) for all clusters from multiple subdirectories on the same plot, with twin y-axes."""
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax2 = ax1.twinx()
    max_average_latency = 0
    total_rps_max = 0
    avg_latencies = {}
    for subdir_name, merged_df in merged_df_list.items():
        merged_df.loc[:, "Request Latency (ms)"] = merged_df["Request Latency"] / 1e6
        merged_df.loc[:, "Start Time (s)"] = (merged_df["Timestamp"] - merged_df["Timestamp"].min()) / 1e9
        merged_df.loc[:, "Time (s)"] = merged_df["Start Time (s)"].astype(int)
        rps_per_second = merged_df.groupby(["Cluster", "Time (s)"]).size()
        merged_df.loc[:, "RPS"] = merged_df["Time (s)"].map(rps_per_second)

        aggregated = merged_df.groupby(["Time (s)"]).agg(
            RPS=("Time (s)", "count"),
            AvgLatency=("Request Latency (ms)", "mean")
        ).reset_index()
        aggregated["p99"] = merged_df.groupby(["Time (s)"])["Request Latency (ms)"].quantile(0.99).values
        
        total_rps_max = max(total_rps_max, aggregated["RPS"].max())
        max_average_latency = max(max_average_latency, aggregated["AvgLatency"].max())
        avg_latencies[subdir_name] = aggregated["AvgLatency"].mean()
        # mk = get_marker_for_subdir_name(subdir_name)
        
        routing_rule_label = subdir_to_label(subdir_name)
        
        # # Plot RPS on the first y-axis
        # ax1.plot(
        #     aggregated["Time (s)"],
        #     aggregated["RPS"],
        #     label=f"RPS-{routing_rule_label}",
        #     linewidth=1,
        #     alpha=0.5,
        # )

        # # Plot average latency on the second y-axis
        # ax2.plot(
        #     aggregated["Time (s)"],
        #     aggregated["AvgLatency"],
        #     label=f"Latency (ms)-{routing_rule_label}",
        #     linewidth=1,
        #     alpha=0.8,
        #     marker='x'
        # )    
        
        # ax2.scatter(aggregated["Time (s)"], aggregated["AvgLatency"], label=f"{subdir_name}", marker=mk, s=20)
        ax2.scatter(aggregated["Time (s)"], aggregated["AvgLatency"], label=f"Avg latency, {routing_rule_label}", alpha=0.5)
        # , facecolors='none', edgecolors=color[routing_rule])
        
        # ax2.scatter(aggregated["Time (s)"], aggregated["p99"], label=f"P99, {routing_rule_label}", alpha=0.5, marker="^", s=20)
    ax1.set_xlabel("Time (s)", fontsize=16)
    # ax1.set_ylabel("Requests Per Second (RPS)", fontsize=16, color="black")
    ax2.set_ylabel("Average Latency (ms)", fontsize=16, color="black")
    ax1.tick_params(axis="y", labelcolor="black")
    ax2.tick_params(axis="y", labelcolor="black")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # legend background non-transparent white. bring the legend to the front
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", ncol=2, fontsize=10, framealpha=1, facecolor='white')
    ax1.set_ylim(bottom=0)
    ax1.set_xlim(left=0)
    ax2.set_ylim(bottom=0)
    plt.axvline(x=warmup_period_seconds, color='black', linestyle='--', label='Warmup Period', alpha=0.7)
    # ax2.set_ylim(0, 1000) # optional for zoom-in
    plt.title("Average lateancy over time", fontsize=20)
    plt.grid()
    plt.tight_layout()
    output_plot = f"{input_dir}/all_clusters_latency_and_load_twins.pdf"
    plt.savefig(output_plot)
    print(f"**** Saving combined plot to {output_plot}")
    plt.close()
    
def get_marker_for_subdir_name(subdir_name):
            if "WATERFALL" in subdir_name:
                mk = "x"
            elif "without" in subdir_name:
                mk = "o"
            elif "SLATE-with-jumping-global" in subdir_name and "continuous" not in subdir_name:
                mk = "D"
            elif "SLATE-with-jumping-global-continuous" in subdir_name:
                mk = "*"
            else:
                print(f"ERROR: Unknown ROUTING_RULE in subdir: {subdir_name}. Use default marker '.'")
                mk = '.'
            return mk

def subdir_to_label(subdir_name):
    if "SLATE-with-jumping-global-with-optimizer-with-continuous-profiling" in subdir_name:
        # label = "w/ Opt, w/ PAPA, w/ CP"
        label = "TEASE"
    elif "SLATE-with-jumping-global-with-optimizer-without-continuous-profiling" in subdir_name:
        # label = "w/ Opt, w/ PAPA"
        label = "TEASE\nw/o CP"
    elif "SLATE-without-jumping-global-with-optimizer-with-continuous-profiling" in subdir_name:
        # label = "w/ Opt, w/o PAPA w/ CP"
        label = "TEASE\nw/o PAPA"
    elif "SLATE-without-jumping-global-with-optimizer-without-continuous-profiling" in subdir_name:
        # label = "w/ Opt, w/o PAPA, w/o CP"
        label = "TEASE\nw/o PAPA"
    elif "SLATE-without-jumping-global-with-optimizer-only-once-without-continuous-profiling" in subdir_name:
        # label = "w/o Opt, w/o PAPA\ninit with optimizer (only once)"
        label = "TEASE\nw/o PAPA"
    elif "SLATE-with-jumping-global-without-optimizer-without-continuous-profiling-init-with-multi-region-routing" in subdir_name:
        # label = "w/o Opt, w/ PAPA, w/o CP\ninit with multi-region routing"
        label = "TEASE\nw/o Opt"
    elif "SLATE-with-jumping-global-without-optimizer-without-continuous-profiling-init-with-optimizer" in subdir_name:
        # label = "w/o Opt, w/ PAPA\ninit with optimizer"
        label = "TEASE\nw/o Opt"
    elif "SLATE-with-jumping-global-without-optimizer-without-continuous-profiling" in subdir_name:
        # label = "w/o Opt, w/ PAPA, w/o CP"
        label = "TEASE\nw/o Opt"
    elif "SLATE-without-jumping-global-without-optimizer-without-continuous-profiling-init-multi-region-routing-only-once" in subdir_name:
        # label = "w/o Opt, w/o PAPA\ninit with multi-region routing (only once)"
        label = "Multi-region routing"
    elif "SLATE-without-jumping-global-with-optimizer-without-continuous-profiling" in subdir_name:
        # label = "w/ Opt, w/o PAPA, w/o CP"
        label = "TEASE w/o PAPA"
    elif "WATERFALL" in subdir_name:
        label = "Waterfall"
    else:
        print(f"Cannot find Routing rule in subdir name: {subdir_name}")
        assert False
    # postfix = subdir_name.split("-")[-1]
    # label += f"\n{postfix}"
    return label

def plot_latency_cdf(merged_df_list, input_dir):
    plt.figure(figsize=(8, 7))
    
    avg_latencies = {}
    p50_latencies = {}
    p90_latencies = {}
    p99_latencies = {}
    p999_latencies = {}
    for subdir_name, merged_df in merged_df_list.items():
        latencies = merged_df["Request Latency (ms)"].sort_values()
        cdf = latencies.rank(method='max').div(len(latencies))
        avg_latencies[subdir_name] = latencies.mean()
        p50_latencies[subdir_name] = latencies.quantile(0.5)
        p90_latencies[subdir_name] = latencies.quantile(0.9)
        p99_latencies[subdir_name] = latencies.quantile(0.99)
        p999_latencies[subdir_name] = latencies.quantile(0.999)
        plt.plot(latencies, cdf, label=f"{subdir_to_label(subdir_name)}", linewidth=2)
    
    # def print_latency_percentile(subdir_name):
    #     print(f"{subdir_name} avg: {avg_latencies[subdir_name]:.2f} ms")
    #     print(f"{subdir_name} p50: {p50_latencies[subdir_name]:.2f} ms")
    #     print(f"{subdir_name} p90: {p90_latencies[subdir_name]:.2f} ms")
    #     print(f"{subdir_name} p99: {p99_latencies[subdir_name]:.2f} ms")
    #     print(f"{subdir_name} p999: {p999_latencies[subdir_name]:.2f} ms")
    # for subdir_name, avg_latency in avg_latencies.items():
    #     print_latency_percentile(subdir_name)
    
    plt.xlabel("Latency (ms)", fontsize=16)
    plt.ylabel("CDF", fontsize=16)
    plt.title("CDF of Average Latency", fontsize=20)
    plt.grid()
    plt.legend(loc="lower right", fontsize=14)
    plt.xlim(left=0)
    plt.ylim(bottom=0, top=1.01)
    plt.tight_layout()
    
    output_pdf = f"{input_dir}/latency_cdf.pdf"
    plt.savefig(output_pdf, format='pdf')
    plt.close()
    
    print(f"**** Saving CDF plot to {output_pdf}")


def plot_latency_bars(merge_df_list, input_dir, warmup_period_seconds):
    avg_latencies = {}
    p50_latencies = {}
    p90_latencies = {}
    p99_latencies = {}
    p999_latencies = {}
    max_latencies = {}
    
    # Create a mapping of subdirectory names to their shortened labels
    subdir_labels = {}
    for subdir_name in merge_df_list.keys():
        subdir_labels[subdir_name] = subdir_to_label(subdir_name)
    
    # # Define the desired order for the labels
    # label_order = [
    #     "Waterfall",
    #     "w/o Opt, w/o PAPA\ninit with multi-region routing (only once)",
    #     "w/o Opt, w/ PAPA",
    #     "w/ Opt, w/o PAPA",
    #     "w/ Opt, w/ PAPA, w/o CP",
    #     "w/ Opt, w/ PAPA, w/ CP",
    # ]
    label_order = [
        "Waterfall",
        "TEASE\nw/o Opt",
        "TEASE\nw/o PAPA",
        "TEASE",
    ]
    
    # Process data for all subdirectories
    for subdir_name, merged_df in merge_df_list.items():
        start_time = merged_df["Timestamp"].min()
        cutoff_time = start_time + (warmup_period_seconds * 1_000_000_000)
        filtered_df = merged_df[merged_df["Timestamp"] >= cutoff_time].copy()  # Add .copy() here
        if filtered_df.empty:
            print(f"Error: No data found in {subdir_name} after warmup period.")
            assert False
        # Use .loc to modify the DataFrame
        if filtered_df["Request Latency"].mean() > 1_000_000:
            filtered_df.loc[:, "Request Latency (ms)"] = filtered_df["Request Latency"] / 1_000_000
        else:
            filtered_df.loc[:, "Request Latency (ms)"] = filtered_df["Request Latency"]
            
        latencies = filtered_df["Request Latency (ms)"].sort_values()
        cdf = latencies.rank(method='max').div(len(latencies))
        avg_latencies[subdir_name] = latencies.mean()
        p50_latencies[subdir_name] = latencies.quantile(0.5)
        p90_latencies[subdir_name] = latencies.quantile(0.9)
        p99_latencies[subdir_name] = latencies.quantile(0.99)
        p999_latencies[subdir_name] = latencies.quantile(0.999)
        max_latencies[subdir_name] = latencies.max()
    
    # Organize the data in the desired order
    ordered_subdirs = []
    ordered_labels = []
    
    # First, add subdirectories in the specified order
    for desired_label in label_order:
        found = False
        for subdir_name, label in subdir_labels.items():
            if label == desired_label:
                ordered_subdirs.append(subdir_name)
                ordered_labels.append(label)
                found = True
                break
        if not found:
            print(f"Warning: No subdirectory found for desired label: {desired_label}")
    
    # Then add any remaining subdirectories not in the specified order
    for subdir_name, label in subdir_labels.items():
        if subdir_name not in ordered_subdirs:
            print(f"Warning: Subdirectory {subdir_name} with label {label} not in specified order")
            ordered_subdirs.append(subdir_name)
            ordered_labels.append(label)
    
    # Create bar chart data with the ordered subdirectories
    bar_data = {
        " ": ordered_labels,
        "p99 Latency": [p99_latencies[subdir] for subdir in ordered_subdirs],
        "p90 Latency": [p90_latencies[subdir] for subdir in ordered_subdirs],
        "Avg Latency": [avg_latencies[subdir] for subdir in ordered_subdirs],
        # "p99.9 Latency": [p999_latencies[subdir] for subdir in ordered_subdirs],
    }
    
    bar_df = pd.DataFrame(bar_data)
    bar_df.set_index(" ", inplace=True)
    
    ax = bar_df.plot(kind="bar", figsize=(10, 3), alpha=0.7, width=0.7)
    # text on each bar
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f', fontsize=12, rotation=45)

    # plt.ylim(top=bar_df.max().max() * 1.15)
    plt.ylim(top=1300)

    ## Omit the title to save space
    # title = "Latency Statistics by Routing rule"
    # if 'no-latency-injection' in input_dir:
    #     title += " (no fault injected)"
    # elif 'latency-injection' in input_dir:
    #     title += " (fault injected)"
    # plt.title(title, fontsize=20, pad=30)


    plt.ylabel("Latency (ms)", fontsize=18)
    plt.xticks(rotation=0, fontsize=18)
    plt.yticks(fontsize=18)
    # plt.subplots_adjust(bottom=0.2, wspace=0.1)
    plt.grid(axis='y')
    plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize=16, ncol=1)
    output_pdf = f"{input_dir}/latency_bars.pdf"
    plt.tight_layout()
    plt.savefig(output_pdf, format='pdf', bbox_inches='tight', pad_inches=0.03)
    plt.close()
    print(f"**** Saving bar plot to {output_pdf}")
    
    
if __name__ == "__main__":
    input_dir = sys.argv[1]
    if not os.path.isdir(input_dir):
        print(f"Error: {input_dir} is not a valid directory.")
        sys.exit(1)
    
    merged_df_list = {}
    high_latency_counts = {}
    slo_satisfaction_counts = {}
    # print(f"input_dir: {input_dir}")
    for subdir in os.listdir(input_dir):
        subdir_path = os.path.join(input_dir, subdir)
        # print(f"subdir_path: {subdir_path}")
        if os.path.isdir(subdir_path):
            convert_binaries_to_csv_parallel(subdir_path)
            merged_df = load_and_merge_csvs_parallel(subdir_path)
            plot_latency_and_load(merged_df, subdir_path, per_cluster=False)

            subdir = subdir_path.split("/")[-1]
            merged_df_list[subdir] = merged_df

            high_latency = 5000 # 5 seconds
            high_latency_count = merged_df[merged_df["Request Latency"] > high_latency * 1e6].shape[0]  # Convert to nanoseconds
            high_latency_counts[subdir] = high_latency_count

            slo = 2000 # 2 seconds
            slo_satisfaction_count = merged_df[merged_df["Request Latency"] < slo * 1e6].shape[0]  # Convert to nanoseconds
            slo_satisfaction_counts[subdir] = slo_satisfaction_count
    
    
    warmup_period_seconds = 30
    if "no-latency-injection" in input_dir:
        warmup_period_seconds = 40
    elif "latency-injection" in input_dir:
        warmup_period_seconds = 10
    plot_latency_cdf(merged_df_list, input_dir)
    plot_latency_bars(merged_df_list, input_dir, warmup_period_seconds)
    plot_latency_and_load_for_all_subdir(merged_df_list, input_dir, warmup_period_seconds)
    
#    print()
#    print(f"<High Latency Request Counts (> {high_latency/1e3} seconds)>")
#    for subdir, count in high_latency_counts.items():
#        print(f"- {subdir}: {count}, ratio: {count / merged_df_list[subdir].shape[0] * 100:.2f}%")
#    print()
#
#    print(f"<SLO Satisfaction Counts (< {slo/1e3} seconds)>")
#    for subdir, count in slo_satisfaction_counts.items():
#        print(f"- {subdir}: {count}, ratio: {count / merged_df_list[subdir].shape[0] * 100:.2f}%")
#    print()

    results = process_subdirectories(input_dir)
    df = pd.DataFrame(results)
    if not df.empty:
        output_file = f"{input_dir}/subdirectory_statistics_summary.csv"
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        print(df)
    else:
        print("No data found in subdirectories.")
