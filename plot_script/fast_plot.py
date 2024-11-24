#!/usr/bin/python

import matplotlib.pyplot as plt
import csv
import sys
import os
from collections import Counter
import numpy as np
import pandas as pd
import argparse

color_dict = {"west": "blue", "east": "red", "central": "orange", "south": "green"}


# Helper function to read CSV files and return x, y values
# def xy_from_csv(filename, data_dir):
#     x = []
#     y = []
#     path = os.path.join(data_dir, filename)
#     if os.path.exists(path):
#         with open(path, 'r') as csvfile:
#             plots = csv.reader(csvfile, delimiter=',')
#             for row in plots:
#                 x.append(float(row[0]))
#                 y.append(float(row[1]))
#     else:
#         print(f"File {path} does not exist.")
#     return x, y
def xy_from_csv(filename, data_dir):
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        print(f"File {path} does not exist.")
        return [], []

    with open(path, 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        x, y = zip(*((float(row[0]), float(row[1])) for row in plots))

    return list(x), list(y)


# Adjust the x-axis time values to start from 0 and convert to seconds
def adjust(xs, xstart):
    return [(x - xstart) / 1e9 for x in xs]

def clip_data(args, x, y, total_duration):
    clipped_x = [(args.clip_front <= t <= (total_duration - args.clip_end)) for t in x]
    return np.array(x)[clipped_x], np.array(y)[clipped_x]

def preprocess_data(args, x_, y_, xstart, total_duration):
    adjusted_x = adjust(x_, xstart)
    x_, y_ = clip_data(args, adjusted_x, y_, total_duration)
    return x_, y_

# Main plotting function
total_load = {}
def plot_data(cluster_data_dir, cluster, ax1, ax2, plt):
    
    # Stats dump interval so we can calculate rates.
    stats_dump_interval = 1.0
    
    
    def load_all_csvs(data_dir, filenames):
        data = {}
        for filename in filenames:
            path = os.path.join(data_dir, filename)
            if not os.path.isfile(path):
                print(f"File {path} does not exist.")
                data[filename] = ([], [])
                continue

            try:
                with open(path, 'r') as csvfile:
                    reader = csv.reader(csvfile, delimiter=',')
                    # Use generator comprehension to process the file in one pass
                    x, y = zip(*((float(row[0]), float(row[1])) for row in reader))
                    data[filename] = (list(x), list(y))  # Convert back to lists if needed
            except (ValueError, StopIteration) as e:
                print(f"Error reading {path}: {e}")
                data[filename] = ([], [])
        return data

    # Specify the filenames and directory
    filenames = [
        "client.rps.0.csv",
        "client.req.count.0.csv",
        "client.req.retry.count.0.csv",
        "client.req.latency.0.csv",
        "client.req.success_hist.0.csv",
        "client.req.success.count.0.csv",
        "client.req.failure.count.0.csv",
        "client.req.timeout.0.csv",
        "client.req.timeout_origin.0.csv"
    ]

    # Call the function once to load all files
    csv_data = load_all_csvs(cluster_data_dir, filenames)

    # Access the data
    in_rq_rate_x, in_rq_rate_y = csv_data["client.rps.0.csv"]
    out_rq_rate_x, out_rq_rate_y = csv_data["client.req.count.0.csv"]
    retry_rate_x, retry_rate_y = csv_data["client.req.retry.count.0.csv"]
    rq_latency_x, rq_latency_y = csv_data["client.req.latency.0.csv"]
    success_x, _ = csv_data["client.req.success_hist.0.csv"]
    goodput_x, goodput_y = csv_data["client.req.success.count.0.csv"]
    failure_x, failure_y = csv_data["client.req.failure.count.0.csv"]
    timeout_x, timeout_y = csv_data["client.req.timeout.0.csv"]
    timeout_origin_x, timeout_origin_y = csv_data["client.req.timeout_origin.0.csv"]
    
    # # Read data from CSV files
    # in_rq_rate_x, in_rq_rate_y = xy_from_csv("client.rps.0.csv", cluster_data_dir)
    # out_rq_rate_x, out_rq_rate_y = xy_from_csv("client.req.count.0.csv", cluster_data_dir)
    # retry_rate_x, retry_rate_y = xy_from_csv("client.req.retry.count.0.csv", cluster_data_dir)
    # rq_latency_x, rq_latency_y = xy_from_csv("client.req.latency.0.csv", cluster_data_dir)
    # success_x, _ = xy_from_csv("client.req.success_hist.0.csv", cluster_data_dir)
    # goodput_x, goodput_y = xy_from_csv("client.req.success.count.0.csv", cluster_data_dir)
    # # goodput_y = [y / stats_dump_interval for y in goodput_y]
    # failure_x, failure_y = xy_from_csv("client.req.failure.count.0.csv", cluster_data_dir)
    # timeout_x, timeout_y = xy_from_csv("client.req.timeout.0.csv", cluster_data_dir)
    # timeout_origin_x, timeout_origin_y = xy_from_csv("client.req.timeout_origin.0.csv", cluster_data_dir)
    
   

    # Determine x-axis limits
    all_x_values = in_rq_rate_x + out_rq_rate_x + rq_latency_x + goodput_x + failure_x + timeout_x + timeout_origin_x
    xstart = min(all_x_values)
    xend = max(all_x_values)
    total_duration = (xend - xstart) / 1e9  # Convert to seconds
    xend = (total_duration - args.clip_end)
    
    # Adjust the x-values
    # adjusted_latency_x = adjust(rq_latency_x, xstart)
    # adjusted_in_rq_rate_x = adjust(in_rq_rate_x, xstart)
    # adjusted_goodput_x = adjust(goodput_x, xstart)
    # adjusted_failure_x = adjust(failure_x, xstart)
    # adjusted_retry_rate_x = adjust(retry_rate_x, xstart)
    # adjusted_timeout_x = adjust(timeout_x, xstart)
    
    ## Filter latency data
    # latency_mask = [(args.clip_front <= t <= (total_duration - args.clip_end)) for t in adjusted_latency_x]
    # adjusted_latency_x = np.array(adjusted_latency_x)[latency_mask]
    # rq_latency_y = np.array(rq_latency_y[latency_mask])
    adjusted_latency_x, rq_latency_y = preprocess_data(args, rq_latency_x, rq_latency_y, xstart, total_duration)

    ## Filter other datasets similarly if they are used in plotting
    # in_rq_rate_mask = [(args.clip_front <= t <= (total_duration - args.clip_end)) for t in adjusted_in_rq_rate_x]
    # adjusted_in_rq_rate_x = np.array(adjusted_in_rq_rate_x)[in_rq_rate_mask]
    # in_rq_rate_y = np.array(in_rq_rate_y)[in_rq_rate_mask]
    adjusted_in_rq_rate_x, in_rq_rate_y = preprocess_data(args, in_rq_rate_x, in_rq_rate_y, xstart, total_duration)

    # goodput_mask = [(args.clip_front <= t <= (total_duration - args.clip_end)) for t in adjusted_goodput_x]
    # adjusted_goodput_x = np.array(adjusted_goodput_x)[goodput_mask]
    # goodput_y = np.array(goodput_y)[goodput_mask]
    adjusted_goodput_x, goodput_y = preprocess_data(args, goodput_x, goodput_y, xstart, total_duration)

    # failure_mask = [(args.clip_front <= t <= (total_duration - args.clip_end)) for t in adjusted_failure_x]
    # adjusted_failure_x = np.array(adjusted_failure_x)[failure_mask]
    # failure_y = np.array(failure_y)[failure_mask]
    adjusted_failure_x, failure_y = preprocess_data(args, failure_x, failure_y, xstart, total_duration)

    # retry_rate_mask = [(args.clip_front <= t <= (total_duration - args.clip_end)) for t in adjusted_retry_rate_x]
    # adjusted_retry_rate_x = np.array(adjusted_retry_rate_x)[retry_rate_mask]
    # retry_rate_y = np.array(retry_rate_y)[retry_rate_mask]
    adjusted_retry_rate_x, retry_rate_y = preprocess_data(args, retry_rate_x, retry_rate_y, xstart, total_duration)
    
    failure_and_success_y = [failure_y[i] + goodput_y[i] for i in range(len(failure_y))]
    failure_and_success_x = adjusted_failure_x
    
    # Plot request latency with binning to improve performance

    latency_trend_plot_scheme = "sample"  # "sample", "average", or "all"
    if latency_trend_plot_scheme == "sample":
        sample_fraction = 0.05
        sample_size = max(int(len(rq_latency_y) * sample_fraction), 1)
        if sample_size < len(rq_latency_y):
            # Use numpy to randomly select indices
            indices = np.random.choice(len(rq_latency_y), size=sample_size, replace=False)
            sampled_adjusted_latency_x = adjusted_latency_x[indices]
            sampled_rq_latency_y = rq_latency_y[indices]
        else:
            sampled_adjusted_latency_x = adjusted_latency_x
            sampled_rq_latency_y = rq_latency_y
        # ax1.scatter(sampled_adjusted_latency_x, sampled_rq_latency_y, label=f"Observed Latency-{cluster} (Sampled {sample_fraction*100}%)", marker='.', alpha=0.1, color=color_dict[cluster])
        bin_width = 0.1  # seconds
        bins = np.arange(args.clip_front, (total_duration - args.clip_end) + bin_width, bin_width)
        bin_indices = np.digitize(adjusted_latency_x, bins)
        binned_latency = [rq_latency_y[bin_indices == i] for i in range(1, len(bins))]
        average_latency = [np.mean(bin) if len(bin) > 0 else np.nan for bin in binned_latency]
        bin_centers = bins[:-1] + bin_width / 2
        ax1.plot(bin_centers, average_latency, label=f"Average Latency-{cluster}", color=color_dict[cluster])
    elif latency_trend_plot_scheme == "all":
        ax1.scatter(adjusted_latency_x, rq_latency_y, label="Observed Latency", marker='.', alpha=0.2, color=color_dict[cluster])
    elif latency_trend_plot_scheme == "average":
        bin_width = 0.1  # seconds
        bins = np.arange(args.clip_front, (total_duration - args.clip_end) + bin_width, bin_width)
        bin_indices = np.digitize(adjusted_latency_x, bins)
        binned_latency = [rq_latency_y[bin_indices == i] for i in range(1, len(bins))]
        average_latency = [np.mean(bin) if len(bin) > 0 else np.nan for bin in binned_latency]
        bin_centers = bins[:-1] + bin_width / 2
        ax1.plot(bin_centers, average_latency, label=f"Average Latency-{cluster}")
    else:
        raise ValueError(f"Invalid latency trend plot scheme: {latency_trend_plot_scheme}")


    # Plot offered load, goodput, failure, and retries
    ax2.plot(adjusted_goodput_x, goodput_y, label=f"Goodput-{cluster}", linestyle='-', color=color_dict[cluster])
    ax2.plot(adjusted_failure_x, failure_y, label=f"Failure-{cluster}", linestyle='--', color='black', alpha=0.5)
    ax2.plot(adjusted_in_rq_rate_x, in_rq_rate_y, label=f"Load-{cluster}", linestyle=':', color=color_dict[cluster], alpha=0.7, marker='o', markersize=3, markerfacecolor='none')
    total_load[cluster] = list(in_rq_rate_y)
    # ax2.plot(adjusted_retry_rate_x, retry_rate_y, color="cyan", label=f"Retries-{cluster}", marker='x', linestyle=linestyle)
    
    # ax2.plot(adjusted_failure_x, failure_and_success_y, label=f"success + failure-{cluster}", linestyle=linestyle)

    # # Update statistics after clipping
    # num_timeout = len([t for t in adjusted_timeout_x if args.clip_front <= t <= (total_duration - args.clip_end)])
    # num_failure = len(adjusted_failure_x)

    # stats_text = (
    #     f"Timeouts: {num_timeout}\n"
    #     f"Failures: {num_failure}"
    # )
    # plt.text(
    #     0.95, 0.05, stats_text,
    #     horizontalalignment='right',
    #     verticalalignment='bottom',
    #     transform=plt.gcf().transFigure,
    #     fontsize=16,
    #     bbox=dict(facecolor='white', alpha=0.5)
    # )
    
    return adjusted_in_rq_rate_x, total_duration

def plot_cdf(cluster_data_dir, cluster, plt):
    # Read latency data along with timestamps
    rq_latency_x, rq_latency_y = xy_from_csv("client.req.latency.0.csv", cluster_data_dir)
    all_x_values = rq_latency_x
    xstart = min(all_x_values)
    xend = max(all_x_values)
    total_duration = (xend - xstart) / 1e9  # Convert to seconds
    adjusted_latency_x = adjust(rq_latency_x, xstart)
    rq_latency_y = np.array(rq_latency_y)
    latency_mask = [(args.clip_front <= t <= (total_duration - args.clip_end)) for t in adjusted_latency_x]
    rq_latency_y = rq_latency_y[latency_mask]
    sorted_y = np.sort(rq_latency_y)
    num_points = 1000  # Adjust based on performance needs
    if len(sorted_y) > num_points:
        indices = np.linspace(0, len(sorted_y) - 1, num_points).astype(int)
        sorted_y = sorted_y[indices]
        p = np.linspace(0, 1, num_points)
    else:
        p = np.arange(1, len(sorted_y) + 1) / len(sorted_y)

    plt.plot(sorted_y, p, label=f"{cluster}", alpha=0.7, marker='o', markerfacecolor='none')

    # Calculate latency statistics
    avg_latency = np.mean(rq_latency_y)
    p50_latency = np.percentile(rq_latency_y, 50)
    p90_latency = np.percentile(rq_latency_y, 90)
    p99_latency = np.percentile(rq_latency_y, 99)
    p999_latency = np.percentile(rq_latency_y, 99.9)
    max_latency = np.max(rq_latency_y)

    stats_text = (
        f"Average: {avg_latency:.2f} ms ({cluster})\n"
        # f"P50: {p50_latency:.2f} ms ({cluster})\n"
        # f"P90: {p90_latency:.2f} ms ({cluster})\n"
        f"P99: {p99_latency:.2f} ms ({cluster})\n"
        f"P99.9: {p999_latency:.2f} ms ({cluster})\n"
        f"Max: {max_latency:.2f} ms ({cluster})\n"
        "---------------------\n"
    )
    
    return stats_text

# Main function to run the script

    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="Path to the data directory", required=True)
    parser.add_argument("--clip_front", type=float, default=0, help="Number of seconds to clip from the start and end")
    parser.add_argument("--clip_end", type=float, default=0, help="Number of seconds to clip from the end")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.data_dir):
        print(f"No data directory provided or found: {args.data_dir}")
        sys.exit(1)
    
    if not os.path.exists(f"{args.data_dir}/client-west") and not os.path.exists(f"{args.data_dir}/client-east"):
        print(f"Data directory {args.data_dir} does not contain client-west or client-east subdirectories.")
        print(f"You need at least one to plot")
        sys.exit(1)
    
    ## Plot timelinen graph
    fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 8)) # This is for plotting all cluster in the same graph
    clusters = ["west", "east", "central", "south"]
    for cluster in clusters:
        if os.path.exists(f"{args.data_dir}/client-{cluster}"):
            adjusted_in_rq_rate_x, total_duration = plot_data(f"{args.data_dir}/client-{cluster}", cluster, ax1, ax2, plt)
    # total_duration = 500
    
    for cluster in total_load:
        print(f"len(total_load[{cluster}]): {len(total_load[cluster])}")
    aggregated_total_load = []
    for i in range(min([len(total_load[cluster]) for cluster in total_load])):
        aggregated_total_load.append(total_load["west"][i] + total_load["east"][i] + total_load["central"][i] + total_load["south"][i])
    if len(adjusted_in_rq_rate_x) != len(aggregated_total_load):
        adjusted_in_rq_rate_x = adjusted_in_rq_rate_x[:min(len(adjusted_in_rq_rate_x), len(aggregated_total_load))]
        aggregated_total_load = aggregated_total_load[:min(len(adjusted_in_rq_rate_x), len(aggregated_total_load))]
    ax2.plot(adjusted_in_rq_rate_x, aggregated_total_load, label=f"Total Load", linestyle='-', color="black")
    
    ax1.set_xlabel('Time (s)', fontsize=22)
    ax1.set_ylabel('Latency (ms)', fontsize=22)
    ax1.tick_params(axis='x', labelsize=20)
    ax1.tick_params(axis='y', labelsize=20)
    ax1.set_xlim([args.clip_front, (total_duration - args.clip_end)])
    ax1.set_ylim(bottom=0)
    # ax1.set_ylim([0, max(rq_latency_y) * 1.05])
    ax1.legend(fontsize=10, loc='upper right')
    
    ax2.set_xlabel('Time (s)', fontsize=22)
    ax2.set_ylabel('Load', fontsize=22)
    ax2.tick_params(axis='x', labelsize=20)
    ax2.tick_params(axis='y', labelsize=20)
    ax2.set_xlim([args.clip_front, (total_duration - args.clip_end)])
    # ax2.set_xlim([0, max(total_load["west"])*1.01])
    ax2.set_ylim(bottom=0)
    ax2.grid(True)
    ax2.legend(fontsize=8, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.20), borderaxespad=0)
    
    plt.tight_layout()
    
    # result_path_pdf = f"{args.data_dir}/new-result.pdf"
    # plt.savefig(result_path_pdf)
    # print(f"Plot latency timeline saved to ./{result_path_pdf}")
    
    result_path_png = f"{args.data_dir}/new-result.png"
    plt.savefig(result_path_png, dpi=300)
    print(f"Plot latency timeline saved to ./{result_path_png}")
    plt.show()
        
    ## Plot CDF
    plt.figure(figsize=(6, 5))
    stats_text = ""
    for cluster in clusters:
        if os.path.exists(f"{args.data_dir}/client-{cluster}"):
            stats_text += plot_cdf(f"{args.data_dir}/client-{cluster}", cluster, plt)
    
    plt.text(
        0.95, 0.05, stats_text,
        horizontalalignment='right',
        verticalalignment='bottom',
        transform=plt.gca().transAxes,
        fontsize=8,
        bbox=dict(facecolor='white', alpha=0.5)
    )
    
    plt.xlabel('Latency (ms)', fontsize=20)
    plt.ylabel('Cumulative Probability', fontsize=20)
    plt.title('CDF of Request Latency', fontsize=20)
    plt.xticks(fontsize=18, rotation=-45)
    plt.yticks(fontsize=18)
    plt.xlim(left=0)
    plt.ylim([0, 1.01])
    plt.legend(fontsize=12, loc='upper left')
    plt.tight_layout()    
    
    # result_path_pdf = f"{args.data_dir}/new-cdf.pdf"
    # plt.savefig(result_path_pdf)
    # print(f"Plot CDF saved to ./{result_path_pdf}")
    
    result_path_png = f"{args.data_dir}/new-cdf.png"
    plt.savefig(result_path_png, dpi=300)
    print(f"Plot CDF saved to ./{result_path_png}")
    
    plt.show()