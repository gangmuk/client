#!/usr/bin/python

import matplotlib.pyplot as plt
import csv
import sys
import os
from collections import Counter
import numpy as np
import pandas as pd

# Stats dump interval so we can calculate rates.
dt = 1.0

colors = ["blue", "green", "red"]

# Helper function to read CSV files and return x, y values
def xy_from_csv(filename, data_dir):
    x = []
    y = []
    path = os.path.join(data_dir, filename)
    if os.path.exists(path):
        with open(path, 'r') as csvfile:
            plots = csv.reader(csvfile, delimiter=',')
            for row in plots:
                x.append(float(row[0]))
                y.append(float(row[1]))
    else:
        print(f"File {path} does not exist.")
    return x, y

# Adjust the x-axis time values to start from 0 and convert to seconds
def adjust(xs, xstart):
    return [(x - xstart) / 1e9 for x in xs]

# Main plotting function
def plot_data(data_dir):
    rq_latency_x, rq_latency_y = [], []
    timeout_timestamps = []

    # Read data from CSV files
    in_rq_rate_x, in_rq_rate_y = xy_from_csv("client.rps.0.csv", data_dir)
    out_rq_rate_x, out_rq_rate_y = xy_from_csv("client.req.count.0.csv", data_dir)
    retry_rate_x, retry_rate_y = xy_from_csv("client.req.retry.count.0.csv", data_dir)
    rq_latency_x, rq_latency_y = xy_from_csv("client.req.latency.0.csv", data_dir)
    success_stamps, _ = xy_from_csv("client.req.success_hist.0.csv", data_dir)
    goodput_x, goodput_y = xy_from_csv("client.req.success.count.0.csv", data_dir)
    failure_x, failure_y = xy_from_csv("client.req.failure.count.0.csv", data_dir)
    timeout_x, timeout_y = xy_from_csv("client.req.timeout.0.csv", data_dir)
    timeout_origin_x, timeout_origin_y = xy_from_csv("client.req.timeout_origin.0.csv", data_dir)
    goodput_y = [y / dt for y in goodput_y]

    fig, (ax1, ax2) = plt.subplots(2)
    ymax = max(in_rq_rate_y + goodput_y + failure_y + retry_rate_y)
    xstart = min(in_rq_rate_x + out_rq_rate_x + rq_latency_x + goodput_x + failure_x + timeout_x + timeout_origin_x)
    xend = (max(in_rq_rate_x + out_rq_rate_x + rq_latency_x + goodput_x + failure_x + timeout_x + timeout_origin_x) - xstart) / 1e9

    adjusted_rq_latency_endtime = [adjusted_start_time + latency for adjusted_start_time, latency in zip(adjust(rq_latency_x, xstart), rq_latency_y)]

    # Plot request latency
    ax1.set_xlabel('Time (s)', fontsize=18)
    ax1.set_ylabel('Latency(ms)', fontsize=16)
    ax1.scatter(adjust(rq_latency_x, xstart), rq_latency_y, color=colors[0], label="observed latency", marker='.', alpha=0.2)
    ax1.tick_params(axis='x', labelsize=16)
    ax1.tick_params(axis='y', labelsize=16)
    ax1.set_xlim([0, xend])
    ax1.set_ylim([0, max(rq_latency_y) * 1.3])
    ax1.legend(fontsize=12, loc='upper right')

    # Plot offered load, goodput, failure, and retries
    ax2.set_xlabel('Time (s)', fontsize=18)
    ax2.set_ylabel('Offered Load', fontsize=16)
    ax2.plot(adjust(failure_x, xstart), failure_y, color="black", label="failure", linestyle="--", marker='o')
    ax2.plot(adjust(retry_rate_x, xstart), retry_rate_y, color="cyan", label="retries", marker='x', linestyle=":")
    ax2.plot(adjust(goodput_x, xstart), goodput_y, color="green", label="goodput")
    ax2.plot(adjust(in_rq_rate_x, xstart), in_rq_rate_y, label="load")

    # Plot timeout lines
    # if len(timeout_x) > 0:
    #     ax2.axvline(adjust(timeout_x, xstart)[0], ymin=0.5, ymax=0.1, color="red", linestyle="-", alpha=0.7, label="timeout")
    #     quantized_timeout_x = [round(tx, 1) for tx in adjust(timeout_x, xstart)]
    #     timeout_counts = Counter(quantized_timeout_x)
    #     max_count = max(timeout_counts.values())
    #     for tx, count in timeout_counts.items():
    #         alpha_value = min(1.0, count / max_count * 0.9 + 0.2)
    #         ax2.axvline(tx, ymin=0.5, ymax=1.0, color="red", linestyle="-", alpha=alpha_value)
    num_timeout = len(timeout_x)
    num_failure = len(failure_x)
    stats_text = (
        f"Time out: {num_timeout:.0f}\n"
        f"Failure: {num_failure:.0f}"
    )
    plt.text(
        0.95, 0.05, stats_text,
        horizontalalignment='right',
        verticalalignment='bottom',
        transform=plt.gca().transAxes,
        fontsize=14,
        bbox=dict(facecolor='white', alpha=0.5)
    )

    ax2.tick_params(axis='x', labelsize=16)
    ax2.tick_params(axis='y', labelsize=16)
    ax2.set_xlim([0, xend])
    ax2.set_ylim([0, ymax * 1.3])
    ax2.grid(True)
    ax2.legend(fontsize=12, ncol=2, loc='upper right')
    plt.tight_layout()
    result_path = f"{data_dir}/result.pdf"
    plt.savefig(result_path)
    print(f"Plot latency timeline saved to ./{result_path}")
    plt.show()

# # Main plotting function
# def plot_data(data_dir):
#     rq_latency_x, rq_latency_y = [], []
#     timeout_timestamps = []

#     fig, (ax1, ax2) = plt.subplots(2)

#     # Read data from CSV files
#     in_rq_rate_x, in_rq_rate_y = xy_from_csv("client.rps.0.csv", data_dir)
#     out_rq_rate_x, out_rq_rate_y = xy_from_csv("client.req.count.0.csv", data_dir)
#     retry_rate_x, retry_rate_y = xy_from_csv("client.req.retry.count.0.csv", data_dir)
#     rq_latency_x, rq_latency_y = xy_from_csv("client.req.latency.0.csv", data_dir)
#     success_stamps, _ = xy_from_csv("client.req.success_hist.0.csv", data_dir)
#     goodput_x, goodput_y = xy_from_csv("client.req.success.count.0.csv", data_dir)
#     failure_x, failure_y = xy_from_csv("client.req.failure.count.0.csv", data_dir)
#     timeout_x, timeout_y = xy_from_csv("client.req.timeout.0.csv", data_dir)
#     timeout_origin_x, timeout_origin_y = xy_from_csv("client.req.timeout_origin.0.csv", data_dir)

#     # Adjust for dt
#     goodput_y = [y / dt for y in goodput_y]

#     ymax = max(in_rq_rate_y + goodput_y + failure_y + retry_rate_y)
#     xstart = min(in_rq_rate_x + out_rq_rate_x + rq_latency_x + goodput_x + failure_x + timeout_x + timeout_origin_x)
#     xend = (max(in_rq_rate_x + out_rq_rate_x + rq_latency_x + goodput_x + failure_x + timeout_x + timeout_origin_x) - xstart) / 1e9

#     adjusted_rq_latency_endtime = [adjusted_start_time + latency for adjusted_start_time, latency in zip(adjust(rq_latency_x, xstart), rq_latency_y)]

#     # Plot request latency
#     ax1.set_xlabel('Time (s)', fontsize=18)
#     ax1.set_ylabel('Latency(ms)', fontsize=16)
#     ax1.scatter(adjust(rq_latency_x, xstart), rq_latency_y, color=colors[0], label="observed latency", marker='.', alpha=0.2)
#     ax1.tick_params(axis='x', labelsize=16)
#     ax1.tick_params(axis='y', labelsize=16)
#     ax1.set_xlim([0, xend])
#     ax1.set_ylim([0, max(rq_latency_y) * 1.3])
#     ax1.legend(fontsize=12, loc='upper right')

#     # Plot offered load, goodput, failure, and retries
#     ax2.set_xlabel('Time (s)', fontsize=18)
#     ax2.set_ylabel('Offered Load', fontsize=16)
#     ax2.plot(adjust(in_rq_rate_x, xstart), in_rq_rate_y, label="load")
#     ax2.plot(adjust(goodput_x, xstart), goodput_y, color="green", label="goodput", marker='^')
#     ax2.plot(adjust(failure_x, xstart), failure_y, color="black", label="failure", linestyle="--", marker='o')
#     ax2.plot(adjust(retry_rate_x, xstart), retry_rate_y, color="cyan", label="retries", marker='x', linestyle=":")

#     # Plot timeout lines
#     if len(timeout_x) > 0:
#         ax2.axvline(adjust(timeout_x, xstart)[0], ymin=0.5, ymax=0.1, color="red", linestyle="-", alpha=0.7, label="timeout")
#         quantized_timeout_x = [round(tx, 1) for tx in adjust(timeout_x, xstart)]
#         timeout_counts = Counter(quantized_timeout_x)
#         max_count = max(timeout_counts.values())
#         for tx, count in timeout_counts.items():
#             alpha_value = min(1.0, count / max_count * 0.9 + 0.2)
#             ax2.axvline(tx, ymin=0.5, ymax=1.0, color="red", linestyle="-", alpha=alpha_value)

#     ax2.tick_params(axis='x', labelsize=16)
#     ax2.tick_params(axis='y', labelsize=16)
#     ax2.set_xlim([0, xend])
#     ax2.set_ylim([0, ymax * 1.3])
#     ax2.grid(True)
#     ax2.legend(fontsize=12, ncol=2, loc='upper right')
#     plt.tight_layout()
#     result_path = f"{data_dir}/result.pdf"
#     plt.savefig(result_path)
#     print(f"Plot latency timeline saved to ./{result_path}")
#     plt.show()
    
def plot_cdf(data_dir): # data_file = 'client.req.latency.0.csv'
    _, rq_latency_y = xy_from_csv("client.req.latency.0.csv", data_dir)
    rq_latency_y = np.array(rq_latency_y)
    sorted_y = np.sort(rq_latency_y)
    p = np.arange(1, len(sorted_y)+1) / len(sorted_y)
    plt.figure(figsize=(6, 5))
    plt.plot(sorted_y, p, marker='.', linestyle='none')
    plt.xlabel('Latency (ms)', fontsize=20)
    plt.ylabel('Cumulative Probability', fontsize=20)
    plt.title('CDF of Request Latency', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim([0, max(sorted_y)])
    plt.ylim([0, 1.01])
    plt.tight_layout()
    
    avg_latency = np.mean(sorted_y)
    p50_latency = np.percentile(sorted_y, 50)
    p90_latency = np.percentile(sorted_y, 90)
    p99_latency = np.percentile(sorted_y, 99)
    p999_latency = np.percentile(sorted_y, 99.9)
    stats_text = (
    f"Average Latency: {avg_latency:.2f} ms\n"
    f"P50 Latency: {p50_latency:.2f} ms\n"
    f"P90 Latency: {p90_latency:.2f} ms\n"
    f"P99 Latency: {p99_latency:.2f} ms\n"
    f"P99.9 Latency: {p999_latency:.2f} ms"
    )
    plt.text(
        0.95, 0.05, stats_text,
        horizontalalignment='right',
        verticalalignment='bottom',
        transform=plt.gca().transAxes,
        fontsize=12,
        bbox=dict(facecolor='white', alpha=0.5)
    )
    
    result_path = f"{data_dir}/cdf.pdf"
    plt.savefig(result_path)
    print(f"Plot CDF saved to ./{result_path}")
    plt.show()
    
    
# Main function to run the script
def main():
    if len(sys.argv) != 2:
        print("Usage: plot.py <data_dir>")
        exit(1)

    data_dir = sys.argv[1]
    if not os.path.exists(data_dir):
        print(f"No data directory provided or found: {data_dir}")
        sys.exit(1)

    plot_data(data_dir)
    
    plot_cdf(data_dir)

if __name__ == "__main__":
    main()
