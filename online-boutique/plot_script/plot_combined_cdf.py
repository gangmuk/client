#!/usr/bin/python

import matplotlib.pyplot as plt
import csv
import sys
import os
from collections import Counter
import numpy as np
import pandas as pd
import argparse

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

def clip_data(args, x, y, total_duration):
    clipped_x = [(args.clip_front <= t <= (total_duration - args.clip_end)) for t in x]
    return np.array(x)[clipped_x], np.array(y)[clipped_x]

def preprocess_data(args, x_, y_, total_duration):
    xstart = min(x_)
    xend = max(x_)
    adjusted_x = adjust(x_, xstart)
    x_, y_ = clip_data(args, adjusted_x, y_, total_duration)
    return x_, y_

def mask(args, latency_x, latency_y, total_duration):
    latency_mask = [(args.clip_front <= t <= (total_duration - args.clip_end)) for t in latency_x]
    masked_latency_y = latency_y[latency_mask]
    return masked_latency_y

def plot_cdf(data_dir, plt):
    west_rq_latency_x, west_rq_latency_y = xy_from_csv("client.req.latency.0.csv", f"{data_dir}/client-west")
    east_rq_latency_x, east_rq_latency_y = xy_from_csv("client.req.latency.0.csv", f"{data_dir}/client-east")
    
    west_total_duration = (max(west_rq_latency_x) - min(west_rq_latency_x)) / 1e9
    east_total_duration = (max(east_rq_latency_x) - min(east_rq_latency_x)) / 1e9
    
    adjusted_west_latency_x, adjusted_west_rq_latency_y = preprocess_data(args, west_rq_latency_x, west_rq_latency_y, west_total_duration)
    adjusted_east_latency_x, adjusted_east_rq_latency_y = preprocess_data(args, east_rq_latency_x, east_rq_latency_y, east_total_duration)
    
    masked_west_rq_latency_y = mask(args, adjusted_west_latency_x, adjusted_west_rq_latency_y, west_total_duration)
    masked_east_rq_latency_y = mask(args, adjusted_east_latency_x, adjusted_east_rq_latency_y, east_total_duration)
    
    print(f"masked_west_rq_latency_y: {masked_west_rq_latency_y}")
    print(f"masked_east_rq_latency_y: {masked_east_rq_latency_y}")
    
    all_rq_latency_y = np.concatenate([masked_west_rq_latency_y, masked_east_rq_latency_y])
    print(all_rq_latency_y)
    sorted_y = np.sort(all_rq_latency_y)
    num_points = 1000  # Adjust based on performance needs
    if len(sorted_y) > num_points:
        indices = np.linspace(0, len(sorted_y) - 1, num_points).astype(int)
        sorted_y = sorted_y[indices]
        p = np.linspace(0, 1, num_points)
    else:
        p = np.arange(1, len(sorted_y) + 1) / len(sorted_y)

    label_ = data_dir.split("/")[-1]
    if label_ == "":
        label_ = data_dir.split("/")[-2]
    print(f"data_dir: {data_dir}, label: {label_}")
    plt.plot(sorted_y, p, label=label_, alpha=0.7) #, marker='o', markerfacecolor='none')

    # Calculate latency statistics
    avg_latency = np.mean(all_rq_latency_y)
    p50_latency = np.percentile(all_rq_latency_y, 50)
    p90_latency = np.percentile(all_rq_latency_y, 90)
    p99_latency = np.percentile(all_rq_latency_y, 99)
    p999_latency = np.percentile(all_rq_latency_y, 99.9)
    max_latency = np.max(all_rq_latency_y)
    stats_text = (
        f"Average: {avg_latency:.2f} ms ({label_})\n"
        f"P50: {p50_latency:.2f} ms ({label_})\n"
        f"P90: {p90_latency:.2f} ms ({label_})\n"
        f"P99: {p99_latency:.2f} ms ({label_})\n"
        f"P99.9: {p999_latency:.2f} ms ({label_})\n"
        f"Max: {max_latency:.2f} ms ({label_})\n"
        "--------------------------\n"
    )
    return stats_text

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir1", type=str, help="Path to the data directory", required=True)
    parser.add_argument("--data_dir2", type=str, help="Path to the data directory", required=True)
    parser.add_argument("--clip_front", type=float, default=0, help="Number of seconds to clip from the start and end")
    parser.add_argument("--clip_end", type=float, default=0, help="Number of seconds to clip from the end")
    parser.add_argument("--out", type=str, help="Output file path")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.data_dir1):
        print(f"No data directory provided or found: {args.data_dir1}")
        sys.exit(1)
    if not os.path.exists(args.data_dir2):
        print(f"No data directory provided or found: {args.data_dir1}")
        sys.exit(1)
    
    # outdir is the directory where the output file will be saved
    # if args.data_dir1 is part3-250ms-withandwithout/addtocart-W400-E100-bg50/slate-with-jumping,
    # then outdir will be part3-250ms-withandwithout/addtocart-W400-E100-bg50
    print(f"data_dir1: {args.data_dir1}")
    normalized_directory = os.path.normpath(args.data_dir1)
    outdir = '/'.join(normalized_directory.split('/')[:2]) + '/'
    print(f"outdir: {outdir}")
    plt.figure(figsize=(6, 5))
    stats_text = ""
    stats_text += plot_cdf(args.data_dir1, plt)
    stats_text += plot_cdf(args.data_dir2, plt)
    
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
    plt.legend(fontsize=12, loc='upper right')
    plt.tight_layout()    
    
    result_path = f"{outdir}/{args.out}"
    plt.savefig(result_path)
    print(f"Plot CDF saved to ./{result_path}")
    plt.show()