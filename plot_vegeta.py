import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt

def extract_latencies_and_weight(file_content):
    """
    Extract specific latencies (50th, 90th, 95th, 99th) and calculate the weight for weighted average.
    """
    total_requests = None
    avg_latency = None
    latencies = []
    
    # Extract total requests
    requests_match = re.search(r"Requests\s+\[.*?\]\s+([^\n]+)", file_content)
    if requests_match:
        total_requests = int(requests_match.group(1).split(",")[0].strip())
        print(f"Total requests: {total_requests}")
    
    # Extract average latency
    latencies_match = re.search(r"Latencies\s+\[.*?\]\s+([^\n]+)", file_content)
    if latencies_match:
        latencies_raw = latencies_match.group(1).split(",")
        try:
            latencies = []
            for i, lat in enumerate(latencies_raw):
                print(f"Latency {i}: {lat.strip()}")
                if i in [2, 3, 4, 5]:
                    if "ms" in lat.strip():
                        latencies.append(float(lat.strip().replace("ms", "")) * 1)
                    else:
                        latencies.append(float(lat.strip().replace("s", "")) * 1000)
                elif i == 1:
                    if "ms" in lat.strip():
                        avg_latency = (float(lat.strip().replace("ms", "")) * 1)
                    else:
                        avg_latency = (float(lat.strip().replace("s", "")) * 1000)
            # print(f"Avg latency: {avg_latency}")
            # print(f"Latencies: {latencies}")
            # print(f"Total requests: {total_requests}")
        except ValueError:
            print("Invalid latency format found. Skipping...")
            return None, None, None

    return latencies, total_requests, avg_latency

def calculate_cdf(data):
    """Calculate the CDF for a dataset."""
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return sorted_data, cdf

def process_directory(directory):
    """
    Process all files in a directory to plot the CDF and compute the weighted average latency.
    """
    latencies_all_files = []
    file_labels = []
    weighted_latency_sum = 0
    total_request_sum = 0

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    latencies, total_requests, avg_latency = extract_latencies_and_weight(content)
                    if latencies and total_requests and avg_latency:
                        # Add to CDF data
                        latencies_all_files.append(latencies)
                        file_labels.append(filename)
                        # Update weighted average calculation
                        weighted_latency_sum += total_requests * avg_latency
                        total_request_sum += total_requests
            except Exception as e:
                print(f"Error reading file {filename}: {e}")
                continue

    plt.figure(figsize=(10, 6))

    # Plot CDFs for individual files
    for latencies, label in zip(latencies_all_files, file_labels):
        sorted_data, cdf = calculate_cdf(latencies)
        plt.plot(sorted_data, cdf, label=f"File: {label}")

    # Compute and display the weighted average latency
    if total_request_sum > 0:
        global_avg_latency = weighted_latency_sum / total_request_sum
        plt.text(
            0.7, 0.95, f"Weighted Avg Latency: {global_avg_latency:.2f} ms",
            transform=plt.gca().transAxes,
            fontsize=14,
            color='red',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='red')
        )

    # Finalize plot
    plt.title("CDF of Latencies (50th to 99th Percentile)")
    plt.xlabel("Latency (ms)")
    plt.ylabel("CDF")
    plt.legend()
    plt.grid(True)

    # Save plot to PDF in the specified directory
    output_path = os.path.join(directory, "latencies.pdf")
    plt.savefig(output_path)
    plt.close()
    print(f"CDF plot saved to {output_path}")

def process_directory_from_argv():
    """Process directory given as argv[1]."""
    if len(sys.argv) < 2:
        print("Usage: python script.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        sys.exit(1)

    process_directory(directory)

if __name__ == "__main__":
    process_directory_from_argv()
