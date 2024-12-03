import os
import re
import pandas as pd
import sys

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
    if value.endswith("ms"):
        return float(value.replace("ms", ""))
    elif value.endswith("s"):
        return float(value.replace("s", "")) * 1000
    else:
        raise ValueError(f"Unexpected latency value: {value}")

def calculate_success_failure_throughput_and_latency(directory, failure_latency=10000.0):
    total_success = 0
    total_failure = 0
    total_throughput = 0.0
    total_requests_sum = 0
    weighted_success_sum = 0
    weighted_latency_sum = 0.0
    weighted_latency_99_sum = 0.0
    total_latency_including_failures = 0.0

    for file_name in os.listdir(directory):
        if file_name.endswith(".txt"):
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

    success_ratio = (weighted_success_sum / total_requests_sum) * 100 if total_requests_sum > 0 else 0.0
    average_latency_mean = (weighted_latency_sum / total_requests_sum) if total_requests_sum > 0 else 0.0
    average_latency_99 = (weighted_latency_99_sum / total_requests_sum) if total_requests_sum > 0 else 0.0
    average_latency_2 = (total_latency_including_failures / total_requests_sum) if total_requests_sum > 0 else 0.0
    return total_success, total_failure, total_throughput, success_ratio, average_latency_mean, average_latency_99, average_latency_2

def process_subdirectories(base_dir):
    results = []
    for sub_dir in [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]:
        total_success, total_failure, total_throughput, weighted_success_ratio, avg_latency, avg_latency_99, avg_latency_2 = calculate_success_failure_throughput_and_latency(sub_dir)
        results.append({
            "Directory": os.path.basename(sub_dir),
            "Total Success": total_success,
            "Total Failure": total_failure,
            "Total Throughput (RPS)": total_throughput,
            "Weighted Success Ratio (%)": weighted_success_ratio,
            "Average Latency (ms)": avg_latency,
            "99th Percentile Latency (ms)": avg_latency_99,
            "Average Latency 2 (ms)": avg_latency_2
        })
    return results

def main():
    if len(sys.argv) < 2:
        print("Usage: python parse-corecontrast.py <base_dir>")
        sys.exit(1)
    base_dir = sys.argv[1]
    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        sys.exit(1)

    results = process_subdirectories(base_dir)
    df = pd.DataFrame(results)
    if not df.empty:
        output_file = f"{base_dir}/subdirectory_statistics.csv"
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        print(df)
    else:
        print("No data found in subdirectories.")

if __name__ == "__main__":
    main()
