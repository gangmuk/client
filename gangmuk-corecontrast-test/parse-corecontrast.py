import os
import pandas as pd

def parse_latency_file(file_path):
    """
    Parse a latency file and extract relevant metrics, including requests and latency percentiles.
    """
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
        return metrics
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None

def convert_to_ms(value):
    """
    Convert a latency value (e.g., '24.855ms,' or '11.729s') to milliseconds as a float.
    Handles values ending in 'ms,' or 's,' with trailing commas.
    """
    value = value.strip(",")  # Remove trailing commas
    if value.endswith("ms"):
        return float(value.replace("ms", ""))
    elif value.endswith("s"):
        return float(value.replace("s", "")) * 1000  # Convert seconds to milliseconds
    else:
        raise ValueError(f"Unexpected latency value: {value}")

def collect_experiment_data(base_dir):
    """
    Collect latency data across experiments.
    """
    experiment_dirs = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if d.startswith("corecontrast-test")
    ]
    results = []

    for experiment_dir in experiment_dirs:
        multicore_dir = os.path.join(experiment_dir, "multicore-w50")
        if not os.path.exists(multicore_dir):
            print(f"Skipping {experiment_dir}: Missing 'multicore-w50' directory.")
            continue

        for scheme in ["SLATE-with-jumping-global", "SLATE-without-jumping"]:
            scheme_path = os.path.join(multicore_dir, scheme, "latency_results")
            if not os.path.exists(scheme_path):
                print(f"Skipping {scheme_path}: Missing directory.")
                continue

            latency_files = [
                os.path.join(scheme_path, f) for f in os.listdir(scheme_path) if f.endswith(".txt")
            ]
            for latency_file in latency_files:
                metrics = parse_latency_file(latency_file)
                if metrics:
                    results.append(
                        {
                            "Experiment": os.path.basename(experiment_dir),
                            "Scheme": scheme,
                            "File": os.path.basename(latency_file),
                            **metrics,
                        }
                    )

    return results

import sys
def main():
    if len(sys.argv) < 2:
        print("Usage: python parse-corecontrast.py <base_dir>")
        sys.exit(1)
    base_dir = sys.argv[1]
    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        sys.exit(1)
    data = collect_experiment_data(base_dir)

    # Convert results to a DataFrame for analysis
    df = pd.DataFrame(data)
    if not df.empty:
        # Save results to a CSV
        output_file = f"{base_dir}/latency_comparison_results.csv"
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
    else:
        print("No data was collected. Check directory structure and files.")

if __name__ == "__main__":
    main()
