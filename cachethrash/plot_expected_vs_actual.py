import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Nice default plotting style
sns.set(style="whitegrid", palette="muted", font_scale=1.2)

def plot_expected_vs_actual(csv_path: str, output_dir: str):
    df = pd.read_csv(csv_path, names=[
        "index", "src_traffic_class", "dst_traffic_class",
        "src_region", "dst_region", "expected", "actual", "ruleset_rps"
    ])

    os.makedirs(output_dir, exist_ok=True)

    # Group by (src_traffic_class, src_region, dst_region)
    grouped = df.groupby(["src_traffic_class", "src_region", "dst_region"])

    for (src_tc, src_region, dst_region), group in grouped:
        group = group.sort_values("index")
        plt.figure(figsize=(10, 6))

        # Optional smoothing
        # group["expected"] = group["expected"].rolling(3, min_periods=1).mean()
        # group["actual"] = group["actual"].rolling(3, min_periods=1).mean()

        plt.plot(group["index"], group["expected"],
                 label="Expected", linestyle='--', linewidth=2, marker='o')
        plt.plot(group["index"], group["actual"],
                 label="Actual", linestyle='-', linewidth=2, marker='x')

        plt.title(f"Latency Over Time\nsrc: {src_tc} ({src_region}) → dst: {dst_region}", fontsize=14)
        plt.xlabel("Time (index)", fontsize=12)
        plt.ylabel("Latency (ms)", fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(alpha=0.3)
        plt.legend(loc='upper left', fontsize=10)
        plt.tight_layout()

        # Sanitize and save file
        filename = f"{src_tc.replace('/', '_').replace('@', '_at_')}__{src_region}__{dst_region}.pdf"
        plt.savefig(os.path.join(output_dir, filename), format='pdf')
        plt.close()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python plot_latencies.py <input_csv> <output_dir>")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_dir = sys.argv[2]
    plot_expected_vs_actual(input_csv, output_dir)
