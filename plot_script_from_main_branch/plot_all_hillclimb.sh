#!/bin/bash
set -x
# Check if the parent directory argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <parent_directory>"
    exit 1
fi

# Set the parent directory from the first argument
parent_dir="$1"

# Iterate over all subdirectories in the parent directory
for sub_dir in "$parent_dir"/*; do
    if [ -d "$sub_dir" ]; then
        # Construct the input CSV file path
        csv_file1="$sub_dir/SLATE-with-jumping-local/SLATE-with-jumping-local-hillclimbing_distribution_history.csv"
        csv_file2="$sub_dir/SLATE-with-jumping-global/SLATE-with-jumping-global-hillclimbing_distribution_history.csv"

        # Construct the output PDF file path
        output_pdf1="$sub_dir/SLATE-with-jumping-local/hill.pdf"
        outavg1="$sub_dir/SLATE-with-jumping-local/hill_avg.pdf"
        output_pdf2="$sub_dir/SLATE-with-jumping-global/hill.pdf"
        outavg2="$sub_dir/SLATE-with-jumping-global/hill_avg.pdf"

        # Check if the CSV file exists before running the Python script
        if [ -f "$csv_file1" ]; then
            python plot_hillclimb.py "$csv_file1" "$output_pdf1"
            python plot_avg_hillclimb.py "$csv_file1" "$outavg1"
        else
            echo "CSV1 file not found for $sub_dir, skipping."
        fi

        if [ -f "$csv_file2" ]; then
            python plot_hillclimb.py "$csv_file2" "$output_pdf2"
            python plot_avg_hillclimb.py "$csv_file2" "$outavg2"
        else
            echo "CSV1 file not found for $sub_dir, skipping."
        fi
    fi
done