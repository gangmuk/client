#!/bin/bash

# Outer directory provided as an argument
outer_dir=$1

# Check if outer directory is provided
if [ -z "$outer_dir" ]; then
    echo "Please provide the outer directory."
    exit 1
fi

# Iterate over each experiment in the outer directory
for experiment in "$outer_dir"/*/; do
    # Get the experiment name
    experiment_name=$(basename "$experiment")

    # Define the directories for the current experiment
    data_dir1="$experiment/SLATE-without-jumping"
    data_dir2_local="$experiment/SLATE-with-jumping-local"
    data_dir2_global="$experiment/SLATE-with-jumping-global"

    # Check if required directories exist
    if [ -d "$data_dir1" ] && [ -d "$data_dir2_local" ]; then
        # Run plot_combined_cdf.py for SLATE-with-jumping-local
        python plot_combined_cdf.py --data_dir1 "$data_dir1" --data_dir2 "$data_dir2_local" --clip_front 60 --clip_end 60 --out "SLATE-local-jumping.pdf"
    else
        echo "Skipping $experiment_name: Required directories for local-jumping comparison are missing."
    fi

    if [ -d "$data_dir1" ] && [ -d "$data_dir2_global" ]; then
        # Run plot_combined_cdf.py for SLATE-with-jumping-global
        python plot_combined_cdf.py --data_dir1 "$data_dir1" --data_dir2 "$data_dir2_global" --clip_front 60 --clip_end 60 --out "SLATE-global-jumping.pdf"
    else
        echo "Skipping $experiment_name: Required directories for global-jumping comparison are missing."
    fi

done
