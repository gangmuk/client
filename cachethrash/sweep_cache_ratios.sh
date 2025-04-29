#!/usr/bin/env bash
#
# drive_cache_sweep.sh  <directory>
#
# This script will run only the specified (west_start, south_start) pairs.

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <base_directory>"
  exit 1
fi

base_dir=$1

if [[ -d "$base_dir" ]]; then
  echo "Directory $base_dir already exists."
else
  mkdir -p "$base_dir"
  echo "Created base directory: $base_dir"
fi

pairs=(
  "10 60"
)

# west_vals=(10 35 60 85)
# south_vals=($(seq 0 10 100))

# for west_start in "${west_vals[@]}"; do
#   for south_start in "${south_vals[@]}"; do
#     pairs+=("$west_start $south_start")
#   done
# done

echo "Will re‑run the following (west_start, south_start) pairs:"
for p in "${pairs[@]}"; do
  echo "  $p"
done
echo

for p in "${pairs[@]}"; do
  # split the two numbers
  read west_start south_start <<< "$p"

  run_dir="$base_dir/west_${west_start}_south_${south_start}"
  echo "→ Running with west=$west_start, south=$south_start"
  echo "  output directory: $run_dir"

  # create the directory (if you like) and invoke your python driver
  # mkdir -p "$run_dir"
  python run_cachethrash_new.py 0 "$run_dir" "$west_start" "$south_start"

  # restart your deployment so the new run sees a clean slate
  echo "  restarting deployment…"
  kubectl rollout restart deploy
  echo
done

echo "All requested runs completed."
