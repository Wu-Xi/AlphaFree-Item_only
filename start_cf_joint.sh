#!/usr/bin/env bash
# GPU SERVER ONLY. Usage: bash start_cf_joint.sh Movie 0,1,2,3 /absolute/output
set -euo pipefail
cf_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cf_dataset="${1:-Movie}"
cf_gpu_csv="${2:-0}"
cf_output="${3:-${cf_script_dir}/weights/cf_joint_${cf_dataset}}"
cf_python="${CF_PYTHON:-python}"
if [[ ! "$cf_gpu_csv" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "GPU list must be comma-separated numeric IDs, e.g. 0,1,2,3" >&2
  exit 1
fi
IFS=',' read -r -a cf_gpus <<< "$cf_gpu_csv"
declare -A cf_seen=()
for cf_gpu in "${cf_gpus[@]}"; do
  if [[ -n "${cf_seen[$cf_gpu]:-}" ]]; then
    echo "Duplicate GPU ID: $cf_gpu" >&2
    exit 1
  fi
  cf_seen[$cf_gpu]=1
done
"$cf_python" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable in this Python environment"'
# One initializer avoids concurrent manifest creation. Paths are not changed by cd.
"$cf_python" "${cf_script_dir}/search_cf.py" --datasets "$cf_dataset" --output "$cf_output" --prepare
mkdir -p "$cf_output/scheduler_logs"
cf_stamp="$(date +%Y%m%d_%H%M%S)"
for cf_index in "${!cf_gpus[@]}"; do
  cf_gpu="${cf_gpus[$cf_index]}"
  cf_log="$cf_output/scheduler_logs/shard_${cf_index}_${cf_stamp}_$$.log"
  nohup env CUDA_VISIBLE_DEVICES="$cf_gpu" "$cf_python" -u "${cf_script_dir}/search_cf.py" \
    --datasets "$cf_dataset" --output "$cf_output" --device cuda:0 \
    --num-shards "${#cf_gpus[@]}" --shard-index "$cf_index" --execute \
    > "$cf_log" 2>&1 &
  cf_pid=$!
  echo "Started GPU=$cf_gpu shard=$cf_index/${#cf_gpus[@]} PID=$cf_pid log=$cf_log"
done
echo "Summarize after completion: $cf_python ${cf_script_dir}/search_cf.py --output $cf_output --summarize"
