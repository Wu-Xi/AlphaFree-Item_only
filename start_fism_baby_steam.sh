#!/usr/bin/env bash
# Usage: bash start_fism_baby_steam.sh 0,1,2,3 weights/fism_baby_steam_v1
# One independent training process per selected GPU, 60 jobs total at seed 2025.
set -euo pipefail
fism_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
fism_gpu_csv="${1:-0}"
fism_output="${2:-${fism_dir}/weights/fism_baby_steam_v1}"
fism_python="${CF_PYTHON:-python}"
if [[ ! "$fism_gpu_csv" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "GPU IDs must be comma-separated integers, e.g. 0,2,3" >&2
  exit 1
fi
IFS=',' read -r -a fism_gpus <<< "$fism_gpu_csv"
declare -A fism_seen=()
for fism_gpu in "${fism_gpus[@]}"; do
  if [[ -n "${fism_seen[$fism_gpu]:-}" ]]; then
    echo "Duplicate GPU ID: $fism_gpu" >&2
    exit 1
  fi
  fism_seen[$fism_gpu]=1
done
if (( ${#fism_gpus[@]} > 60 )); then
  echo "At most 60 GPU workers for 60 jobs" >&2
  exit 1
fi
# Check all requested visibility IDs before starting any worker.
for fism_gpu in "${fism_gpus[@]}"; do
  CUDA_VISIBLE_DEVICES="$fism_gpu" "$fism_python" -c \
    'import torch; assert torch.cuda.is_available() and torch.cuda.device_count() == 1, "Selected GPU unavailable"; torch.zeros(1, device="cuda:0")'
done
fism_common=(
  --config "${fism_dir}/configs/fism_baby_steam.json"
  --datasets Baby Steam --models fism --phase both --seeds 2025
  --duplicate-items count
  --output "$fism_output"
)
"$fism_python" "${fism_dir}/search_cf.py" "${fism_common[@]}" --prepare
mkdir -p "$fism_output/scheduler_logs"
fism_stamp="$(date +%Y%m%d_%H%M%S)"
for fism_index in "${!fism_gpus[@]}"; do
  fism_gpu="${fism_gpus[$fism_index]}"
  fism_log="$fism_output/scheduler_logs/shard_${fism_index}_${fism_stamp}_$$.log"
  nohup env CUDA_VISIBLE_DEVICES="$fism_gpu" "$fism_python" -u "${fism_dir}/search_cf.py" \
    "${fism_common[@]}" --device cuda:0 --num-shards "${#fism_gpus[@]}" \
    --shard-index "$fism_index" --execute > "$fism_log" 2>&1 &
  echo "Started GPU=$fism_gpu shard=$fism_index/${#fism_gpus[@]} PID=$! log=$fism_log"
done
echo "Summary: $fism_python ${fism_dir}/search_cf.py --output $fism_output --summarize"
