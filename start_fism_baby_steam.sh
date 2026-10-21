#!/usr/bin/env bash
# Usage: bash start_fism_baby_steam.sh 0,1,2,3 weights/fism_baby_steam_reg1e-6_v2 3
# Three independent workers per GPU by default; 60 jobs total at seed 2025.
set -euo pipefail
fism_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
fism_gpu_csv="${1:-0}"
fism_output="${2:-${fism_dir}/weights/fism_baby_steam_reg1e-6_v2}"
fism_per_gpu="${3:-3}"
fism_python="${CF_PYTHON:-python}"
if [[ ! "$fism_gpu_csv" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "GPU IDs must be comma-separated integers, e.g. 0,2,3" >&2
  exit 1
fi
IFS=',' read -r -a fism_gpus <<< "$fism_gpu_csv"
if [[ ! "$fism_per_gpu" =~ ^[1-9][0-9]?$ ]] || (( fism_per_gpu > 60 )); then
  echo "Processes per GPU must be an integer between 1 and 60" >&2
  exit 1
fi
fism_workers=$(( ${#fism_gpus[@]} * fism_per_gpu ))
if (( fism_workers > 60 )); then
  echo "At most 60 total workers for 60 jobs; reduce GPUs or processes per GPU" >&2
  exit 1
fi
fism_seen_gpus=","
fism_worker_gpus=()
for fism_gpu in "${fism_gpus[@]}"; do
  if [[ "$fism_seen_gpus" == *",${fism_gpu},"* ]]; then
    echo "GPU IDs must be unique; use the third argument for processes per GPU" >&2
    exit 1
  fi
  fism_seen_gpus+="${fism_gpu},"
  for ((fism_slot=0; fism_slot<fism_per_gpu; fism_slot++)); do
    fism_worker_gpus+=("$fism_gpu")
  done
done
# Check all requested visibility IDs before starting any worker.
for fism_gpu in "${fism_gpus[@]}"; do
  CUDA_VISIBLE_DEVICES="$fism_gpu" "$fism_python" -c \
    'import torch; assert torch.cuda.is_available() and torch.cuda.device_count() == 1, "Selected GPU unavailable"; torch.zeros(1, device="cuda:0")'
done
fism_common=(
  --config "${fism_dir}/configs/fism_baby_steam.json"
  --datasets Baby Steam --models fism --phase both --seeds 2025
  --duplicate-items count
  --threads 2
  --test-after-train --delete-checkpoint-after-test --log-every 0
  --output "$fism_output"
)
"$fism_python" "${fism_dir}/search_cf.py" "${fism_common[@]}" --prepare
mkdir -p "$fism_output/scheduler_logs"
fism_stamp="$(date +%Y%m%d_%H%M%S)"
echo "Launching ${#fism_gpus[@]} GPUs x $fism_per_gpu workers = $fism_workers processes"
for fism_index in "${!fism_worker_gpus[@]}"; do
  fism_gpu="${fism_worker_gpus[$fism_index]}"
  fism_log="$fism_output/scheduler_logs/shard_${fism_index}_${fism_stamp}_$$.log"
  nohup env CUDA_VISIBLE_DEVICES="$fism_gpu" "$fism_python" -u "${fism_dir}/search_cf.py" \
    "${fism_common[@]}" --device cuda:0 --num-shards "$fism_workers" \
    --shard-index "$fism_index" --execute > "$fism_log" 2>&1 &
  echo "Started GPU=$fism_gpu shard=$fism_index/$fism_workers PID=$! log=$fism_log"
done
echo "Summary: $fism_python ${fism_dir}/search_cf.py --output $fism_output --summarize"
