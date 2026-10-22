#!/usr/bin/env bash
# Usage: bash start_alpha_baby_steam.sh 0,1,2,3 weights/alpha_baby_steam_v2
# Independent jobs on exactly four GPUs, not single-model distributed training.
set -euo pipefail
alpha_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
alpha_gpu_csv="${1:-0,1,2,3}"
alpha_output="${2:-${alpha_dir}/weights/alpha_baby_steam_v2}"
alpha_python="${ALPHA_PYTHON:-python}"
if [[ ! "$alpha_gpu_csv" =~ ^[0-9]+(,[0-9]+){3}$ ]]; then
  echo "Specify exactly four distinct GPU IDs, e.g. 0,2,4,5" >&2
  exit 1
fi
IFS=',' read -r -a alpha_gpus <<< "$alpha_gpu_csv"
declare -A alpha_seen=()
for alpha_gpu in "${alpha_gpus[@]}"; do
  if [[ -n "${alpha_seen[$alpha_gpu]:-}" ]]; then
    echo "Duplicate GPU ID: $alpha_gpu" >&2
    exit 1
  fi
  alpha_seen[$alpha_gpu]=1
  CUDA_VISIBLE_DEVICES="$alpha_gpu" "$alpha_python" -c \
    'import torch; assert torch.cuda.is_available() and torch.cuda.device_count() == 1, "GPU unavailable"; torch.zeros(1, device="cuda:0")'
done
alpha_common=(--datasets Baby Steam --seed 2025 --num-workers 2 --output "$alpha_output")
"$alpha_python" "${alpha_dir}/search_alphafree2.py" "${alpha_common[@]}" --prepare
mkdir -p "$alpha_output/scheduler_logs"
alpha_stamp="$(date +%Y%m%d_%H%M%S)"
for alpha_index in "${!alpha_gpus[@]}"; do
  alpha_gpu="${alpha_gpus[$alpha_index]}"
  alpha_log="$alpha_output/scheduler_logs/shard_${alpha_index}_${alpha_stamp}_$$.log"
  nohup env CUDA_VISIBLE_DEVICES="$alpha_gpu" "$alpha_python" -u "${alpha_dir}/search_alphafree2.py" \
    "${alpha_common[@]}" --cuda 0 --num-shards 4 --shard-index "$alpha_index" --execute \
    > "$alpha_log" 2>&1 &
  echo "Started GPU=$alpha_gpu shard=$alpha_index/4 PID=$! log=$alpha_log"
done
echo "Summary: $alpha_python ${alpha_dir}/search_alphafree2.py --output $alpha_output --summarize"
