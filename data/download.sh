#!/bin/bash
set -euo pipefail

# -----------------------------
# Dependencies
# -----------------------------
PYTHON="${PYTHON:-python}"
if ! command -v "$PYTHON" &> /dev/null; then
  echo "Error: Python not found: $PYTHON. Activate your Python environment first." >&2
  exit 1
fi
run_gdown() {
  "$PYTHON" -c 'from gdown.cli import main; main()' "$@"
}
if ! run_gdown --help >/dev/null 2>&1; then
  echo "gdown is unavailable in the selected Python. Installing..."
  "$PYTHON" -m pip install gdown
  run_gdown --help >/dev/null
fi
if ! command -v unzip &> /dev/null; then
  echo "Error: unzip not found. Please install unzip first."
  exit 1
fi

# -----------------------------
# Paths
# -----------------------------
DATA_DIR="./"
WEIGHT_DIR="./pretrained"
mkdir -p "$DATA_DIR" "$WEIGHT_DIR"

# -----------------------------
# Google Drive IDs
#   DATA_ZIPS: dataset zip
#   WEIGHTS  : pretrained .pth.tar
# -----------------------------
declare -A DATA_ZIPS
DATA_ZIPS["Steam"]="1abykeNcnxAJpqZB1mYM9bg9MI3pehSN4"
DATA_ZIPS["Movie"]="1Wpl3xJUlVhPPyvbWX81e8RLfaYFBOk8_"
DATA_ZIPS["Book"]="1FME3AMNpk7EMgmGiFOZa67AKupR-5QuF"
DATA_ZIPS["Video"]="1fslBxD3kcP58eylGAsjRoy9cOOn1S7uI"
DATA_ZIPS["Baby"]="15KQtrujfR-SjMU4NzVOCSSayNfmtJjDu"
DATA_ZIPS["Beauty"]="1sVMNdBgdK7uAdAqY8c1rc9ds9ywyFhX9"
DATA_ZIPS["Health"]="14F5QSXSNKrpnTeKPx757Xx04j1jvt-V7"

declare -A WEIGHTS
WEIGHTS["Movie"]="125c6BeG-AVPaFO519aW_PkDni9Dp6-IZ"
WEIGHTS["Book"]="1Y18ZFq9yMscdl3Sc395SlnF37dBAKlUz"
WEIGHTS["Video"]="1k-Mi3OjadXL4zL6_aN2UjhPUPwMHH7tj"
WEIGHTS["Baby"]="1E8-cZILoU-5o62kY5edBV84ZkzK3A3IE"
WEIGHTS["Steam"]="1sEh_WbFm5mA7gTBtFuvkS5AmkPvRDpdr"
WEIGHTS["Beauty"]="1bNRKzE_s4SXpDF3qWl7ox4YIXlpcw1M7"
WEIGHTS["Health"]="1LScqWmIu7ZB7tByxG_OB3UfNMsvmU5u3"

# Canonical folder names used after extraction (only folder name is changed)
declare -A CANON
CANON["Steam"]="steam"
CANON["Movie"]="amazon_movie"
CANON["Book"]="amazon_book_2014"
CANON["Video"]="amazon_video"
CANON["Baby"]="amazon_baby"
CANON["Beauty"]="amazon_beauty_personal"
CANON["Health"]="amazon_health"

ALL_DATASETS=(
  Steam
  Movie
  Book
  Video
  Baby
  Beauty
  Health
)

# -----------------------------
# Args
# -----------------------------
SELECTED_DATASETS=()
FORCE_ALL=false
JOBS=3  

usage() {
  echo "Usage:"
  echo "  $0                    # download ALL (parallel)"
  echo "  $0 --all              # download ALL (parallel)"
  echo "  $0 --dataset A,B      # download selected only (parallel)"
  echo "  $0 --jobs N           # parallelism (default: 3)"
  echo ""
  echo "Available datasets: ${ALL_DATASETS[*]}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)
      FORCE_ALL=true
      shift
      ;;
    --dataset)
      IFS=',' read -r -a SELECTED_DATASETS <<< "${2:-}"
      shift 2
      ;;
    --jobs)
      JOBS="${2:-3}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown parameter: $1"
      usage
      exit 1
      ;;
  esac
done

if ! [[ "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: --jobs must be a positive integer." >&2
  exit 1
fi

# -----------------------------
# Helpers
# -----------------------------
download_weight() {
  local NAME="$1"
  local FILE_ID="${WEIGHTS[$NAME]:-}"
  
  local CAN_NAME="${CANON[$NAME]:-}"
  if [[ -z "$CAN_NAME" ]]; then
    CAN_NAME=$(echo "$NAME" | tr '[:upper:]' '[:lower:]')
  fi

  local OUT_PATH="${WEIGHT_DIR}/${CAN_NAME}.pth.tar"

  if [[ -z "$FILE_ID" ]]; then
    echo "[WARN] No weight id for '$NAME' (skip weight)"
    return 0
  fi
  if [[ -f "$OUT_PATH" ]]; then
    echo "[SKIP] weight exists: $OUT_PATH"
    return 0
  fi

  echo "  -> (W) Downloading weight: $NAME as ${CAN_NAME}.pth.tar"
  run_gdown "https://drive.google.com/uc?id=$FILE_ID" -O "${OUT_PATH}.part" || return 1
  mv "${OUT_PATH}.part" "$OUT_PATH" || return 1
  echo "  -> (W) Saved: $OUT_PATH"
}

download_dataset() {
  local NAME="$1"
  local FILE_ID="${DATA_ZIPS[$NAME]:-}"

  # Use canonical folder name for the extracted dataset directory
  local CAN="${CANON[$NAME]:-}"
  local OUT_DIR="${DATA_DIR}/${CAN}"

  if [[ -z "$FILE_ID" ]]; then
    echo "Error: dataset '$NAME' not in DATA_ZIPS"
    exit 1
  fi
  if [[ -z "$CAN" ]]; then
    echo "Error: canonical folder name not found for '$NAME'"
    exit 1
  fi

  # Skip if already extracted under the canonical folder name
  if [[ -d "$OUT_DIR" ]]; then
    echo "[SKIP] dataset exists: $OUT_DIR"
    return 0
  fi

  local TMP_DIR
  TMP_DIR="$(mktemp -d)"
  local ZIP_PATH="${TMP_DIR}/${NAME}.zip"

  echo "  -> (D) Downloading dataset zip: $NAME"
  run_gdown "https://drive.google.com/uc?id=$FILE_ID" -O "$ZIP_PATH" || return 1

  echo "  -> (D) Extracting dataset: $NAME"
  unzip -o "$ZIP_PATH" -d "$TMP_DIR/extracted" >/dev/null || return 1

  # Find the extracted top-level directory (whatever its name is)
  local TOP_DIR=""
  TOP_DIR="$(find "$TMP_DIR/extracted" -mindepth 1 -maxdepth 1 -type d | head -n 1 || true)"

  if [[ -z "$TOP_DIR" ]]; then
    echo "Error: cannot find extracted folder for $NAME"
    rm -rf "$TMP_DIR"
    exit 1
  fi

  # Move it to the canonical folder name
  mv "$TOP_DIR" "$OUT_DIR" || return 1

  rm -rf "$TMP_DIR"
  echo "  -> (D) Ready: $OUT_DIR"
}

download_both_one_dataset() {
  local NAME="$1"
  echo "========================================"
  echo "[DATA+WEIGHT] $NAME"

  # data/weight pararrel
  download_dataset "$NAME" &
  local pid_d=$!
  download_weight "$NAME" &
  local pid_w=$!

  local failed=0
  wait "$pid_d" || failed=1
  wait "$pid_w" || failed=1
  if [[ "$failed" -ne 0 ]]; then
    echo "[FAILED] $NAME: data or weight download failed." >&2
    return 1
  fi

  echo "[DONE] $NAME"
}

# -----------------------------
# Build target list
# -----------------------------
TARGETS=()
if $FORCE_ALL || [[ ${#SELECTED_DATASETS[@]} -eq 0 ]]; then
  TARGETS=("${ALL_DATASETS[@]}")
else
  TARGETS=("${SELECTED_DATASETS[@]}")
fi

echo "Parallel jobs: $JOBS"
echo "Targets: ${TARGETS[*]}"

# -----------------------------
# Execute
# -----------------------------
PIDS=()
FAILED=0
for NAME in "${TARGETS[@]}"; do
  download_both_one_dataset "$NAME" &
  PIDS+=("$!")
  if [[ ${#PIDS[@]} -ge "$JOBS" ]]; then
    wait "${PIDS[0]}" || FAILED=1
    PIDS=("${PIDS[@]:1}")
  fi
done

for PID in "${PIDS[@]}"; do
  wait "$PID" || FAILED=1
done

if [[ "$FAILED" -ne 0 ]]; then
  echo "Download failed for one or more datasets. Check the [FAILED] messages above." >&2
  exit 1
fi

echo "✅ Completed (data + weights)."
