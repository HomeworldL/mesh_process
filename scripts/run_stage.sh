#!/usr/bin/env bash
set -euo pipefail

# One-shot pipeline for regular datasets: Stage-1 ingest -> Stage-2 process -> Stage-3 descriptions
# Usage:
#   bash scripts/run_stage.sh YCB
#   bash scripts/run_stage.sh MSO --workers 8 --force

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <DATASET> [--workers N] [--force]"
  exit 1
fi

DATASET="$1"
shift

WORKERS="8"
FORCE_FLAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workers)
      WORKERS="${2:-8}"
      shift 2
      ;;
    --force)
      FORCE_FLAG="--force"
      shift
      ;;
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
done

echo "[stage] dataset=${DATASET}"
echo "[stage] ingest download"
DOWNLOAD_CMD=(python src/ingest_assets.py download --source "${DATASET}")
if [[ -n "${FORCE_FLAG}" ]]; then
  DOWNLOAD_CMD+=("${FORCE_FLAG}")
fi
"${DOWNLOAD_CMD[@]}"

echo "[stage] ingest organize"
ORGANIZE_CMD=(python src/ingest_assets.py organize --source "${DATASET}")
if [[ -n "${FORCE_FLAG}" ]]; then
  ORGANIZE_CMD+=("${FORCE_FLAG}")
fi
"${ORGANIZE_CMD[@]}"

echo "[stage] ingest verify"
python src/ingest_assets.py verify --source "${DATASET}" --check-paths

echo "[stage] process meshes"
PROCESS_CMD=(python src/process_meshes.py --dataset "${DATASET}" --workers "${WORKERS}")
if [[ -n "${FORCE_FLAG}" ]]; then
  PROCESS_CMD+=("${FORCE_FLAG}")
fi
"${PROCESS_CMD[@]}"

echo "[stage] build descriptions"
BUILD_CMD=(python src/build_object_descriptions.py --dataset "${DATASET}")
if [[ -n "${FORCE_FLAG}" ]]; then
  BUILD_CMD+=("${FORCE_FLAG}")
fi
"${BUILD_CMD[@]}"

echo "[stage] done"
