#!/usr/bin/env bash
set -euo pipefail

# One-shot pipeline: Stage-1 ingest -> Stage-2 process -> Stage-3 descriptions
# Usage:
#   bash scripts/run_stage123.sh YCB
#   bash scripts/run_stage123.sh Objaverse --subset Daily-Used --sample-n 500 --sample-seed 0 --workers 8

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <DATASET> [--subset <name>] [--sample-n N] [--sample-seed S] [--workers N] [--force]"
  exit 1
fi

DATASET="$1"
shift

SUBSET=""
SAMPLE_N=""
SAMPLE_SEED="0"
WORKERS="8"
FORCE_FLAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --subset)
      SUBSET="${2:-}"
      shift 2
      ;;
    --sample-n)
      SAMPLE_N="${2:-}"
      shift 2
      ;;
    --sample-seed)
      SAMPLE_SEED="${2:-0}"
      shift 2
      ;;
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

echo "[stage123] dataset=${DATASET}"
echo "[stage123] ingest download"
DOWNLOAD_CMD=(python src/ingest_assets.py download --source "${DATASET}")
if [[ -n "${SUBSET}" ]]; then
  DOWNLOAD_CMD+=(--subset "${SUBSET}")
fi
if [[ -n "${FORCE_FLAG}" ]]; then
  DOWNLOAD_CMD+=("${FORCE_FLAG}")
fi
"${DOWNLOAD_CMD[@]}"

echo "[stage123] ingest organize"
ORGANIZE_CMD=(python src/ingest_assets.py organize --source "${DATASET}")
if [[ -n "${SUBSET}" ]]; then
  ORGANIZE_CMD+=(--subset "${SUBSET}")
fi
if [[ -n "${SAMPLE_N}" ]]; then
  ORGANIZE_CMD+=(--sample-n "${SAMPLE_N}" --sample-seed "${SAMPLE_SEED}")
fi
if [[ -n "${FORCE_FLAG}" ]]; then
  ORGANIZE_CMD+=("${FORCE_FLAG}")
fi
"${ORGANIZE_CMD[@]}"

echo "[stage123] ingest verify"
python src/ingest_assets.py verify --source "${DATASET}" --check-paths

echo "[stage123] process meshes"
PROCESS_CMD=(python src/process_meshes.py --dataset "${DATASET}" --workers "${WORKERS}")
if [[ -n "${FORCE_FLAG}" ]]; then
  PROCESS_CMD+=("${FORCE_FLAG}")
fi
"${PROCESS_CMD[@]}"

echo "[stage123] build descriptions"
BUILD_CMD=(python src/build_object_descriptions.py --dataset "${DATASET}")
if [[ -n "${FORCE_FLAG}" ]]; then
  BUILD_CMD+=("${FORCE_FLAG}")
fi
"${BUILD_CMD[@]}"

echo "[stage123] done"
