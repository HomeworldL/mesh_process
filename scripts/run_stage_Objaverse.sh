#!/usr/bin/env bash
set -euo pipefail

# One-shot pipeline dedicated to Objaverse.
# Usage:
#   bash scripts/run_stage_Objaverse.sh
#   bash scripts/run_stage_Objaverse.sh --subset Daily-Used --sample-n 500 --sample-seed 0 --workers 8

DATASET="Objaverse"
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

echo "[stage-objaverse] dataset=${DATASET}"
echo "[stage-objaverse] ingest download"
DOWNLOAD_CMD=(python src/ingest_assets.py download --source "${DATASET}")
if [[ -n "${SUBSET}" ]]; then
  DOWNLOAD_CMD+=(--subset "${SUBSET}")
fi
if [[ -n "${FORCE_FLAG}" ]]; then
  DOWNLOAD_CMD+=("${FORCE_FLAG}")
fi
"${DOWNLOAD_CMD[@]}"

echo "[stage-objaverse] ingest organize"
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

echo "[stage-objaverse] ingest verify"
python src/ingest_assets.py verify --source "${DATASET}" --check-paths

echo "[stage-objaverse] process meshes"
PROCESS_CMD=(python src/process_meshes.py --dataset "${DATASET}" --workers "${WORKERS}")
if [[ -n "${FORCE_FLAG}" ]]; then
  PROCESS_CMD+=("${FORCE_FLAG}")
fi
"${PROCESS_CMD[@]}"

echo "[stage-objaverse] build descriptions"
BUILD_CMD=(python src/build_object_descriptions.py --dataset "${DATASET}")
if [[ -n "${FORCE_FLAG}" ]]; then
  BUILD_CMD+=("${FORCE_FLAG}")
fi
"${BUILD_CMD[@]}"

echo "[stage-objaverse] done"
