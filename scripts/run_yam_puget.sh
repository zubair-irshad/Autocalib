#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -lt 1 ]]; then
  echo 'Usage: bash scripts/run_yam_puget.sh /absolute/path/ABC-130k [pipeline options]'
  exit 2
fi
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
.venv-yam-puget/bin/python -u scripts/run_yam_pipeline.py --data "$1" --device cuda --flow-backend raft "${@:2}"
