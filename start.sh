#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

MODE="${1:-full}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

usage() {
  cat <<'EOF'
Usage:
  ./start.sh [smoke|baseline|full]

Modes:
  smoke     Small-sample end-to-end run: Melody model, 1 sample/genre, 1 epoch.
  baseline  Pretrained baseline only, no LoRA training.
  full      Full configured workflow. This is the default.

Common overrides:
  MODEL_NAME=facebook/musicgen-small ./start.sh full
  RAW_GTZAN_DIR=/openbayes/input/input0 ./start.sh full
  SAMPLES_PER_GENRE=5 EPOCHS=2 ./start.sh full
  PYTHON_VERSION=3.10 ./start.sh smoke
EOF
}

if [[ "${MODE}" == "-h" || "${MODE}" == "--help" ]]; then
  usage
  exit 0
fi

if ! command -v uv >/dev/null 2>&1; then
  cat >&2 <<'EOF'
Error: uv is not installed.

Install it first:
  curl -LsSf https://astral.sh/uv/install.sh | sh

Then restart the shell and run:
  ./start.sh smoke
EOF
  exit 1
fi

case "${MODE}" in
  smoke)
    export MODEL_NAME="${MODEL_NAME:-facebook/musicgen-melody}"
    export SAMPLES_PER_GENRE="${SAMPLES_PER_GENRE:-1}"
    export EPOCHS="${EPOCHS:-1}"
    export BATCH_SIZE="${BATCH_SIZE:-1}"
    export MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-20}"
    export MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-10}"
    export RUN_LORA="${RUN_LORA:-1}"
    ;;
  baseline)
    export RUN_LORA=0
    export SAMPLES_PER_GENRE="${SAMPLES_PER_GENRE:-10}"
    ;;
  full)
    export RUN_LORA="${RUN_LORA:-1}"
    export SAMPLES_PER_GENRE="${SAMPLES_PER_GENRE:-10}"
    export EPOCHS="${EPOCHS:-3}"
    export BATCH_SIZE="${BATCH_SIZE:-1}"
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    usage >&2
    exit 2
    ;;
esac

echo "Mode: ${MODE}"
echo "Python: ${PYTHON_VERSION}"
echo "Model: ${MODEL_NAME:-facebook/musicgen-melody}"
echo "Samples per genre: ${SAMPLES_PER_GENRE}"
echo "Run LoRA: ${RUN_LORA}"

echo "[setup] Ensure Python ${PYTHON_VERSION} is available through uv"
if uv python find "${PYTHON_VERSION}" >/dev/null 2>&1; then
  uv python find "${PYTHON_VERSION}"
else
  uv python install "${PYTHON_VERSION}"
fi

echo "[setup] Sync dependencies with uv"
uv sync --python "${PYTHON_VERSION}" --locked --reinstall-package xformers

echo "[run] Start experiment pipeline"
uv run --python "${PYTHON_VERSION}" bash scripts/run_platform.sh
