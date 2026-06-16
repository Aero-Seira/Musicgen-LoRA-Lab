#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

MODE="${1:-compare}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

MODEL_NAME="${MODEL_NAME:-facebook/musicgen-melody}"
DEVICE="${DEVICE:-auto}"
SAMPLES_PER_GENRE="${SAMPLES_PER_GENRE:-10}"
INPUT_SEC="${INPUT_SEC:-20}"
OUTPUT_SEC="${OUTPUT_SEC:-10}"
TOP_K="${TOP_K:-250}"
TOP_P="${TOP_P:-0.0}"
TEMPERATURE="${TEMPERATURE:-1.0}"
CFG_COEF="${CFG_COEF:-3.0}"
DESCRIPTION="${DESCRIPTION:-}"

PROCESSED_DIR="${PROCESSED_DIR:-data/processed}"
ADAPTER_PATH="${ADAPTER_PATH:-outputs/lora/lora_adapter}"
BASELINE_OUTPUT_DIR="${BASELINE_OUTPUT_DIR:-outputs/infer/baseline}"
LORA_OUTPUT_DIR="${LORA_OUTPUT_DIR:-outputs/infer/lora}"
EXTERNAL_DIR="${EXTERNAL_DIR:-data/raw/external}"
EXTERNAL_OUTPUT_DIR="${EXTERNAL_OUTPUT_DIR:-outputs/infer/external}"
METRICS_OUTPUT="${METRICS_OUTPUT:-outputs/metrics/infer_eval.json}"
METRICS_CSV_OUTPUT="${METRICS_CSV_OUTPUT:-outputs/metrics/infer_eval.csv}"
LISTENING_OUTPUT="${LISTENING_OUTPUT:-reports/infer_listening_test.html}"
LISTENING_MAPPING_OUTPUT="${LISTENING_MAPPING_OUTPUT:-reports/infer_listening_test_mapping.json}"
LISTENING_SAMPLES="${LISTENING_SAMPLES:-10}"
MAKE_LISTENING="${MAKE_LISTENING:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

usage() {
  cat <<'EOF'
Usage:
  ./infer_eval.sh [compare|lora|baseline|evaluate|external]

Modes:
  compare   Generate baseline + LoRA on processed test set, then evaluate both. Default.
  lora      Generate LoRA outputs on processed test set, then evaluate LoRA only.
  baseline  Generate pretrained baseline outputs, then evaluate baseline only.
  evaluate  Evaluate existing generated outputs without running inference.
  external  Run LoRA inference on wav files in data/raw/external; no GTZAN metrics.

Common overrides:
  DEVICE=cuda ./infer_eval.sh compare
  SAMPLES_PER_GENRE=3 ./infer_eval.sh compare
  ADAPTER_PATH=outputs/lora/lora_adapter ./infer_eval.sh lora
  EXTERNAL_DIR=data/raw/external EXTERNAL_OUTPUT_DIR=outputs/demo ./infer_eval.sh external
  BASELINE_OUTPUT_DIR=outputs/baseline LORA_OUTPUT_DIR=outputs/lora ./infer_eval.sh evaluate
  MAKE_LISTENING=0 ./infer_eval.sh compare
EOF
}

if [[ "${MODE}" == "-h" || "${MODE}" == "--help" ]]; then
  usage
  exit 0
fi

case "${MODE}" in
  compare|lora|baseline|evaluate|external) ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    usage >&2
    exit 2
    ;;
esac

require_uv() {
  if ! command -v uv >/dev/null 2>&1; then
    cat >&2 <<'EOF'
Error: uv is not installed.

Install it first:
  curl -LsSf https://astral.sh/uv/install.sh | sh
EOF
    exit 1
  fi
}

ensure_python_and_deps() {
  echo "[setup] Ensure Python ${PYTHON_VERSION} is available through uv"
  if uv python find "${PYTHON_VERSION}" >/dev/null 2>&1; then
    uv python find "${PYTHON_VERSION}"
  else
    uv python install "${PYTHON_VERSION}"
  fi

  echo "[setup] Sync dependencies with uv"
  uv sync --python "${PYTHON_VERSION}" --locked
}

resolve_raw_gtzan_dir() {
  if [[ -n "${RAW_GTZAN_DIR:-}" ]]; then
    echo "${RAW_GTZAN_DIR}"
  elif [[ -d "/openbayes/input/input0" ]]; then
    echo "/openbayes/input/input0"
  else
    echo "data/raw/gtzan"
  fi
}

has_processed_test_wavs() {
  [[ -d "${PROCESSED_DIR}/test" ]] && \
    find "${PROCESSED_DIR}/test" -mindepth 2 -maxdepth 2 -type f -name "*.wav" ! -name "._*" -print -quit | grep -q .
}

ensure_processed_test() {
  if has_processed_test_wavs; then
    echo "[data] Reuse processed test set: ${PROCESSED_DIR}/test"
    return
  fi

  local raw_dir
  raw_dir="$(resolve_raw_gtzan_dir)"
  if [[ ! -d "${raw_dir}" ]]; then
    cat >&2 <<EOF
Error: processed test set does not exist and raw GTZAN dir was not found.

Expected processed dir:
  ${PROCESSED_DIR}/test

Expected raw dir:
  ${raw_dir}

Set RAW_GTZAN_DIR=/path/to/gtzan and rerun.
EOF
    exit 1
  fi

  echo "[data] Preprocess GTZAN"
  echo "Raw GTZAN dir: ${raw_dir}"
  echo "Processed dir: ${PROCESSED_DIR}"
  uv run --python "${PYTHON_VERSION}" python scripts/preprocess_gtzan.py \
    --raw-dir "${raw_dir}" \
    --output-dir "${PROCESSED_DIR}"
}

require_adapter() {
  if [[ ! -f "${ADAPTER_PATH}/adapter_model.safetensors" || ! -f "${ADAPTER_PATH}/adapter_config.json" ]]; then
    cat >&2 <<EOF
Error: LoRA adapter is incomplete or missing:
  ${ADAPTER_PATH}

Expected:
  ${ADAPTER_PATH}/adapter_model.safetensors
  ${ADAPTER_PATH}/adapter_config.json
EOF
    exit 1
  fi
}

skip_flag() {
  if [[ "${SKIP_EXISTING}" == "1" ]]; then
    echo "--skip-existing"
  fi
}

generate_baseline() {
  echo "[infer] Generate pretrained baseline -> ${BASELINE_OUTPUT_DIR}"
  uv run --python "${PYTHON_VERSION}" python scripts/generate_baseline.py \
    --data-dir "${PROCESSED_DIR}/test" \
    --output-dir "${BASELINE_OUTPUT_DIR}" \
    --model-name "${MODEL_NAME}" \
    --samples-per-genre "${SAMPLES_PER_GENRE}" \
    --input-sec "${INPUT_SEC}" \
    --output-sec "${OUTPUT_SEC}" \
    --description "${DESCRIPTION}" \
    --top-k "${TOP_K}" \
    --top-p "${TOP_P}" \
    --temperature "${TEMPERATURE}" \
    --cfg-coef "${CFG_COEF}" \
    --device "${DEVICE}" \
    $(skip_flag)
}

generate_lora() {
  require_adapter
  echo "[infer] Generate LoRA continuations -> ${LORA_OUTPUT_DIR}"
  uv run --python "${PYTHON_VERSION}" python scripts/generate_lora.py \
    --adapter-path "${ADAPTER_PATH}" \
    --model-name "${MODEL_NAME}" \
    --data-dir "${PROCESSED_DIR}/test" \
    --output-lora "${LORA_OUTPUT_DIR}" \
    --external-dir "__no_external_dir__" \
    --samples-per-genre "${SAMPLES_PER_GENRE}" \
    --input-sec "${INPUT_SEC}" \
    --output-sec "${OUTPUT_SEC}" \
    --description "${DESCRIPTION}" \
    --top-k "${TOP_K}" \
    --top-p "${TOP_P}" \
    --temperature "${TEMPERATURE}" \
    --cfg-coef "${CFG_COEF}" \
    --device "${DEVICE}" \
    $(skip_flag)
}

generate_external() {
  require_adapter
  if [[ ! -d "${EXTERNAL_DIR}" ]]; then
    cat >&2 <<EOF
Error: external audio dir does not exist:
  ${EXTERNAL_DIR}

Create it and put wav files inside, or set EXTERNAL_DIR=/path/to/wavs.
EOF
    exit 1
  fi

  echo "[infer] Generate LoRA external continuations"
  echo "External input: ${EXTERNAL_DIR}"
  echo "External output: ${EXTERNAL_OUTPUT_DIR}"
  uv run --python "${PYTHON_VERSION}" python scripts/generate_lora.py \
    --adapter-path "${ADAPTER_PATH}" \
    --model-name "${MODEL_NAME}" \
    --data-dir "__no_gtzan_data__" \
    --external-dir "${EXTERNAL_DIR}" \
    --output-lora "${LORA_OUTPUT_DIR}" \
    --output-external "${EXTERNAL_OUTPUT_DIR}" \
    --input-sec "${INPUT_SEC}" \
    --output-sec "${OUTPUT_SEC}" \
    --description "${DESCRIPTION}" \
    --top-k "${TOP_K}" \
    --top-p "${TOP_P}" \
    --temperature "${TEMPERATURE}" \
    --cfg-coef "${CFG_COEF}" \
    --device "${DEVICE}" \
    $(skip_flag)
}

evaluate_dirs() {
  local dirs=("$@")
  if [[ "${#dirs[@]}" -eq 0 ]]; then
    echo "[eval] No generated dirs to evaluate; skipped."
    return
  fi

  echo "[eval] Evaluate generated outputs"
  uv run --python "${PYTHON_VERSION}" python scripts/evaluate_audio.py \
    --real-dir "${PROCESSED_DIR}/test" \
    --generated-dirs "${dirs[@]}" \
    --output "${METRICS_OUTPUT}" \
    --csv-output "${METRICS_CSV_OUTPUT}" \
    --input-sec "${INPUT_SEC}" \
    --target-sec "${OUTPUT_SEC}"
}

build_listening_test() {
  if [[ "${MAKE_LISTENING}" != "1" ]]; then
    echo "[listen] MAKE_LISTENING=${MAKE_LISTENING}; skipped."
    return
  fi
  if [[ ! -d "${BASELINE_OUTPUT_DIR}" || ! -d "${LORA_OUTPUT_DIR}" ]]; then
    echo "[listen] Baseline or LoRA output missing; skipped."
    return
  fi

  echo "[listen] Build A/B listening page -> ${LISTENING_OUTPUT}"
  uv run --python "${PYTHON_VERSION}" python scripts/make_listening_test.py \
    --baseline-dir "${BASELINE_OUTPUT_DIR}" \
    --lora-dir "${LORA_OUTPUT_DIR}" \
    --output "${LISTENING_OUTPUT}" \
    --mapping-output "${LISTENING_MAPPING_OUTPUT}" \
    --n-samples "${LISTENING_SAMPLES}"
}

echo "Mode: ${MODE}"
echo "Python: ${PYTHON_VERSION}"
echo "Model: ${MODEL_NAME}"
echo "Device: ${DEVICE}"
echo "Samples per genre: ${SAMPLES_PER_GENRE}"

require_uv
ensure_python_and_deps

case "${MODE}" in
  compare)
    ensure_processed_test
    generate_baseline
    generate_lora
    evaluate_dirs "${BASELINE_OUTPUT_DIR}" "${LORA_OUTPUT_DIR}"
    build_listening_test
    ;;
  lora)
    ensure_processed_test
    generate_lora
    evaluate_dirs "${LORA_OUTPUT_DIR}"
    ;;
  baseline)
    ensure_processed_test
    generate_baseline
    evaluate_dirs "${BASELINE_OUTPUT_DIR}"
    ;;
  evaluate)
    ensure_processed_test
    EVAL_DIRS=()
    [[ -d "${BASELINE_OUTPUT_DIR}" ]] && EVAL_DIRS+=("${BASELINE_OUTPUT_DIR}")
    [[ -d "${LORA_OUTPUT_DIR}" ]] && EVAL_DIRS+=("${LORA_OUTPUT_DIR}")
    evaluate_dirs "${EVAL_DIRS[@]}"
    build_listening_test
    ;;
  external)
    generate_external
    ;;
esac

echo "Done."
