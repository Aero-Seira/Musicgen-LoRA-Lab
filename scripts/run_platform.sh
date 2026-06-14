#!/usr/bin/env bash
set -euo pipefail

# Platform runner for MusicGen-LoRA-Lab.
#
# Common overrides:
#   RAW_GTZAN_DIR=/openbayes/input/input0 bash scripts/run_platform.sh
#   MODEL_NAME=facebook/musicgen-small bash scripts/run_platform.sh
#   SAMPLES_PER_GENRE=1 EPOCHS=1 MAX_TRAIN_SAMPLES=20 MAX_VAL_SAMPLES=10 bash scripts/run_platform.sh
#   RUN_LORA=0 bash scripts/run_platform.sh

MODEL_NAME="${MODEL_NAME:-facebook/musicgen-melody}"
DEVICE="${DEVICE:-auto}"
SAMPLES_PER_GENRE="${SAMPLES_PER_GENRE:-10}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LR="${LR:-1e-4}"
RANK="${RANK:-8}"
ALPHA="${ALPHA:-16}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-}"
RUN_LORA="${RUN_LORA:-1}"
AMP="${AMP:-1}"

if [[ -z "${RAW_GTZAN_DIR:-}" ]]; then
  if [[ -d "/openbayes/input/input0" ]]; then
    RAW_GTZAN_DIR="/openbayes/input/input0"
  else
    RAW_GTZAN_DIR="data/raw/gtzan"
  fi
fi
PROCESSED_DIR="${PROCESSED_DIR:-data/processed}"

echo "Model: ${MODEL_NAME}"
echo "Raw GTZAN dir: ${RAW_GTZAN_DIR}"
echo "Processed dir: ${PROCESSED_DIR}"

echo "[1/7] Preprocess GTZAN"
python scripts/preprocess_gtzan.py \
  --raw-dir "${RAW_GTZAN_DIR}" \
  --output-dir "${PROCESSED_DIR}"

echo "[2/7] Generate pretrained baseline"
python scripts/generate_baseline.py \
  --data-dir "${PROCESSED_DIR}/test" \
  --model-name "${MODEL_NAME}" \
  --device "${DEVICE}" \
  --samples-per-genre "${SAMPLES_PER_GENRE}" \
  --skip-existing

echo "[3/7] Evaluate baseline"
python scripts/evaluate_audio.py \
  --real-dir "${PROCESSED_DIR}/test" \
  --generated-dirs outputs/baseline \
  --output outputs/metrics/baseline_results.json \
  --csv-output outputs/metrics/baseline_results.csv

if [[ "${RUN_LORA}" != "1" ]]; then
  echo "RUN_LORA=${RUN_LORA}; stopping after baseline."
  exit 0
fi

TRAIN_ARGS=(
  --model-name "${MODEL_NAME}"
  --train-dir "${PROCESSED_DIR}/train"
  --val-dir "${PROCESSED_DIR}/val"
  --device "${DEVICE}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --lr "${LR}"
  --rank "${RANK}"
  --alpha "${ALPHA}"
)

if [[ -n "${MAX_TRAIN_SAMPLES}" ]]; then
  TRAIN_ARGS+=(--max-train-samples "${MAX_TRAIN_SAMPLES}")
fi
if [[ -n "${MAX_VAL_SAMPLES}" ]]; then
  TRAIN_ARGS+=(--max-val-samples "${MAX_VAL_SAMPLES}")
fi
if [[ "${AMP}" == "1" ]]; then
  TRAIN_ARGS+=(--amp)
fi

echo "[4/7] Train LoRA"
python scripts/train_lora.py "${TRAIN_ARGS[@]}"

echo "[5/7] Generate LoRA continuations"
python scripts/generate_lora.py \
  --model-name "${MODEL_NAME}" \
  --adapter-path outputs/lora/lora_adapter \
  --data-dir "${PROCESSED_DIR}/test" \
  --device "${DEVICE}" \
  --samples-per-genre "${SAMPLES_PER_GENRE}" \
  --skip-existing

echo "[6/7] Evaluate baseline vs LoRA"
python scripts/evaluate_audio.py \
  --real-dir "${PROCESSED_DIR}/test" \
  --generated-dirs outputs/baseline outputs/lora \
  --output outputs/metrics/baseline_vs_lora.json \
  --csv-output outputs/metrics/baseline_vs_lora.csv

echo "[7/7] Build listening test"
python scripts/make_listening_test.py \
  --baseline-dir outputs/baseline \
  --lora-dir outputs/lora \
  --output reports/listening_test.html \
  --mapping-output reports/listening_test_mapping.json \
  --n-samples 10

echo "Done."
