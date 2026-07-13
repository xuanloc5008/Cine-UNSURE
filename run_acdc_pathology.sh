#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
STAGE="${1:-all}"
CONFIG="${CONFIG:-configs/acdc/train_pathology_classifier.yaml}"
CHECKPOINT="${CHECKPOINT:-runs/acdc/pathology/best.pt}"
TEST_CLINICAL="${TEST_CLINICAL:-runs/acdc/clinical/test_predictions/*.json}"
TEST_OUTPUT="${TEST_OUTPUT:-runs/acdc/pathology/test_predictions}"

run_train() {
  "${PYTHON_BIN}" scripts/train_pathology_classifier.py --config "${CONFIG}"
}

run_infer() {
  "${PYTHON_BIN}" scripts/infer_pathology_classifier.py \
    --checkpoint "${CHECKPOINT}" \
    --clinical "${TEST_CLINICAL}" \
    --output-dir "${TEST_OUTPUT}" \
    --device "${DEVICE}" \
    --coverage 0.95
}

run_evaluate() {
  "${PYTHON_BIN}" scripts/evaluate_pathology_classifier.py \
    --predictions "${TEST_OUTPUT}/*.json" \
    --output runs/acdc/pathology/test_evaluation.json
}

case "${STAGE}" in
  train) run_train ;;
  infer) run_infer ;;
  evaluate) run_evaluate ;;
  all) run_train; run_infer; run_evaluate ;;
  *) echo "Usage: $0 {train|infer|evaluate|all}"; exit 2 ;;
esac
