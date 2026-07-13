#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/acdc/run_sde_sequence_posthoc.yaml}"
STAGE="${1:-help}"

run_split() {
  local split="$1"
  local cmd=(
    "${PYTHON_BIN}" scripts/run_sde_sequence_posthoc.py
    --config "${CONFIG}"
    --split "${split}"
  )
  if [[ -n "${START_INDEX:-}" ]]; then
    cmd+=(--start-index "${START_INDEX}")
  fi
  if [[ -n "${LIMIT:-}" ]]; then
    cmd+=(--limit "${LIMIT}")
  fi
  if [[ "${OVERWRITE:-0}" == "1" ]]; then
    cmd+=(--overwrite)
  fi
  "${cmd[@]}"
}

run_ef() {
  : "${DEFORMATION:?Set DEFORMATION to one per-sequence .pt output}"
  : "${REFERENCE_MASK:?Set REFERENCE_MASK to the ED LV segmentation NIfTI}"
  : "${ES_INDEX:?Set ES_INDEX to the ES position inside the cine sequence}"
  : "${EF_OUTPUT:?Set EF_OUTPUT to the output JSON path}"
  "${PYTHON_BIN}" scripts/compute_clinical_metrics.py \
    --deformation "${DEFORMATION}" \
    --reference-mask "${REFERENCE_MASK}" \
    --output "${EF_OUTPUT}" \
    --displacement-key total_displacement \
    --volume-size 16 96 96 \
    --labels 3 \
    --ed-index "${ED_INDEX:-0}" \
    --es-index "${ES_INDEX}" \
    --coverage "${COVERAGE:-0.95}" \
    --roi-mask-crop \
    --roi-mask-margin 0 16 16
}

case "${STAGE}" in
  train) run_split train ;;
  val) run_split val ;;
  test) run_split test ;;
  train-val) run_split train; run_split val ;;
  all) run_split train; run_split val; run_split test ;;
  ef) run_ef ;;
  *)
    echo "Usage: $0 {train|val|test|train-val|all|ef}"
    echo "Optional sequence controls: START_INDEX=0 LIMIT=1 OVERWRITE=1"
    exit 2
    ;;
esac
