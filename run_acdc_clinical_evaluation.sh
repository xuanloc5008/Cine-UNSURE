#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
CONFIG="${CONFIG:-configs/acdc/train_sde_rnn_uncertainty.yaml}"
CHECKPOINT="${CHECKPOINT:-runs/acdc/sde_rnn/best.pt}"
ROOT_OUT="${ROOT_OUT:-runs/acdc/clinical}"
VAL_H5="${VAL_H5:-runs/acdc/latent_train_val.h5}"
TEST_H5="${TEST_H5:-runs/acdc/latent_test.h5}"
VAL_INDEX="${VAL_INDEX:-runs/acdc/sde_sequence_index_train_val.jsonl}"
TEST_INDEX="${TEST_INDEX:-runs/acdc/sde_sequence_index_test.jsonl}"
VAL_SUMMARY="${VAL_SUMMARY:-runs/acdc/nodeo_dir/val/summary.jsonl}"
TEST_SUMMARY="${TEST_SUMMARY:-runs/acdc/nodeo_dir/test/summary.jsonl}"
CALIBRATION_MANIFEST="${ROOT_OUT}/calibration_targets.jsonl"
TEST_MANIFEST="${ROOT_OUT}/test_targets.jsonl"
CALIBRATION_FILE="${ROOT_OUT}/calibration_95.json"

manifest_rows() {
  "${PYTHON_BIN}" - "$1" <<'PY'
import json
import sys
for line in open(sys.argv[1], encoding="utf-8"):
    if line.strip():
        row = json.loads(line)
        print(row["sequence_index"], row["reference_mask"], row["ed_index"], row["es_index"], sep="\t")
PY
}

process_manifest() {
  local manifest="$1"
  local h5="$2"
  local summary="$3"
  local prediction_dir="$4"
  local calibration_file="${5:-}"
  local chart_dir="${6:-}"
  local deformation_dir="${prediction_dir}_deformations"
  mkdir -p "${prediction_dir}" "${deformation_dir}"
  [[ -z "${chart_dir}" ]] || mkdir -p "${chart_dir}"

  while IFS=$'\t' read -r sequence_index reference_mask ed_index es_index; do
    local deformation="${deformation_dir}/sequence${sequence_index}.pt"
    local prediction="${prediction_dir}/sequence${sequence_index}.json"
    if [[ ! -f "${deformation}" ]]; then
      "${PYTHON_BIN}" scripts/infer_sde_rnn_uncertainty.py \
        --checkpoint "${CHECKPOINT}" \
        --h5 "${h5}" \
        --nodeo-summary "${summary}" \
        --sequence-index "${sequence_index}" \
        --output "${deformation}" \
        --device "${DEVICE}"
    fi
    local cmd=(
      "${PYTHON_BIN}" scripts/compute_clinical_metrics.py
      --deformation "${deformation}"
      --reference-mask "${reference_mask}"
      --volume-size 16 96 96
      --labels 3
      --roi-mask-crop
      --roi-mask-margin 0 16 16
      --ed-index "${ed_index}"
      --es-index "${es_index}"
      --coverage 0.95
      --output "${prediction}"
    )
    [[ -z "${calibration_file}" ]] || cmd+=(--calibration "${calibration_file}")
    "${cmd[@]}"
    if [[ -n "${chart_dir}" ]]; then
      "${PYTHON_BIN}" scripts/plot_clinical_prediction_bands.py \
        --input "${prediction}" \
        --output "${chart_dir}/sequence${sequence_index}.png"
    fi
  done < <(manifest_rows "${manifest}")
}

mkdir -p "${ROOT_OUT}"

"${PYTHON_BIN}" scripts/prepare_acdc_pilot_calibration.py \
  --config "${CONFIG}" \
  --sequence-index-file "${VAL_INDEX}" \
  --h5 "${VAL_H5}" \
  --split val \
  --output "${CALIBRATION_MANIFEST}" \
  --label 3

"${PYTHON_BIN}" scripts/prepare_acdc_pilot_calibration.py \
  --config "${CONFIG}" \
  --sequence-index-file "${TEST_INDEX}" \
  --h5 "${TEST_H5}" \
  --split test \
  --output "${TEST_MANIFEST}" \
  --label 3

process_manifest "${CALIBRATION_MANIFEST}" "${VAL_H5}" "${VAL_SUMMARY}" "${ROOT_OUT}/calibration_predictions"

"${PYTHON_BIN}" scripts/calibrate_clinical_bands.py \
  --predictions "${ROOT_OUT}/calibration_predictions/*.json" \
  --targets "${CALIBRATION_MANIFEST}" \
  --coverage 0.95 \
  --output "${CALIBRATION_FILE}"

process_manifest \
  "${TEST_MANIFEST}" "${TEST_H5}" "${TEST_SUMMARY}" \
  "${ROOT_OUT}/test_predictions" "${CALIBRATION_FILE}" "${ROOT_OUT}/test_charts"

"${PYTHON_BIN}" scripts/evaluate_clinical_prediction_bands.py \
  --predictions "${ROOT_OUT}/test_predictions/*.json" \
  --targets "${TEST_MANIFEST}" \
  --output "${ROOT_OUT}/test_evaluation_summary.json"

echo "Calibration: ${CALIBRATION_FILE}"
echo "Independent ACDC test: ${ROOT_OUT}/test_evaluation_summary.json"
echo "Charts: ${ROOT_OUT}/test_charts"
