#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
CONFIG="${CONFIG:-configs/acdc/train_sde_rnn_uncertainty.yaml}"
CHECKPOINT="${CHECKPOINT:-runs/acdc/sde_rnn/best.pt}"
ROOT_OUT="${ROOT_OUT:-runs/acdc/clinical}"
TRAIN_VAL_H5="${TRAIN_VAL_H5:-runs/acdc/latent_train_val.h5}"
TEST_H5="${TEST_H5:-runs/acdc/latent_test.h5}"
TRAIN_VAL_INDEX="${TRAIN_VAL_INDEX:-runs/acdc/sde_sequence_index_train_val.jsonl}"
TEST_INDEX="${TEST_INDEX:-runs/acdc/sde_sequence_index_test.jsonl}"
TRAIN_SUMMARY="${TRAIN_SUMMARY:-runs/acdc/nodeo_dir/train/summary.jsonl}"
VAL_SUMMARY="${VAL_SUMMARY:-runs/acdc/nodeo_dir/val/summary.jsonl}"
TEST_SUMMARY="${TEST_SUMMARY:-runs/acdc/nodeo_dir/test/summary.jsonl}"
TRAIN_MANIFEST="${ROOT_OUT}/train_targets.jsonl"
VAL_MANIFEST="${ROOT_OUT}/val_targets.jsonl"
TEST_MANIFEST="${ROOT_OUT}/test_targets.jsonl"

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
  local chart_dir="${5:-}"
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
    "${cmd[@]}"
    if [[ -n "${chart_dir}" ]]; then
      "${PYTHON_BIN}" scripts/plot_clinical_prediction_bands.py \
        --input "${prediction}" \
        --output "${chart_dir}/sequence${sequence_index}.png"
    fi
  done < <(manifest_rows "${manifest}")
}

mkdir -p "${ROOT_OUT}"

"${PYTHON_BIN}" scripts/prepare_acdc_clinical_targets.py \
  --config "${CONFIG}" \
  --sequence-index-file "${TRAIN_VAL_INDEX}" \
  --h5 "${TRAIN_VAL_H5}" \
  --split train \
  --output "${TRAIN_MANIFEST}" \
  --label 3

"${PYTHON_BIN}" scripts/prepare_acdc_clinical_targets.py \
  --config "${CONFIG}" \
  --sequence-index-file "${TRAIN_VAL_INDEX}" \
  --h5 "${TRAIN_VAL_H5}" \
  --split val \
  --output "${VAL_MANIFEST}" \
  --label 3

"${PYTHON_BIN}" scripts/prepare_acdc_clinical_targets.py \
  --config "${CONFIG}" \
  --sequence-index-file "${TEST_INDEX}" \
  --h5 "${TEST_H5}" \
  --split test \
  --output "${TEST_MANIFEST}" \
  --label 3

process_manifest \
  "${TRAIN_MANIFEST}" "${TRAIN_VAL_H5}" "${TRAIN_SUMMARY}" \
  "${ROOT_OUT}/train_predictions"

process_manifest \
  "${VAL_MANIFEST}" "${TRAIN_VAL_H5}" "${VAL_SUMMARY}" \
  "${ROOT_OUT}/val_predictions"

process_manifest \
  "${TEST_MANIFEST}" "${TEST_H5}" "${TEST_SUMMARY}" \
  "${ROOT_OUT}/test_predictions" "${ROOT_OUT}/test_charts"

"${PYTHON_BIN}" scripts/evaluate_clinical_prediction_bands.py \
  --predictions "${ROOT_OUT}/test_predictions/*.json" \
  --targets "${TEST_MANIFEST}" \
  --output "${ROOT_OUT}/test_evaluation_summary.json"

echo "Independent ACDC test (direct model uncertainty): ${ROOT_OUT}/test_evaluation_summary.json"
echo "Clinical train features: ${ROOT_OUT}/train_predictions"
echo "Clinical validation features: ${ROOT_OUT}/val_predictions"
echo "Charts: ${ROOT_OUT}/test_charts"
