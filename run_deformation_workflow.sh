#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"
STAGE="${1:-all}"

PREPARE_CONFIG="${PREPARE_CONFIG:-configs/prepare_hdf5.yaml}"
SCORE_CONFIG="${SCORE_CONFIG:-configs/train_cunsure_score.yaml}"
SCORE_CINEMA_CONFIG="${SCORE_CINEMA_CONFIG:-configs/infer_cinema_score_cunsure_all_datasets.yaml}"
MEAN_CONFIG="${MEAN_CONFIG:-configs/train_nodeo_mean_deformation.yaml}"
SCORE_OUTPUT_DIR="${SCORE_OUTPUT_DIR:-runs/selected/cinema_score_cunsure_roi_all_datasets}"

LATENT_H5="${LATENT_H5:-}"
if [[ -z "${LATENT_H5}" ]]; then
  LATENT_H5="$("${PYTHON_BIN}" - "${MEAN_CONFIG}" <<'PY'
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
print(cfg["data"]["h5"])
PY
)"
fi

SEQUENCE_INDEX_FILE="${SEQUENCE_INDEX_FILE:-}"
if [[ -z "${SEQUENCE_INDEX_FILE}" ]]; then
  SEQUENCE_INDEX_FILE="$("${PYTHON_BIN}" - "${MEAN_CONFIG}" <<'PY'
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
print(cfg["data"]["sequence_index"])
PY
)"
fi

MEAN_RUN_DIR="${MEAN_RUN_DIR:-}"
if [[ -z "${MEAN_RUN_DIR}" ]]; then
  MEAN_RUN_DIR="$("${PYTHON_BIN}" - "${MEAN_CONFIG}" <<'PY'
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
print(cfg["output"]["run_dir"])
PY
)"
fi

MEAN_CHECKPOINT="${MEAN_CHECKPOINT:-${MEAN_RUN_DIR}/best.pt}"
SEQUENCE_ID="${SEQUENCE_ID:-0}"
DEVICE="${DEVICE:-auto}"
COVARIANCE="${COVARIANCE:-diag}"
MEAN_DEFORMATION_OUTPUT="${MEAN_DEFORMATION_OUTPUT:-${MEAN_RUN_DIR}/mean_deformation_sequence${SEQUENCE_ID}.pt}"
CLINICAL_OUTPUT="${CLINICAL_OUTPUT:-${MEAN_RUN_DIR}/clinical_metrics_sequence${SEQUENCE_ID}.json}"

REFERENCE_MASK="${REFERENCE_MASK:-}"
LABELS="${LABELS:-1}"
ED_INDEX="${ED_INDEX:-0}"
ES_INDEX="${ES_INDEX:--1}"
MASK_TIME_INDEX="${MASK_TIME_INDEX:-}"
SPACING_MM="${SPACING_MM:-}"

VERIFY_NUM_SEQUENCES="${VERIFY_NUM_SEQUENCES:-5}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
TEST_FRACTION="${TEST_FRACTION:-0.1}"
MIN_LENGTH="${MIN_LENGTH:-2}"

run_prepare_hdf5() {
  "${PYTHON_BIN}" scripts/verify_roi_masks.py --config "${PREPARE_CONFIG}" --max-missing 50
  "${PYTHON_BIN}" scripts/prepare_hdf5.py --config "${PREPARE_CONFIG}"
  "${PYTHON_BIN}" scripts/verify_hdf5_splits.py --config "${PREPARE_CONFIG}"
}

run_train_score() {
  "${PYTHON_BIN}" scripts/train_cunsure_score.py --config "${SCORE_CONFIG}"
}

run_cinema_score_covariance() {
  "${PYTHON_BIN}" scripts/infer_score_cunsure_cinema_batch.py \
    --config "${SCORE_CINEMA_CONFIG}"
}

run_package_latent() {
  "${PYTHON_BIN}" scripts/package_latent_observations.py \
    --input-dir "${SCORE_OUTPUT_DIR}" \
    --output "${LATENT_H5}" \
    --compression lzf
}

run_sequence_index() {
  "${PYTHON_BIN}" scripts/export_sde_sequence_index.py \
    --h5 "${LATENT_H5}" \
    --output "${SEQUENCE_INDEX_FILE}" \
    --min-length "${MIN_LENGTH}" \
    --val-fraction "${VAL_FRACTION}" \
    --test-fraction "${TEST_FRACTION}"
}

run_verify_deformation_inputs() {
  "${PYTHON_BIN}" scripts/verify_deformation_training_inputs.py \
    --config "${MEAN_CONFIG}" \
    --num-sequences "${VERIFY_NUM_SEQUENCES}" \
    --random
}

run_train_mean_deformation() {
  "${PYTHON_BIN}" scripts/train_nodeo_mean_deformation.py \
    --config "${MEAN_CONFIG}"
}

run_infer_mean_deformation() {
  "${PYTHON_BIN}" scripts/infer_nodeo_mean_deformation.py \
    --checkpoint "${MEAN_CHECKPOINT}" \
    --h5 "${LATENT_H5}" \
    --output "${MEAN_DEFORMATION_OUTPUT}" \
    --sequence-index "${SEQUENCE_ID}" \
    --covariance "${COVARIANCE}" \
    --device "${DEVICE}"
}

run_clinical_metrics() {
  if [[ -z "${REFERENCE_MASK}" ]]; then
    echo "Skipping clinical metrics: set REFERENCE_MASK=path/to/reference_mask.nii.gz to enable."
    return 0
  fi

  cmd=(
    "${PYTHON_BIN}" scripts/compute_clinical_metrics.py
    --deformation "${MEAN_DEFORMATION_OUTPUT}"
    --reference-mask "${REFERENCE_MASK}"
    --output "${CLINICAL_OUTPUT}"
    --labels "${LABELS}"
    --ed-index "${ED_INDEX}"
    --es-index "${ES_INDEX}"
  )

  if [[ -n "${MASK_TIME_INDEX}" ]]; then
    cmd+=(--mask-time-index "${MASK_TIME_INDEX}")
  fi
  if [[ -n "${SPACING_MM}" ]]; then
    cmd+=(--spacing-mm "${SPACING_MM}")
  fi

  "${cmd[@]}"
}

run_deformation_all() {
  if [[ ! -f "${SEQUENCE_INDEX_FILE}" ]]; then
    run_sequence_index
  fi
  run_verify_deformation_inputs
  run_train_mean_deformation
  run_infer_mean_deformation
  run_clinical_metrics
}

run_full_all() {
  run_prepare_hdf5
  run_train_score
  run_cinema_score_covariance
  run_package_latent
  run_sequence_index
  run_verify_deformation_inputs
  run_train_mean_deformation
  run_infer_mean_deformation
  run_clinical_metrics
}

case "${STAGE}" in
  prepare)
    run_prepare_hdf5
    ;;
  train-score)
    run_train_score
    ;;
  cinema-score)
    run_cinema_score_covariance
    ;;
  package-latent)
    run_package_latent
    ;;
  index)
    run_sequence_index
    ;;
  verify)
    run_verify_deformation_inputs
    ;;
  train-mean|train)
    run_train_mean_deformation
    ;;
  infer-mean|infer)
    run_infer_mean_deformation
    ;;
  clinical)
    run_clinical_metrics
    ;;
  all)
    run_deformation_all
    ;;
  full)
    run_full_all
    ;;
  *)
    echo "Unknown stage: ${STAGE}"
    echo "Usage: $0 [prepare|train-score|cinema-score|package-latent|index|verify|train-mean|infer-mean|clinical|all|full]"
    exit 2
    ;;
esac
