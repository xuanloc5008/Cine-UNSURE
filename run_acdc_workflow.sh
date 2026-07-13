#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"
STAGE="${1:-help}"
DEVICE="${DEVICE:-cuda}"
PREPARE_CONFIG="configs/acdc/prepare_hdf5.yaml"
SCORE_CONFIG="configs/acdc/train_cunsure_score.yaml"
NODEO_SPLIT_CONFIG="configs/acdc/nodeo_roi_splits.yaml"
NODEO_CONFIG="${NODEO_CONFIG:-configs/acdc/train_nodeo_dir_roi.yaml}"
SDE_CONFIG="configs/acdc/train_sde_rnn_uncertainty.yaml"
SDE_POSTHOC_CONFIG="configs/acdc/run_sde_sequence_posthoc.yaml"
NODEO_EULER_CONFIG="configs/acdc/train_nodeo_dir_roi_euler.yaml"
NODEO_EULER_30S_CONFIG="configs/acdc/train_nodeo_dir_roi_euler_30s.yaml"

run_prepare() {
  "${PYTHON_BIN}" scripts/verify_roi_masks.py --config "${PREPARE_CONFIG}" --max-missing 20
  "${PYTHON_BIN}" scripts/prepare_hdf5.py --config "${PREPARE_CONFIG}"
  "${PYTHON_BIN}" scripts/verify_hdf5_splits.py --config "${PREPARE_CONFIG}"
}

run_score_infer() {
  local split="$1"
  "${PYTHON_BIN}" scripts/infer_score_cunsure_cinema_batch.py \
    --config "configs/acdc/infer_cinema_score_${split}.yaml"
}

run_package() {
  "${PYTHON_BIN}" scripts/package_latent_observations.py \
    --input-dir runs/acdc/cinema_score/train \
    --input-dir runs/acdc/cinema_score/val \
    --output runs/acdc/latent_train_val.h5 \
    --compression lzf
  "${PYTHON_BIN}" scripts/package_latent_observations.py \
    --input-dir runs/acdc/cinema_score/test \
    --output runs/acdc/latent_test.h5 \
    --compression lzf
}

run_manifest() {
  "${PYTHON_BIN}" scripts/build_nodeo_roi_splits.py --config "${NODEO_SPLIT_CONFIG}"
}

run_nodeo() {
  local split="$1"
  local run_dir
  run_dir="$("${PYTHON_BIN}" - "${NODEO_CONFIG}" <<'PY'
import sys
import yaml
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    print(yaml.safe_load(handle)["output"]["run_dir"])
PY
)"
  "${PYTHON_BIN}" scripts/run_nodeo_dir.py --config "${NODEO_CONFIG}" --split "${split}"
  "${PYTHON_BIN}" scripts/summarize_nodeo_dir.py \
    --summary "${run_dir}/${split}/summary.jsonl" \
    --output "${run_dir}/${split}/metrics_summary.json"
}

run_indices() {
  "${PYTHON_BIN}" scripts/export_sde_sequence_index.py \
    --h5 runs/acdc/latent_train_val.h5 \
    --output runs/acdc/sde_sequence_index_train_val.jsonl \
    --min-length 2 \
    --split-manifest processed/acdc/nodeo_roi_splits.jsonl
  "${PYTHON_BIN}" scripts/export_sde_sequence_index.py \
    --h5 runs/acdc/latent_test.h5 \
    --output runs/acdc/sde_sequence_index_test.jsonl \
    --min-length 2 \
    --split-manifest processed/acdc/nodeo_roi_splits.jsonl
}

run_sde() {
  "${PYTHON_BIN}" scripts/verify_deformation_training_inputs.py \
    --config "${SDE_CONFIG}" --num-sequences 10 --random
  "${PYTHON_BIN}" scripts/train_sde_rnn_uncertainty.py \
    --config "${SDE_CONFIG}" --mode full
}

case "${STAGE}" in
  prepare) run_prepare ;;
  train-score) "${PYTHON_BIN}" scripts/train_cunsure_score.py --config "${SCORE_CONFIG}" ;;
  evaluate-score) "${PYTHON_BIN}" scripts/evaluate_cunsure_unsure_protocol.py --config configs/acdc/evaluate_cunsure_unsure_protocol.yaml ;;
  infer-train) run_score_infer train ;;
  infer-val) run_score_infer val ;;
  infer-test) run_score_infer test ;;
  infer-all) run_score_infer train; run_score_infer val; run_score_infer test ;;
  package) run_package ;;
  manifest) run_manifest ;;
  nodeo-train) run_nodeo train ;;
  nodeo-val) run_nodeo val ;;
  nodeo-test) run_nodeo test ;;
  nodeo-all) run_nodeo train; run_nodeo val; run_nodeo test ;;
  nodeo-euler-train) NODEO_CONFIG="${NODEO_EULER_CONFIG}" run_nodeo train ;;
  nodeo-euler-val) NODEO_CONFIG="${NODEO_EULER_CONFIG}" run_nodeo val ;;
  nodeo-euler-test) NODEO_CONFIG="${NODEO_EULER_CONFIG}" run_nodeo test ;;
  nodeo-euler-all)
    NODEO_CONFIG="${NODEO_EULER_CONFIG}" run_nodeo train
    NODEO_CONFIG="${NODEO_EULER_CONFIG}" run_nodeo val
    NODEO_CONFIG="${NODEO_EULER_CONFIG}" run_nodeo test
    ;;
  nodeo-euler30-train) NODEO_CONFIG="${NODEO_EULER_30S_CONFIG}" run_nodeo train ;;
  nodeo-euler30-val) NODEO_CONFIG="${NODEO_EULER_30S_CONFIG}" run_nodeo val ;;
  nodeo-euler30-test) NODEO_CONFIG="${NODEO_EULER_30S_CONFIG}" run_nodeo test ;;
  nodeo-euler30-all)
    NODEO_CONFIG="${NODEO_EULER_30S_CONFIG}" run_nodeo train
    NODEO_CONFIG="${NODEO_EULER_30S_CONFIG}" run_nodeo val
    NODEO_CONFIG="${NODEO_EULER_30S_CONFIG}" run_nodeo test
    ;;
  index) run_indices ;;
  train-sde) run_sde ;;
  sde-posthoc-train) CONFIG="${SDE_POSTHOC_CONFIG}" ./run_sde_sequence_posthoc_workflow.sh train ;;
  sde-posthoc-val) CONFIG="${SDE_POSTHOC_CONFIG}" ./run_sde_sequence_posthoc_workflow.sh val ;;
  sde-posthoc-test) CONFIG="${SDE_POSTHOC_CONFIG}" ./run_sde_sequence_posthoc_workflow.sh test ;;
  sde-posthoc-train-val) CONFIG="${SDE_POSTHOC_CONFIG}" ./run_sde_sequence_posthoc_workflow.sh train-val ;;
  clinical) DEVICE="${DEVICE}" ./run_acdc_clinical_evaluation.sh ;;
  pathology) DEVICE="${DEVICE}" ./run_acdc_pathology.sh all ;;
  full)
    run_prepare
    "${PYTHON_BIN}" scripts/train_cunsure_score.py --config "${SCORE_CONFIG}"
    run_score_infer train
    run_score_infer val
    run_score_infer test
    run_package
    run_manifest
    run_nodeo train
    run_nodeo val
    run_nodeo test
    run_indices
    run_sde
    DEVICE="${DEVICE}" ./run_acdc_clinical_evaluation.sh
    DEVICE="${DEVICE}" ./run_acdc_pathology.sh all
    ;;
  *)
    echo "Usage: $0 {prepare|train-score|evaluate-score|infer-train|infer-val|infer-test|infer-all|package|manifest|nodeo-train|nodeo-val|nodeo-test|nodeo-all|nodeo-euler-train|nodeo-euler-val|nodeo-euler-test|nodeo-euler-all|nodeo-euler30-train|nodeo-euler30-val|nodeo-euler30-test|nodeo-euler30-all|index|train-sde|sde-posthoc-train|sde-posthoc-val|sde-posthoc-test|sde-posthoc-train-val|clinical|pathology|full}"
    exit 2
    ;;
esac
