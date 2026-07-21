#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"
STAGE="${1:-help}"
PREPARE_CONFIG="configs/acdc/prepare_hdf5.yaml"
NODEO_SPLIT_CONFIG="configs/acdc/nodeo_roi_splits.yaml"
NODEO_CONFIG="${NODEO_CONFIG:-configs/acdc/train_nodeo_dir_roi_dopri5.yaml}"
SDE_POSTHOC_CONFIG="configs/acdc/run_sde_sequence_posthoc.yaml"
NODEO_EULER_CONFIG="configs/acdc/train_nodeo_dir_roi_euler.yaml"
NODEO_EULER_30S_CONFIG="configs/acdc/train_nodeo_dir_roi_euler_30s.yaml"

run_prepare() {
  "${PYTHON_BIN}" scripts/verify_roi_masks.py --config "${PREPARE_CONFIG}" --max-missing 20
  "${PYTHON_BIN}" scripts/prepare_hdf5.py --config "${PREPARE_CONFIG}"
  "${PYTHON_BIN}" scripts/verify_hdf5_splits.py --config "${PREPARE_CONFIG}"
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

case "${STAGE}" in
  prepare) run_prepare ;;
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
  sde-posthoc-train) CONFIG="${SDE_POSTHOC_CONFIG}" ./run_sde_sequence_posthoc_workflow.sh train ;;
  sde-posthoc-val) CONFIG="${SDE_POSTHOC_CONFIG}" ./run_sde_sequence_posthoc_workflow.sh val ;;
  sde-posthoc-test) CONFIG="${SDE_POSTHOC_CONFIG}" ./run_sde_sequence_posthoc_workflow.sh test ;;
  sde-posthoc-train-val) CONFIG="${SDE_POSTHOC_CONFIG}" ./run_sde_sequence_posthoc_workflow.sh train-val ;;
  patient) ./run_patient_sequence_workflow.sh ;;
  full)
    run_prepare
    run_manifest
    run_nodeo train
    run_nodeo val
    run_nodeo test
    CONFIG="${SDE_POSTHOC_CONFIG}" ./run_sde_sequence_posthoc_workflow.sh all
    ;;
  *)
    echo "Usage: $0 {prepare|manifest|nodeo-train|nodeo-val|nodeo-test|nodeo-all|nodeo-euler-train|nodeo-euler-val|nodeo-euler-test|nodeo-euler-all|nodeo-euler30-train|nodeo-euler30-val|nodeo-euler30-test|nodeo-euler30-all|sde-posthoc-train|sde-posthoc-val|sde-posthoc-test|sde-posthoc-train-val|patient|full}"
    exit 2
    ;;
esac
