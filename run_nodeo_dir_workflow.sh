#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"
STAGE="${1:-all}"
SPLIT_CONFIG="${SPLIT_CONFIG:-configs/nodeo_roi_splits.yaml}"
NODEO_CONFIG="${NODEO_CONFIG:-configs/train_nodeo_dir_roi.yaml}"
START_INDEX="${START_INDEX:-0}"
LIMIT="${LIMIT:-}"

run_split() {
  "${PYTHON_BIN}" scripts/build_nodeo_roi_splits.py --config "${SPLIT_CONFIG}"
}

run_cohort() {
  local split="$1"
  local cmd=(
    "${PYTHON_BIN}" scripts/run_nodeo_dir.py
    --config "${NODEO_CONFIG}"
    --split "${split}"
    --start-index "${START_INDEX}"
  )
  if [[ -n "${LIMIT}" ]]; then
    cmd+=(--limit "${LIMIT}")
  fi
  "${cmd[@]}"
}

run_summary() {
  local split="$1"
  "${PYTHON_BIN}" scripts/summarize_nodeo_dir.py \
    --summary "runs/nodeo_dir_roi/${split}/summary.jsonl" \
    --output "runs/nodeo_dir_roi/${split}/metrics_summary.json"
}

case "${STAGE}" in
  split|prepare)
    run_split
    ;;
  train)
    run_cohort train
    ;;
  val)
    run_cohort val
    ;;
  test)
    run_cohort test
    ;;
  summarize-train)
    run_summary train
    ;;
  summarize-val)
    run_summary val
    ;;
  summarize-test)
    run_summary test
    ;;
  all)
    run_split
    run_cohort train
    run_cohort val
    run_cohort test
    run_summary train
    run_summary val
    run_summary test
    ;;
  *)
    echo "Usage: $0 [split|train|val|test|summarize-train|summarize-val|summarize-test|all]"
    exit 2
    ;;
esac
