#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/acdc/patient_sequence_workflow.yaml}"

cmd=("${PYTHON_BIN}" scripts/run_patient_sequence_workflow.py --config "${CONFIG}")
if [[ -n "${OPTION:-}" ]]; then
  cmd+=(--option "${OPTION}")
fi
if [[ -n "${PATIENT:-}" ]]; then
  cmd+=(--patient "${PATIENT}")
fi
if [[ -n "${SPLIT:-}" ]]; then
  cmd+=(--split "${SPLIT}")
fi
if [[ -n "${SCORE_CHECKPOINT:-}" ]]; then
  cmd+=(--score-checkpoint "${SCORE_CHECKPOINT}")
fi
if [[ "${OVERWRITE_NODEO:-0}" == "1" ]]; then
  cmd+=(--overwrite-nodeo)
fi

"${cmd[@]}"
