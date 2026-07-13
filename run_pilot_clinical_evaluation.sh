#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
CHECKPOINT="${CHECKPOINT:-runs/sde_rnn_uncertainty_roi_pilot62/best.pt}"
H5="${H5:-runs/selected/latent_observations_cinema_score_cunsure_roi.h5}"
NODEO_SUMMARY="${NODEO_SUMMARY:-runs/nodeo_dir_roi/train/summary.jsonl}"
ROOT_OUT="${ROOT_OUT:-runs/sde_rnn_uncertainty_roi_pilot62}"
EVALUATION_MANIFEST="${ROOT_OUT}/clinical_evaluation_targets.jsonl"

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
  local prediction_dir="$2"
  local chart_dir="${3:-}"
  local deformation_dir="${ROOT_OUT}/deformations"
  mkdir -p "${prediction_dir}" "${deformation_dir}"
  [[ -z "${chart_dir}" ]] || mkdir -p "${chart_dir}"

  while IFS=$'\t' read -r sequence_index reference_mask ed_index es_index; do
    deformation="${deformation_dir}/sequence${sequence_index}.pt"
    prediction="${prediction_dir}/sequence${sequence_index}.json"
    if [[ ! -f "${deformation}" ]]; then
      "${PYTHON_BIN}" scripts/infer_sde_rnn_uncertainty.py \
        --checkpoint "${CHECKPOINT}" \
        --h5 "${H5}" \
        --nodeo-summary "${NODEO_SUMMARY}" \
        --sequence-index "${sequence_index}" \
        --output "${deformation}" \
        --device "${DEVICE}"
    fi
    cmd=(
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

"${PYTHON_BIN}" scripts/prepare_acdc_clinical_targets.py \
  --config configs/train_sde_rnn_uncertainty.yaml \
  --checkpoint "${CHECKPOINT}" \
  --split val \
  --output "${EVALUATION_MANIFEST}" \
  --label 3

process_manifest \
  "${EVALUATION_MANIFEST}" \
  "${ROOT_OUT}/clinical_evaluation" \
  "${ROOT_OUT}/clinical_evaluation_charts"

"${PYTHON_BIN}" scripts/evaluate_clinical_prediction_bands.py \
  --predictions "${ROOT_OUT}/clinical_evaluation/*.json" \
  --targets "${EVALUATION_MANIFEST}" \
  --output "${ROOT_OUT}/clinical_evaluation_summary.json"

echo "Validation evaluation (direct model uncertainty): ${ROOT_OUT}/clinical_evaluation_summary.json"
echo "Charts: ${ROOT_OUT}/clinical_evaluation_charts"
