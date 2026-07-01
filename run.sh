cd /Users/xuanloc/Documents/Codex/2026-06-30/o/outputs/score_cunsure_foundation_compare

python -m pip install -e ".[medical-io]"

python scripts/train_score.py \
  --data "$ACDC_TRAIN" "$MM1_TRAIN" "$MNM2_ROOT" \
  --output "$SCORE_CKPT" \
  --channels 1 \
  --spatial-dims 3 \
  --time-axis 0 \
  --frame-layout dhw \
  --include "*_4d.nii.gz" "*_sa.nii.gz" "*_SA_CINE.nii.gz" \
  --exclude "*_gt.nii.gz" "*_ED.nii.gz" "*_ES.nii.gz" "*_LA_*.nii.gz" \
  --image-size 192 \
  --depth-size 16 \
  --epochs 50 \
  --batch-size 1 \
  --base-channels 32 \
  --depth 3 \
  --device mps \
  --log-every 100