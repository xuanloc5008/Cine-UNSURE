# NODEO registration ablation protocol

Run every experiment on the same patient, split, ROI H5 files, seed, and evaluation
scripts. Do not continue from a previous experiment checkpoint: each profile changes
the objective or parameterization and writes to a separate output directory.

## Common environment

```bash
cd /workspace/Cine-UNSURE
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"
export PATIENT=patient101
export SPLIT=test
```

The commands below use `NODEO_ONLY=1` so SDE and clinical computations do not run
during the registration ablation.

## 0. RK4 baseline

```bash
SOLVER=rk4 OPTION=1 NODEO_ONLY=1 OVERWRITE_NODEO=1 \
PATIENT="$PATIENT" SPLIT="$SPLIT" ./run_patient_sequence_workflow.sh
```

## 1. Dopri5 solver check

```bash
SOLVER=dopri5 OPTION=1 NODEO_ONLY=1 OVERWRITE_NODEO=1 \
PATIENT="$PATIENT" SPLIT="$SPLIT" ./run_patient_sequence_workflow.sh
```

Keep RK4 if Dopri5 changes MAE by less than 1%. This isolates numerical integration
error because the model, loss, and optimizer match the baseline.

## 2. RK4 with multi-scale LNCC

```bash
SOLVER=rk4_multiscale OPTION=1 NODEO_ONLY=1 OVERWRITE_NODEO=1 \
PATIENT="$PATIENT" SPLIT="$SPLIT" ./run_patient_sequence_workflow.sh
```

This uses image scales `1.0, 0.5, 0.25`, NCC windows `9, 5, 3`, and weights
`0.5, 0.3, 0.2`.

## 3. Full-resolution velocity and weaker Gaussian smoothing

```bash
SOLVER=rk4_highres OPTION=1 NODEO_ONLY=1 OVERWRITE_NODEO=1 \
PATIENT="$PATIENT" SPLIT="$SPLIT" ./run_patient_sequence_workflow.sh
```

This retains multi-scale LNCC, changes `output_downsamples` from `1` to `0`,
increases the bottleneck to `64`, and changes Gaussian smoothing from
`window=11, sigma=2.0` to `window=7, sigma=1.25`.

## 4. Structural gradient loss

```bash
SOLVER=rk4_structural OPTION=1 NODEO_ONLY=1 OVERWRITE_NODEO=1 \
PATIENT="$PATIENT" SPLIT="$SPLIT" ./run_patient_sequence_workflow.sh
```

This adds bidirectional 3D gradient-orientation loss with weight `0.10` to the
configuration from experiment 3.

## 5. Adam followed by LBFGS

```bash
SOLVER=rk4_lbfgs OPTION=1 NODEO_ONLY=1 OVERWRITE_NODEO=1 \
PATIENT="$PATIENT" SPLIT="$SPLIT" ./run_patient_sequence_workflow.sh
```

The selected Adam state is refined with strong-Wolfe LBFGS for up to 25 iterations.
The output records `best_stage: adam` or `best_stage: lbfgs`.

## Evaluation

```bash
for PROFILE in rk4 dopri5 rk4_multiscale rk4_highres rk4_structural rk4_lbfgs; do
  WORK="runs/acdc/patient_sequence_workflow/${PROFILE}/${SPLIT}/${PATIENT}"
  NODEO=$(find "$WORK/nodeo/$SPLIT" -maxdepth 1 -name '[0-9]*.pt' | sort | head -1)
  test -n "$NODEO" || continue
  mkdir -p "$WORK/evaluation"
  python scripts/evaluate_nodeo_sequence.py \
    --input "$NODEO" \
    --output "$WORK/evaluation/sequence_metrics.json" \
    --device cuda \
    --ncc-window 9
  python scripts/evaluate_nodeo_anatomy.py \
    --input "$NODEO" \
    --output "$WORK/evaluation/anatomy_metrics.json" \
    --datasets-root datasets \
    --device cuda
  python scripts/visualize_nodeo_dir_sequence.py \
    --input "$NODEO" \
    --output "$WORK/evaluation/predicted_vs_target.gif" \
    --slice-index 5 \
    --fps 4
done
```

Print a compact comparison table:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("runs/acdc/patient_sequence_workflow")
profiles = ("rk4", "dopri5", "rk4_multiscale", "rk4_highres", "rk4_structural", "rk4_lbfgs")
print(f"{'profile':20} {'MAE':>9} {'NCC':>9} {'fold':>9} {'Jmin':>9} {'J<0.5':>9}")
for profile in profiles:
    path = root / profile / "test" / "patient101" / "evaluation" / "sequence_metrics.json"
    if not path.exists():
        continue
    row = json.loads(path.read_text())
    image = row["intensity_summary"]["predicted_to_target"]
    frames = row["per_frame"]
    mean_below = sum(frame["below_minimum_jacobian_fraction"] for frame in frames) / len(frames)
    print(
        f"{profile:20} {image['mae']:9.6f} {image['global_ncc']:9.6f} "
        f"{row['trajectory_summary']['fold_fraction_mean']:9.2e} "
        f"{min(frame['jacobian_min'] for frame in frames):9.4f} {mean_below:9.4f}"
    )
PY
```

Keep a new stage only if it lowers predicted-to-target MAE and does not introduce
folding, materially increase the fraction with `det(J) < 0.5`, or reduce ED-to-ES
LV/MYO/RV Dice. Prefer at least a 2% relative MAE reduction before accepting a more
complex stage.

## 6. Optional residual SVF refinement

Only run this if experiment 5 still leaves clinically meaningful boundary errors.

```bash
WORK="runs/acdc/patient_sequence_workflow/rk4_lbfgs/${SPLIT}/${PATIENT}"
NODEO=$(find "$WORK/nodeo/$SPLIT" -maxdepth 1 -name '[0-9]*.pt' | sort | head -1)

python scripts/refine_nodeo_residual.py \
  --config configs/acdc/experiments/06_residual_refinement.yaml \
  --input "$NODEO" \
  --output "$WORK/nodeo/${SPLIT}/residual_refined.pt" \
  --device cuda

python scripts/evaluate_nodeo_sequence.py \
  --input "$WORK/nodeo/${SPLIT}/residual_refined.pt" \
  --output "$WORK/evaluation/residual_refined_metrics.json" \
  --device cuda \
  --ncc-window 9

python scripts/evaluate_nodeo_anatomy.py \
  --input "$WORK/nodeo/${SPLIT}/residual_refined.pt" \
  --output "$WORK/evaluation/residual_refined_anatomy.json" \
  --datasets-root datasets \
  --device cuda

python scripts/visualize_nodeo_dir_sequence.py \
  --input "$WORK/nodeo/${SPLIT}/residual_refined.pt" \
  --output "$WORK/evaluation/residual_refined.gif" \
  --slice-index 5 \
  --fps 4
```

The residual is a per-frame stationary velocity field integrated by
scaling-and-squaring and composed with the locked NODEO trajectory. Spatial,
temporal, cycle, magnitude, and Jacobian penalties prevent unrestricted image
fitting.
