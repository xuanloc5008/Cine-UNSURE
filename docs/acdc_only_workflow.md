# ACDC-only workflow

All commands are run from the repository root. The deterministic split is made
by patient/source with seed 2026:

- 80% of ACDC official training: train.
- 20% of ACDC official training: validation.
- ACDC official testing: independent test.

## Run stage by stage

```bash
./run_acdc_workflow.sh prepare
./run_acdc_workflow.sh train-score

./run_acdc_workflow.sh infer-train
./run_acdc_workflow.sh infer-val
./run_acdc_workflow.sh infer-test
./run_acdc_workflow.sh package

./run_acdc_workflow.sh manifest
./run_acdc_workflow.sh nodeo-train
./run_acdc_workflow.sh nodeo-val
./run_acdc_workflow.sh nodeo-test

./run_acdc_workflow.sh index
./run_acdc_workflow.sh train-sde
./run_acdc_workflow.sh clinical
./run_acdc_workflow.sh pathology
```

The equivalent single command is:

```bash
./run_acdc_workflow.sh full
```

Stage-by-stage execution is recommended because NODEO performs a separate
optimization for every sequence. Its numbered outputs and summaries are
resumable; rerunning a NODEO stage skips completed sequence files.

For a time-limited screening run targeting approximately 30 seconds per
sequence on a high-end CUDA GPU, use the fast NODEO config:

```bash
NODEO_CONFIG=configs/acdc/train_nodeo_dir_roi_fast.yaml \
  ./run_acdc_workflow.sh nodeo-train
```

The fast config uses Euler integration, a smaller velocity network and a
30-second optimization budget. It is intended for iteration and quality
screening; use `train_nodeo_dir_roi.yaml` for the final experiment.

For a controlled Euler-versus-RK4 comparison that keeps the full network,
losses and 500 optimization epochs unchanged, run:

```bash
./run_acdc_workflow.sh nodeo-euler-train
./run_acdc_workflow.sh nodeo-euler-val
./run_acdc_workflow.sh nodeo-euler-test
```

Euler outputs are isolated under `runs/acdc/nodeo_dir_euler`.

## Main outputs

- Score C-UNSURE checkpoint: `runs/acdc/cunsure_score/best.pt`
- Latent observations: `runs/acdc/latent_train_val.h5` and `runs/acdc/latent_test.h5`
- NODEO trajectories: `runs/acdc/nodeo_dir/{train,val,test}`
- SDE-RNN checkpoint: `runs/acdc/sde_rnn/best.pt`
- Independent test report with model-derived uncertainty coverage: `runs/acdc/clinical/test_evaluation_summary.json`
- Test charts: `runs/acdc/clinical/test_charts`
- Pathology classifier: `runs/acdc/pathology/best.pt`
- Pathology test probabilities and bands: `runs/acdc/pathology/test_predictions`
- Pathology test report: `runs/acdc/pathology/test_evaluation.json`

Clinical uncertainty bands are computed directly from the SDE-RNN deformation
covariance using analytical delta-method propagation. No validation target,
conformal factor, or self-calibration stage modifies these bands. The ED mask
provides the reference anatomy propagated through the deformation trajectory.
The held-out ES mask and derived ground-truth clinical values are used only for
validation/test accuracy and empirical coverage, never to rescale a band.

## Pathology prediction

The pathology stage consumes the predicted clinical trajectories and their
analytically propagated variances, not raw images and not ground-truth clinical
metrics. Each variable-length trajectory is resampled to 20 normalized cardiac
time points. The classifier receives the metric means together with
`log(1 + standard_error)` for EF, LV volume, wall motion, and the three strain
components. ACDC `Group` metadata supplies the five supervised classes:
`NOR`, `DCM`, `HCM`, `MINF`, and `RV`.

At inference, the Jacobian of each softmax probability with respect to the
clinical feature vector propagates the diagonal clinical covariance to a
pathology probability variance. The resulting 95% probability bands are direct
model-derived bands and are never rescaled using validation or test targets.

To restart only the ACDC workflow, first archive any results that must be kept,
then remove `processed/acdc` and `runs/acdc`.
