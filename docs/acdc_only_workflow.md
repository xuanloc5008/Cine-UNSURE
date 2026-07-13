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
```

The equivalent single command is:

```bash
./run_acdc_workflow.sh full
```

Stage-by-stage execution is recommended because NODEO performs a separate
optimization for every sequence. Its numbered outputs and summaries are
resumable; rerunning a NODEO stage skips completed sequence files.

## Main outputs

- Score C-UNSURE checkpoint: `runs/acdc/cunsure_score/best.pt`
- Latent observations: `runs/acdc/latent_train_val.h5` and `runs/acdc/latent_test.h5`
- NODEO trajectories: `runs/acdc/nodeo_dir/{train,val,test}`
- SDE-RNN checkpoint: `runs/acdc/sde_rnn/best.pt`
- Conformal calibration: `runs/acdc/clinical/calibration_95.json`
- Independent test report: `runs/acdc/clinical/test_evaluation_summary.json`
- Test charts: `runs/acdc/clinical/test_charts`

To restart only the ACDC workflow, first archive any results that must be kept,
then remove `processed/acdc` and `runs/acdc`.
