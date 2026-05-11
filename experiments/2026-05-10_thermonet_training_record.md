# ThermoNet Training Record - 2026-05-10

This file records the reproducibility details for the current TE film surrogate-model training stage.

## Project Context

- The repository stores planning documents and code.
- Large database generation and model training are performed on a separate Windows training machine after pulling this repository.
- The database is not stored in git.

## Reproducibility Unit

To reproduce a training result, use the full tuple:

```text
git commit + database snapshot + training command + random seed + saved model checkpoint
```

Changing `train.py` later is acceptable as long as the original git commit is recorded and checked out before reproduction.

## Database Snapshot

- Location on training machine: `E:\TE-ML\TE_film_optimization\data\simulations`
- Metadata file: `data/simulations/metadata.csv`
- Field files: `data/simulations/fields/*.h5`
- Successful samples used by evaluation: 50,000 total inferred from 5,000 test samples under an 80/10/10 split.
- Split seed: `42`

The database directory must be backed up separately because it is ignored by git.

## Code Versions

- Evaluation script added in commit: `44ef6ab`
- High-delta-T training improvements added in commit: `5595fdd`
- Recommended code commit for the second training run: `5595fdd`

To reproduce the second training setup:

```bash
git checkout 5595fdd
```

## First Training Run

Run name:

```text
thermonet_training
```

Likely command, based on the original training defaults:

```powershell
python src/optimization/train.py --batch-size 32 --epochs 50 --seed 42
```

Model checkpoint:

```text
results/models/best_thermonet.pth
```

Evaluation command:

```powershell
python src/optimization/evaluate.py --split test --seed 42 --batch-size 64 --workers 4
```

Test metrics:

```text
count: 5000
MAE: 0.919937 K
RMSE: 1.335848 K
bias: -0.067181 K
R2: 0.849846
```

Top 10% true delta_T region:

```text
true delta_T cutoff: 8.757979 K
count: 500
MAE: 2.215557 K
RMSE: 2.695210 K
bias: -1.534395 K
R2: 0.342022
```

Ranking metrics:

```text
top overlap: 379 / 500
top recall: 0.758
top precision: 0.758
Spearman rank correlation: 0.914570
```

Interpretation:

```text
The model is globally useful but systematically underestimates the high-delta-T region. It should not be used for final inverse design before improving high-delta-T accuracy.
```

## Second Training Run

Goal:

```text
Improve high-delta-T prediction by normalizing the target and increasing loss weight for the top 10% high-delta-T samples.
```

Command:

```powershell
python src/optimization/train.py --batch-size 32 --epochs 80 --seed 42 --normalize-target --top-quantile 0.9 --top-weight 3.0 --run-name thermonet_v2_top_weighted
```

TensorBoard:

```powershell
tensorboard --logdir runs
```

Evaluation command after training:

```powershell
python src/optimization/evaluate.py --split test --seed 42 --batch-size 64 --workers 4
```

Training result:

```text
Stopped at epoch: 26 / 80
Early stopping patience: 12 epochs
Best validation loss: 0.1633
Final epoch train loss: 0.2011
Final epoch val loss: 0.1749
Final learning rate: 0.000075
```

Test metrics:

```text
count: 5000
MAE: 0.983772 K
RMSE: 1.385167 K
bias: 0.113681 K
R2: 0.838554
```

Top 10% true delta_T region:

```text
true delta_T cutoff: 8.757979 K
count: 500
MAE: 2.136587 K
RMSE: 2.651470 K
bias: -1.438906 K
R2: 0.363205
```

Ranking metrics:

```text
top overlap: 380 / 500
top recall: 0.760
top precision: 0.760
Spearman rank correlation: 0.910944
```

Interpretation:

```text
This run slightly improves the high-delta-T MAE and bias compared with the first run, but the improvement is small. Overall MAE, RMSE, R2, and Spearman ranking are worse than the first run. This checkpoint should not replace the first run as the default surrogate.
```

## Third Training Run

Goal:

```text
Test a softer high-delta-T weighting strategy to reduce the over-bias introduced by the second run.
```

Command:

```powershell
python src/optimization/train.py --batch-size 32 --epochs 80 --seed 42 --normalize-target --top-quantile 0.9 --top-weight 1.5 --run-name thermonet_v3_top_weight_1p5
```

Evaluation command after training:

```powershell
python src/optimization/evaluate.py --split test --seed 42 --batch-size 64 --workers 4
```

Test metrics:

```text
count: 5000
MAE: 0.978504 K
RMSE: 1.359299 K
bias: 0.199149 K
R2: 0.844528
```

Top 10% true delta_T region:

```text
true delta_T cutoff: 8.757979 K
count: 500
MAE: 2.092051 K
RMSE: 2.604356 K
bias: -1.276524 K
R2: 0.385634
```

Ranking metrics:

```text
top overlap: 381 / 500
top recall: 0.762
top precision: 0.762
Spearman rank correlation: 0.916661
```

Interpretation:

```text
This run is better than the second run and slightly improves top-region MAE, bias, R2, and ranking versus the first run. However, overall MAE and R2 are still worse than the first run, and the top-region bias remains too large. This suggests that simple top-region loss weighting alone is not enough.
```

## Current Decision

```text
Do not continue blindly extending v2/v3 training. The next useful step is to change the training objective or sampling strategy, then compare against the first run as the baseline.
```

## Fourth Training Run Plan

Goal:

```text
Penalize systematic underprediction in the high-delta-T region directly, instead of only increasing sample weights.
```

Code change:

```text
train.py now supports an optional high-delta-T underprediction penalty. The extra term is only applied to samples above the selected delta_T quantile and only when prediction is below the true value.
```

Recommended first command:

```powershell
python src/optimization/train.py --batch-size 32 --epochs 80 --seed 42 --normalize-target --top-quantile 0.9 --underpredict-penalty 0.5 --run-name thermonet_v4_underpredict_0p5
```

Batch-size 128 command used for the first completed v4 test:

```powershell
python src/optimization/train.py --batch-size 128 --epochs 80 --seed 42 --normalize-target --top-quantile 0.9 --underpredict-penalty 0.5 --run-name thermonet_v4_underpredict_0p5_bs128
```

Evaluation command after training:

```powershell
python src/optimization/evaluate.py --split test --seed 42 --batch-size 64 --workers 4
```

Completed v4 bs128 training result:

```text
Stopped at epoch: 44 / 80
Early stopping patience: 12 epochs
Best validation loss: 0.2773
Final epoch train loss: 0.2088
Final epoch val loss: 0.2903
Final learning rate: 0.000037
```

Test metrics:

```text
count: 5000
MAE: 1.012976 K
RMSE: 1.471687 K
bias: 0.408504 K
R2: 0.817756
```

Top 10% true delta_T region:

```text
true delta_T cutoff: 8.757979 K
count: 500
MAE: 1.844950 K
RMSE: 2.406587 K
bias: 0.194557 K
R2: 0.475398
```

Ranking metrics:

```text
top overlap: 380 / 500
top recall: 0.760
top precision: 0.760
Spearman rank correlation: 0.913197
```

Interpretation:

```text
The underprediction penalty successfully reduces high-delta-T underprediction. Top-region bias improves strongly from negative bias to near-neutral/slightly positive bias, and top-region MAE improves versus prior runs. However, overall MAE, RMSE, and R2 degrade substantially. The penalty coefficient 0.5 is too strong as a default setting.
```

Interrupted v4 bs192 note:

```text
A batch-size 192 run was manually interrupted during epoch 1 and should not be treated as a completed experiment. The evaluation above corresponds to the completed bs128 checkpoint.
```

## Fifth Training Run

Goal:

```text
Reduce the high-delta-T underprediction penalty from 0.5 to 0.2 to keep the bias improvement while recovering overall accuracy.
```

Command:

```powershell
python src/optimization/train.py --batch-size 128 --epochs 80 --seed 42 --normalize-target --top-quantile 0.9 --underpredict-penalty 0.2 --run-name thermonet_v5_underpredict_0p2_bs128
```

Evaluation command after training:

```powershell
python src/optimization/evaluate.py --split test --seed 42 --batch-size 128 --workers 4
```

Test metrics:

```text
count: 5000
MAE: 0.957993 K
RMSE: 1.364937 K
bias: 0.189765 K
R2: 0.843236
```

Top 10% true delta_T region:

```text
true delta_T cutoff: 8.757979 K
count: 500
MAE: 1.932834 K
RMSE: 2.417620 K
bias: -0.711206 K
R2: 0.470577
```

Ranking metrics:

```text
top overlap: 381 / 500
top recall: 0.762
top precision: 0.762
Spearman rank correlation: 0.913849
```

Interpretation:

```text
This is the best balanced result so far among the underprediction-penalty runs. Compared with v4, it recovers much of the overall accuracy while keeping the top-region bias within the target threshold. It still does not meet the target top-region MAE or top recall/precision criteria, so it is useful as a better surrogate checkpoint but not a final inverse-design model.
```

Decision rule:

```text
If top-region bias improves without damaging overall R2 more than v3, keep tuning the penalty. If it does not improve materially, the next step should be data-level balancing or model-capacity changes rather than more loss-weight sweeps.
```

Pass criteria before moving to surrogate-assisted inverse design:

```text
overall R2 >= 0.88
top 10% MAE <= 1.5 K
abs(top 10% bias) <= 0.8 K
top recall >= 0.82
top precision >= 0.82
```

## Output Files To Preserve

For each serious run, preserve:

```text
runs/<run_name>/
results/models/best_thermonet.pth
results/evaluation/metrics_test.json
results/evaluation/per_family_metrics_test.csv
results/evaluation/predictions_test.csv
results/evaluation/prediction_scatter_test.png
results/evaluation/residual_histogram_test.png
```

If multiple runs are compared, copy or rename `best_thermonet.pth` after each run so it is not overwritten.

## Automated Experiment Sweep

The repository now includes:

```text
src/optimization/run_experiments.py
```

Purpose:

```text
Run multiple training/evaluation configurations on the training machine, preserve each checkpoint, and maintain a ranked leaderboard under results/experiments/.
```

Import a manually completed current checkpoint:

```powershell
python src/optimization/run_experiments.py --import-current-run thermonet_v6_underpredict_0p1_bs128 --no-sweep
```

Run an automatic underprediction-penalty sweep:

```powershell
python src/optimization/run_experiments.py --penalties 0.05 0.1 0.15 0.2 0.25 --batch-size 128
```

Run advisor-driven adaptive experiments:

```powershell
python src/optimization/run_experiments.py --adaptive --max-adaptive-runs 4 --batch-size 128
```

Primary output:

```text
results/experiments/leaderboard.csv
results/experiments/advisor_decisions.json
```

## First Inverse-Design Pass

Selected surrogate checkpoint:

```text
results/experiments/thermonet_auto_adaptive_under_0p2_bs128/best_thermonet.pth
```

Rationale:

```text
Penalty 0.2 provides the best balance for the next stage: overall R2 remains near the useful range, top-region bias is within the current threshold, and top-region MAE is better than lighter-penalty runs. Higher penalties improve top MAE but damage overall R2 too much.
```

Candidate screening command:

```powershell
python src/optimization/inverse_design.py screen --model-path results/experiments/thermonet_auto_adaptive_under_0p2_bs128/best_thermonet.pth --num-candidates 100000 --top-k 500 --batch-size 256 --mode mixed --structured-ratio 0.9 --seed 20260511
```

FDM verification command:

```powershell
python src/optimization/inverse_design.py verify --screen-dir results/inverse_design/screen_<timestamp> --verify-count 50
```

Decision rule:

```text
Use surrogate predictions only to rank and shortlist candidates. Do not treat surrogate predictions as final optimized results. Final claims must come from FDM-verified candidates.
```
