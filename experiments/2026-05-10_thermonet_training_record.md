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
