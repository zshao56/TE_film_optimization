# OpenCode ThermoNet Tuning Rules

These rules are for an LLM agent running on the Windows training machine.

## Scope

Work only inside this repository:

```text
E:\TE-ML\TE_film_optimization
```

Do not read, write, delete, move, or modify files outside this repository.

Do not change source code unless the human explicitly asks for code changes.

For the current tuning stage, change only this training parameter:

```text
--underpredict-penalty
```

Keep these fixed unless the human explicitly asks otherwise:

```text
--batch-size 128
--epochs 80
--seed 42
--normalize-target
--top-quantile 0.9
--workers 4
```

## Allowed Commands

Run commands only from the repository root:

```powershell
cd E:\TE-ML\TE_film_optimization
```

Allowed:

```powershell
python src/optimization/train.py ...
python src/optimization/evaluate.py ...
python src/optimization/run_experiments.py ...
git status
git pull
```

Allowed only when preserving experiment outputs inside the repository:

```powershell
Copy-Item
New-Item
```

Not allowed:

```powershell
Remove-Item
del
rmdir
git reset
git checkout
git clean
pip install
conda install
```

Do not edit system settings, environment variables, GPU driver settings, Windows settings, or files outside the repository.

## Current Baselines

Use these known results as context:

| Run | underpredict_penalty | Overall R2 | Top MAE K | Top Bias K | Top Recall |
|---|---:|---:|---:|---:|---:|
| v1 | none | 0.849846 | 2.215557 | -1.534395 | 0.758 |
| v3 | top-weight 1.5 | 0.844528 | 2.092051 | -1.276524 | 0.762 |
| v4 | 0.5 | 0.817756 | 1.844950 | 0.194557 | 0.760 |
| v5 | 0.2 | 0.843236 | 1.932834 | -0.711206 | 0.762 |

Interpretation:

```text
Penalty 0.5 fixes underprediction but damages overall R2 too much.
Penalty 0.2 is the best balanced result so far.
Ranking/top recall has not improved much yet.
```

## Targets

Primary pass criteria:

```text
overall R2 >= 0.88
top 10% MAE <= 1.5 K
abs(top 10% bias) <= 0.8 K
top recall >= 0.82
top precision >= 0.82
```

Current realistic near-term target:

```text
Keep overall R2 >= 0.84
Keep abs(top bias) <= 0.8 K
Reduce top MAE below v5's 1.932834 K if possible
Improve top recall/precision above 0.762 if possible
```

## Decision Logic

After every completed run, always run evaluation:

```powershell
python src/optimization/evaluate.py --split test --seed 42 --batch-size 128 --workers 4
```

Then decide the next `--underpredict-penalty`:

1. If `top_bias_K < -0.8`, the model is still underpredicting high-delta-T structures.
   Increase penalty by `0.05`.

2. If `top_bias_K > +0.8`, the model is overcorrecting high-delta-T structures.
   Decrease penalty by `0.05`.

3. If `abs(top_bias_K) <= 0.8` but `overall_r2 < 0.84`, the penalty is too damaging globally.
   Decrease penalty by `0.05`.

4. If `abs(top_bias_K) <= 0.8`, `overall_r2 >= 0.84`, and `top_mae_K > 1.5`,
   try one nearby penalty not yet tested.
   Prefer the direction that moves bias toward zero:
   - negative bias: increase by `0.05`
   - positive bias: decrease by `0.05`

5. If several penalties give similar metrics, prefer the run with:
   - higher `overall_r2`
   - lower `top_mae_K`
   - lower `abs(top_bias_K)`
   - higher `top_recall`

6. If `top_recall` and `top_precision` stay around `0.76` after testing nearby penalties,
   stop penalty tuning. Report that ranking is the bottleneck and recommend a future code change:

```text
Add ranking loss or top-region classification loss.
```

## Penalty Search Bounds

Do not test outside this range:

```text
0.0 <= underpredict_penalty <= 0.5
```

Recommended next candidates around current best:

```text
0.1
0.15
0.25
0.05
0.3
```

Do not repeat a penalty that already has a completed evaluation unless the previous run failed or was interrupted.

## Run Naming

Every run must use a unique run name:

```powershell
--run-name thermonet_auto_underpredict_<penalty>_bs128
```

Examples:

```powershell
--run-name thermonet_auto_underpredict_0p15_bs128
--run-name thermonet_auto_underpredict_0p25_bs128
```

Use `p` instead of `.` in run names.

## Preferred Manual Commands

For penalty `0.15`:

```powershell
python src/optimization/train.py --batch-size 128 --epochs 80 --seed 42 --normalize-target --top-quantile 0.9 --underpredict-penalty 0.15 --run-name thermonet_auto_underpredict_0p15_bs128
python src/optimization/evaluate.py --split test --seed 42 --batch-size 128 --workers 4
```

For penalty `0.25`:

```powershell
python src/optimization/train.py --batch-size 128 --epochs 80 --seed 42 --normalize-target --top-quantile 0.9 --underpredict-penalty 0.25 --run-name thermonet_auto_underpredict_0p25_bs128
python src/optimization/evaluate.py --split test --seed 42 --batch-size 128 --workers 4
```

## Preferred Automated Command

If using the repository's advisor script:

```powershell
python src/optimization/run_experiments.py --adaptive --max-adaptive-runs 4 --batch-size 128
```

The advisor writes:

```text
results/experiments/leaderboard.csv
results/experiments/advisor_decisions.json
```

## Required Report To Human

After each run, report only:

```text
run_name
underpredict_penalty
overall mae_K, rmse_K, bias_K, r2
top-region mae_K, rmse_K, bias_K, r2
top_recall, top_precision, spearman_rank_corr
decision for next penalty
reason for the decision
```

Do not make code changes. Do not delete previous results.
