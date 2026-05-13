# V2 Flat Unified Thickness Experiment

This branch is a clean v2 experiment setup for rerunning the workflow from data sampling through filtering, training, evaluation, and real-world inverse screening.

## Key Changes

- Thickness is sampled uniformly from `0.0004` to `0.004` m.
- Curvature is disabled for generated data and benchmark scenarios; all cases are flat.
- Hot-boundary sampling uses a low/mid/high mixture so the model sees both small and large `delta_T_parallel` regimes.
- Training keeps the high-delta-T weighting and underprediction penalty, and adds a low-delta-T relative-error penalty.
- The engine benchmark scenario is revised to `center_temp = 433 K` and `edge_temp = 373 K`.
- Previous presentation outputs and diagnostic files are intentionally not part of this branch.

## Run

```bash
conda activate teml
python src/optimization/run_configured_pipeline.py --config configs/v2_flat_unified_thickness.json
```

To run one stage at a time:

```bash
python src/optimization/run_configured_pipeline.py --config configs/v2_flat_unified_thickness.json --stages data_generation
python src/optimization/run_configured_pipeline.py --config configs/v2_flat_unified_thickness.json --stages metadata_filter
python src/optimization/run_configured_pipeline.py --config configs/v2_flat_unified_thickness.json --stages training
python src/optimization/run_configured_pipeline.py --config configs/v2_flat_unified_thickness.json --stages evaluation
python src/optimization/run_configured_pipeline.py --config configs/v2_flat_unified_thickness.json --stages real_world_benchmark
```

## Checks

After metadata filtering:

```bash
python scripts/check_v2_dataset.py --metadata data/simulations/metadata_clean.csv --config configs/v2_flat_unified_thickness.json
```

Expected sanity checks:

- `thickness_h` should stay within `0.0004` to `0.004`.
- `curvature_type` should be `flat`.
- Low-delta-T bins (`<=10K` and `10-15K`) should remain represented.
- The `>100K` bin should be materially larger than the v1 run.

## Outputs

- Clean metadata: `data/simulations/metadata_clean.csv`
- Split reports: `results/metadata/v2_flat_unified_thickness/`
- Evaluation: `results/evaluation/v2_flat_unified_thickness/`
- Benchmark: `results/real_world_benchmarks/v2_flat_unified_thickness/`
- Logs: `logs/*v2_flat_unified_thickness*`
