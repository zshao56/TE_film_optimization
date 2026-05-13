# TE Film Optimization

**English** | [中文](README_zh.md)

This repository focuses on the computational design and optimization of 3D thin thermoelectric (TE) film structures. The goal is to geometrically architect composite structures using two materials with different thermal conductivities to convert an out-of-plane thermal gradient into a maximized in-plane temperature difference on the top surface.

## 🌟 Key Features
- **3D Steady-State Heat Conduction**: A custom-built 3D Finite Difference Method (FDM) solver using `scipy.sparse` for efficient calculation of large voxel-based meshes, avoiding the need for heavy commercial software.
- **Top-Surface Electrode Modeling**: Replaces idealized point-based metrics with a physically realistic 2D area-averaged temperature measurement over specific electrode windows.
- **Structured Geometry Library**: Supports low-dimensional wedge, step, double-layer, and arc families so database generation is guided by interpretable structures instead of only random smoothed noise.
- **Automated Workflow Pipeline**: Supports automated geometry generation, 3D simulation, metric post-processing, and unified data storage (HDF5 field data + CSV metadata).
- **Physical Boundary Conditions**: Rigorously defined boundary conditions (Bottom: fixed hot temperature; Top and Sides: natural convection to ambient air) ensuring fair comparison across diverse topological variants.

## 📂 Project Structure
```text
TE_film_optimization/
├── src/
│   ├── geometry/        # 3D parameterization and voxel generation
│   ├── simulation/      # FDM and analytical solver engines
│   ├── postprocess/     # Metric extraction, electrode search
│   ├── optimization/    # Bayesian/Active learning loop (Planned)
│   ├── data_io/         # HDF5 and CSV metadata management
│   └── main.py          # Pipeline execution script
├── data/
│   └── simulations/     # Local storage for metadata.csv and HDF5 field files
├── results/             # Generated figures and analysis plots
├── plan.md              # Long-term strategy and milestones
└── database_temperature_difference_protocol.md # Core rules for simulation setup and validation
```

## 🔄 Project Pipeline: From Physics to Inverse Design

The ultimate goal of this project is **Inverse Design**: given constraints like film thickness and environmental temperatures, what is the theoretical optimal 3D shape that maximizes the temperature difference? We achieve this through a rigorous three-phase pipeline:

```mermaid
flowchart TD
    %% Styling
    classDef phase fill:#f9f9f9,stroke:#333,stroke-width:2px,color:#000,font-weight:bold;
    classDef data fill:#e1f5fe,stroke:#1565c0,color:#000;
    classDef model fill:#e8f5e9,stroke:#2e7d32,color:#000;
    classDef action fill:#fff3e0,stroke:#f57c00,color:#000;
    classDef highlight fill:#ffe0b2,stroke:#e65100,stroke-width:2px,color:#000,font-weight:bold;

    subgraph Phase1 [Phase 1: High-Fidelity Data Generation]
        direction TB
        A1[Randomize Environment\nThickness, K-values, Temperatures]:::action --> B1
        A2[Parameterized Geometry\nCurved Wedge, Step, Double-Layer]:::action --> B1
        B1{3D FDM Solver\n(Computational Cost: High)}:::model --> C1
        C1[(50,000 High-Fidelity Samples\nmetadata.csv + 3D .h5)]:::data
    end

    subgraph Phase2 [Phase 2: Forward Surrogate Model Training]
        direction TB
        C1 --> D1(Dataset Split 80/10/10)
        D1 --> E1[3D CNN Branch\nSpatial Topology]:::model
        D1 --> E2[MLP Branch\nPhysical Params]:::model
        E1 --> F1((Fusion Block))
        E2 --> F1
        F1 --> G1[Predict In-plane ΔT]:::data
        G1 -->|MSE Loss / Backpropagation| E1
        G1 -->|Save Best Weights| H1[Millisecond AI Referee\nThermoNet]:::highlight
    end

    subgraph Phase3 [Phase 3: Ultimate Goal - Inverse Design]
        direction TB
        I1[/User Input Constraints:\nFixed thickness & Temperatures/]:::data --> J1
        J1[AI Optimizer\nGenetic Algorithm / Gradient Ascent]:::action -->|1. Generate 100k candidate shapes| K1
        H1 -->|Deploy| K1{AI Referee Scoring\n(Cost: 1ms/inference)}:::highlight
        K1 -->|2. Return Predicted ΔT| J1
        J1 -->|3. Evolve over generations| L1[/Output: Theoretical Max ΔT\nand Optimal 3D Geometry/]:::data
    end

    %% Cross-phase links
    Phase1 ===> Phase2
    Phase2 ===> Phase3
    
    %% Validation Loop
    L1 -.->|Final High-Fidelity Verification| B1
```

### Pipeline Breakdown
1. **Phase 1 (Data Generation)**: We strictly constrain geometric generators to physical rules and utilize our custom FDM solver to generate a massive, noise-free database. This establishes the absolute physical ground truth.
2. **Phase 2 (Forward Training)**: We train a Multi-modal neural network (`ThermoNet`). The 3D CNN branch learns the spatial geometry, while the MLP branch understands the thermodynamic context. Once converged, this acts as a surrogate solver that is roughly 10,000x faster than traditional FEM.
3. **Phase 3 (Inverse Design)**: By deploying the trained surrogate model as a lightning-fast "fitness function", we can unleash Genetic Algorithms or Gradient Ascent to comb through millions of complex topological variations in seconds, hunting down the exact 3D architecture that yields the ultimate thermal gradient.

## 🛠 Installation

Requires **Python 3.8+**.

Clone the repository:
```bash
git clone https://github.com/zshao56/TE_film_optimization.git
cd TE_film_optimization
```

Install the dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

Run the automated simulation pipeline on a small batch of structured geometry examples:
```bash
python src/main.py
```

After execution:
- The target metric (`delta_T_parallel`) and execution details are appended to `data/simulations/metadata.csv`.
- The full 3D temperature and thermal conductivity fields are stored in `data/simulations/fields/<sim_id>.h5`.

For larger database generation, use mostly structured geometries with optional random exploration:
```bash
python src/generate_database.py --samples 50000 --cores 8 --mode mixed --structured-ratio 0.8 --seed 42
```

Use `--mode structured` to exclude random-smoothed topologies, or `--mode random` to reproduce the old noise-filtering workflow.

For the v2 flat unified-thickness application database, use the configured pipeline. It samples thickness uniformly from `0.0004` to `0.004` m, keeps all generated cases flat, balances low/mid/high hot-boundary temperature differences, and records natural-to-strong-forced convection regimes:
```bash
python src/optimization/run_configured_pipeline.py --config configs/v2_flat_unified_thickness.json --stages data_generation
```

For full reproducibility, the whole workflow can be driven from one JSON config file. Edit `configs/v2_flat_unified_thickness.json` to change database sampling ranges, training hyperparameters, GPU visibility, and real-world verification scenarios. The top-level `run` block controls which stages execute:

```json
"run": {
  "data_generation": false,
  "training": false,
  "evaluation": false,
  "real_world_benchmark": true
}
```

Then run:

```bash
python -u src/optimization/run_configured_pipeline.py --config configs/v2_flat_unified_thickness.json
```

The same config can also be passed directly to data generation:

```bash
python src/generate_database.py --config configs/v2_flat_unified_thickness.json
```

When generating the expanded database from scratch, clear old metadata and HDF5 fields first so legacy and expanded schemas do not mix:
```bash
rm -f data/simulations/metadata.csv
rm -f data/simulations/fields/*.h5
mkdir -p data/simulations/fields logs
```

Train the surrogate model:
```bash
python src/optimization/train.py --batch-size 32 --epochs 50 --seed 42
```

For v2 training, include the hot-boundary temperature map as a second CNN channel and use the combined high-delta-T/low-delta-T loss:
```bash
python src/optimization/run_configured_pipeline.py --config configs/v2_flat_unified_thickness.json --stages training
```

If the first run clearly underestimates the high `delta_T_parallel` region, use the second-stage training setup:
```bash
python src/optimization/train.py --batch-size 32 --epochs 80 --seed 42 --normalize-target --top-quantile 0.9 --top-weight 3.0
```

After training, evaluate the model on the held-out test split before adding more epochs:
```bash
python src/optimization/evaluate.py --split test --seed 42
```

The report is written to `results/evaluation/` and includes overall MAE/RMSE/R², per-geometry-family errors, prediction-vs-FDM scatter plots, a separate error check for the high `delta_T_parallel` region, and top-region ranking hit rates. Move to surrogate-assisted inverse design only if the test split, especially the high-temperature-difference region, is accurate enough.

To let the training machine automatically sweep high-`delta_T_parallel` underprediction penalties, evaluate each run, and maintain a leaderboard:
```bash
python src/optimization/run_experiments.py --penalties 0.05 0.1 0.15 0.2 0.25 --batch-size 128
```

To let the local advisor inspect each evaluation result and choose the next run automatically:
```bash
python src/optimization/run_experiments.py --adaptive --max-adaptive-runs 4 --batch-size 128
```

If a manual training run has already finished and the current checkpoint is still at `results/models/best_thermonet.pth`, import that run into the experiment leaderboard first:
```bash
python src/optimization/run_experiments.py --import-current-run thermonet_v6_underpredict_0p1_bs128 --no-sweep
```

Automated experiment outputs are written under `results/experiments/`. The script preserves each run's checkpoint, evaluation metrics, prediction CSVs, figures, a ranked `leaderboard.csv`, and advisor reasoning in `advisor_decisions.json`, so later training runs do not overwrite the model being compared.

For the first surrogate-assisted inverse-design pass, screen a large candidate pool with the selected surrogate, then verify the top candidates with the real FDM solver:
```bash
python src/optimization/inverse_design.py screen --model-path results/experiments/thermonet_auto_adaptive_under_0p2_bs128/best_thermonet.pth --num-candidates 100000 --top-k 500 --batch-size 256 --mode mixed --structured-ratio 0.9 --seed 20260511
```

For fixed-condition inverse design, explicitly fix the operating condition and film thickness. Leave material values variable when the goal is to find the best material/geometry combination under the same thermal boundary condition:
```bash
python src/optimization/inverse_design.py screen --model-path results/experiments/thermonet_auto_adaptive_under_0p2_bs128/best_thermonet.pth --num-candidates 100000 --top-k 500 --batch-size 256 --mode mixed --structured-ratio 0.9 --seed 20260511 --fixed-h 0.001 --fixed-T-hot 350.0 --fixed-T-air 298.15 --fixed-h-c 10.0 --fixed-h-c-side 10.0
```

Screening outputs are written to `results/inverse_design/screen_<timestamp>/`. Then verify the first 50 candidates from that directory:
```bash
python src/optimization/inverse_design.py verify --screen-dir results/inverse_design/screen_<timestamp> --verify-count 50
```

The `verify` command skips candidates that are already present in `verified_candidates.csv`, so this extends the same verification file to the top 200 without recomputing the first 50:
```bash
python src/optimization/inverse_design.py verify --screen-dir results/inverse_design/screen_<timestamp> --verify-count 200
```

To export one combined figure for the top 10 structures ranked by real FDM `fdm_delta_T`, save the PNG into the same folder as `verified_candidates.csv`:
```bash
python src/optimization/inverse_design.py plot-top --screen-dir results/inverse_design/screen_<timestamp> --top-n 10
```

The `screen` command only runs neural-network inference. The `verify` command runs FDM, appends verified simulations to the database, and writes `verified_candidates.csv`. The final candidate ranking should use the FDM values in `verified_candidates.csv`, not the surrogate rank.

## 🌍 Real-World Scenario Benchmark

For fairer application-oriented comparisons, use the real-world benchmark runner instead of unconstrained inverse design. The script fixes each operating scenario's hot-boundary map, curvature, ambient temperature, and convection strength, while still letting the surrogate search over film thickness, material contrast, and geometry. Each scenario is then verified with the FDM solver.

The current benchmark scenarios are:

| Scenario | Hot-boundary condition | Curvature | Convection |
| :--- | :--- | :--- | :--- |
| Battery surface cooling | Center hotspot 390 K, edge 360 K | Flat | Strong forced/AC cooling (`h_c=180`) |
| Skin patch | Uniform 310.15 K (37 C) | Flat | Natural convection (`h_c=8`) |
| Glass panel | Center hotspot 343.15 K (70 C), edge 323.15 K (50 C) | Flat | Natural convection (`h_c=8`) |
| Automotive engine surface | Center hotspot 433 K, edge 373 K | Flat | Strong forced driving airflow (`h_c=300`) |
| Phone surface | Linear gradient from 303.15 K to 333.15 K | Flat | Natural convection (`h_c=8`) |

Run all five scenarios with one command:

```bash
python -u src/optimization/run_configured_pipeline.py \
  --config configs/v2_flat_unified_thickness.json \
  --stages real_world_benchmark
```

The output directory contains one subfolder per scenario, plus:

```text
scenario_definitions.csv
benchmark_summary.csv
<scenario_key>/screened_candidates.csv
<scenario_key>/top_candidates.csv
<scenario_key>/verified_candidates.csv
<scenario_key>/top_candidate_masks.npz
```

Note: the v2 sampling explicitly includes low, mid, and high hot-boundary temperature-difference bands. After filtering, run `scripts/check_v2_dataset.py` to confirm that the low-delta-T and high-delta-T regions remain represented.

## 📐 Grid Independence and Mesh Selection

For the massive database generation (e.g., 50,000 samples), selecting the right mesh resolution is crucial. A **Grid Independence Test** was performed on the highly sensitive `curved_wedge` structure to evaluate accuracy versus computational cost:

| Mesh (nx, ny, nz) | DoF (Nodes) | $\Delta T_{\parallel}$ (K) | Time/Sample (s) |
| :--- | :--- | :--- | :--- |
| (40, 40, 15) | 24,000 | 3.3118 | ~3.2 |
| **(50, 50, 20)** | **50,000** | **3.5888** | **~9.3** |
| (60, 60, 30) | 108,000 | 3.5294 | ~63.2 |

**Conclusion:** The configuration `nx=50, ny=50, nz=20` was chosen as the default. It successfully smooths out truncation errors on curved interfaces (capturing a more accurate temperature gradient) while keeping the solver matrix small enough to compute a 50,000-sample database in approximately 15-18 hours on an 8-core machine. Finer meshes (e.g., 60x60x30) result in an exponential time increase with diminishing returns in physical accuracy.

## 📜 Physics Rules & Validation
All simulations must adhere to the rules strictly defined in `database_temperature_difference_protocol.md`. This ensures consistent physical conditions and comparable $\Delta T_{\parallel}$ objectives for any downstream machine learning tasks, preventing "cheating" by optimizing for unmanufacturable local artifacts.
