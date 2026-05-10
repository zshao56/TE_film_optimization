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

Train the surrogate model:
```bash
python src/optimization/train.py --batch-size 32 --epochs 50 --seed 42
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

If a manual training run has already finished and the current checkpoint is still at `results/models/best_thermonet.pth`, import that run into the experiment leaderboard first:
```bash
python src/optimization/run_experiments.py --import-current-run thermonet_v6_underpredict_0p1_bs128 --no-sweep
```

Automated experiment outputs are written under `results/experiments/`. The script preserves each run's checkpoint, evaluation metrics, prediction CSVs, figures, and a ranked `leaderboard.csv`, so later training runs do not overwrite the model being compared.

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
