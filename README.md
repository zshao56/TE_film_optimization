# TE Film Optimization

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
