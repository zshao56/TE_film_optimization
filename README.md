# TE Film Optimization

This repository focuses on the computational design and optimization of 3D thin thermoelectric (TE) film structures. The goal is to geometrically architect composite structures using two materials with different thermal conductivities to convert an out-of-plane thermal gradient into a maximized in-plane temperature difference on the top surface.

## 🌟 Key Features
- **3D Steady-State Heat Conduction**: A custom-built 3D Finite Difference Method (FDM) solver using `scipy.sparse` for efficient calculation of large voxel-based meshes, avoiding the need for heavy commercial software.
- **Top-Surface Electrode Modeling**: Replaces idealized point-based metrics with a physically realistic 2D area-averaged temperature measurement over specific electrode windows.
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

Run the automated simulation pipeline which will test the solver on a uniform baseline block and a 3D wedge structure:
```bash
python src/main.py
```

After execution:
- The target metric (`delta_T_parallel`) and execution details are appended to `data/simulations/metadata.csv`.
- The full 3D temperature and thermal conductivity fields are stored in `data/simulations/fields/<sim_id>.h5`.

## 📜 Physics Rules & Validation
All simulations must adhere to the rules strictly defined in `database_temperature_difference_protocol.md`. This ensures consistent physical conditions and comparable $\Delta T_{\parallel}$ objectives for any downstream machine learning tasks, preventing "cheating" by optimizing for unmanufacturable local artifacts.
