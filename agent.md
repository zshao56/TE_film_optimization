# Agent Guide for TE Film Optimization

## Role

You are working on a thermal-gradient conversion project for thin thermoelectric films. The project optimizes two-material geometries with thickness $h \le 2\ \mathrm{mm}$ to convert an out-of-plane thermal gradient into an in-plane temperature difference.

Your job is to build reproducible simulation, data, postprocessing, and optimization code. Do not treat the machine learning model as the source of truth. FEM or an equivalent physics solver is the source of truth for final structure ranking.

## Must Read First

Before writing or changing code, read:

1. `plan.md`
2. `database_temperature_difference_protocol.md`
3. This `agent.md`

If there is a conflict, follow this priority:

1. `database_temperature_difference_protocol.md`
2. `agent.md`
3. `plan.md`

The temperature-difference protocol is the strictest document because it defines the target used for database construction and machine learning.

## Project Goal

Given:

- film thickness $h$
- in-plane length $L$
- material conductivities $\kappa_{\mathrm{low}}$, $\kappa_{\mathrm{high}}$
- a parametric structure $\theta$

compute:

- temperature field $T(x,z)$
- heat flux $\mathbf{q}(x,z)$
- top-electrode temperature difference $\Delta T_{\parallel}$
- best hot and cold electrode-window positions
- conversion efficiency $\eta_T$

Then use these values to construct a high-quality simulation database and optimize structures.

## Non-Negotiable Physical Definitions

The first-stage physics model is 2D steady-state heat conduction:

$$
\nabla \cdot \left(\kappa(x,z;\theta)\nabla T(x,z)\right)=0
$$

Heat flux:

$$
\mathbf{q}(x,z)=-\kappa(x,z;\theta)\nabla T(x,z)
$$

The main target is not a single-point temperature difference. Because the thermoelectric material and electrodes are placed on top of the substrate, the main target must search for two fixed-size top electrode-contact windows with the largest average temperature difference.

$$
\Omega^{elec}(x_c)=
\left[x_c-\frac{w_m}{2}, x_c+\frac{w_m}{2}\right]\times[h-t_m,h]
$$

The legal center positions satisfy:
$$
\frac{w_m}{2}\le x_c\le L-\frac{w_m}{2}
$$

Two electrode windows must not overlap:

$$
|x_{c,1}-x_{c,2}| \ge w_m+s_{\min}
$$

Window average temperature:
$$
\overline{T}_{elec}(x_c)=
\frac{1}{|\Omega^{elec}(x_c)|}
\int_{\Omega^{elec}(x_c)}T(x,z)\,d\Omega
$$

Find the best hot and cold electrode-window pair:

$$
(x_a^\*,x_b^\*)=
\arg\max_{x_a,x_b}
\left|\overline{T}_{elec}(x_a)-\overline{T}_{elec}(x_b)\right|
$$

Then reorder the pair by temperature to define `x_hot_electrode` and `x_cold_electrode`, and compute:

$$
\Delta T_{\parallel}=\overline{T}_{hot}-\overline{T}_{cold}
$$

Main optimization target:

$$
\eta_T=\frac{\Delta T_{\parallel}}{\Delta T_{\perp}}
$$

For fixed-temperature top and bottom boundaries:

$$
\Delta T_{\perp}=T_{\mathrm{hot}}-T_{\mathrm{cold}}
$$

However, do not use a whole-top-surface fixed-temperature boundary when the target is the top-electrode temperature difference. A uniform top Dirichlet boundary destroys the top temperature distribution being measured.

Do not replace this target with endpoint, centerline, mid-film, or single-point maximum-minimum temperatures.

## Default Boundary Condition

Use `BC-001-TOP-ELECTRODE` unless the user or a task explicitly asks for another boundary condition.

`BC-001-TOP-ELECTRODE`:

- bottom boundary $z=0$: fixed thermal source, for example $T=T_{\mathrm{hot}}$
- top boundary $z=h$: convection, finite contact resistance, or an explicit thermoelectric layer model; do not use a whole-surface fixed temperature on the top measurement surface
- left and right boundaries $x=0,L$: adiabatic, $\mathbf{q}\cdot\mathbf{n}=0$

Different boundary conditions must use a new `boundary_condition_id`. Do not mix results from different boundary conditions in the same target column unless the model explicitly includes boundary-condition features.

## Measurement Region Defaults

Unless otherwise specified:

- $w_m=0.03L$
- $t_m=0.03h$
- $s_{\min}=0.02L$
- $z_1=h-t_m$
- $z_2=h$

These values must be stored in `metadata.csv` for every simulation.

Measurement regions must be fixed for a given dataset. Do not adjust them per geometry to make a structure look better.

## Directory Structure

Use this structure for new project code and generated data:

```text
src/
  geometry/
  simulation/
  postprocess/
  optimization/
  data_io/
data/
  simulations/
    metadata.csv
    boundary_conditions.csv
    materials.csv
    fields/
results/
  figures/
  optimized_structures/
```

Rules:

- Put Python source files under `src/`.
- Put raw and reusable simulation data under `data/`.
- Put generated figures, reports, and optimization outputs under `results/`.
- Do not write generated figures into the repository root.

## Recommended Code Modules

Use small modules with clear responsibilities.

### `src/geometry/`

Responsible for geometry generation and validation.

Expected functions:

- create wedge structures
- create step structures
- create wave or sinusoidal interface structures
- compute material labels
- compute volume fraction
- check self-intersection
- check minimum feature size
- check material-region connectivity

### `src/simulation/`

Responsible for solver setup and execution.

Expected functions:

- build mesh
- assign material conductivity
- apply boundary conditions
- solve steady-state heat equation
- compute heat flux
- return a structured simulation result

### `src/postprocess/`

Responsible for all target calculations.

Expected functions:

- compute region-average temperature
- compute $\Delta T_{\parallel}$
- compute $\Delta T_{\perp}$
- compute $\eta_T$
- compute centerline diagnostic temperature difference
- compute line-average diagnostic temperature difference
- compute heat-flux redirect ratio
- perform energy-conservation check

The target calculation must live here. Do not duplicate target logic in optimization scripts or notebooks.

### `src/data_io/`

Responsible for database writing and reading.

Expected functions:

- create or append `metadata.csv`
- write HDF5 field files
- read HDF5 field files
- validate required metadata columns
- load a clean training dataset using only `qc_pass = true`

### `src/optimization/`

Responsible for parameter search and surrogate models.

Expected functions:

- baseline parameter sweep
- Bayesian optimization or active learning loop
- candidate selection
- FEM re-evaluation of model-suggested candidates

Final reported results must use FEM re-evaluation, not only surrogate predictions.

## Required Metadata Columns

Every simulation record must include at least:

```text
simulation_id
geometry_type
geometry_parameters
thickness_h
length_L
width_W
k_low
k_high
k_ratio
volume_fraction_high
boundary_condition_id
T_hot
T_cold
delta_T_perp
measurement_width
measurement_depth
electrode_min_gap
measurement_z1
measurement_z2
x_hot_electrode
x_cold_electrode
T_hot_electrode_avg
T_cold_electrode_avg
delta_T_parallel
delta_T_parallel_signed_x
eta_T
delta_T_midline
delta_T_line_avg
delta_T_top_point_max
heat_flux_redirect_ratio
mesh_element_count
mesh_min_quality
mesh_convergence_error
solver_converged
qc_pass
field_file
created_at
```

Use SI units in stored data:

- length in m
- temperature in K
- thermal conductivity in W m^-1 K^-1

## HDF5 Field File Requirements

Each simulation must have one HDF5 file under `data/simulations/fields/`.

Required groups or datasets:

```text
/mesh/nodes
/mesh/elements
/fields/temperature
/fields/qx
/fields/qz
/fields/material_id
/geometry/boundary_points
/postprocess/electrode_window_hot
/postprocess/electrode_window_cold
/postprocess/electrode_window_candidates
```

The HDF5 file must contain enough information to reproduce:

- geometry plot
- material map
- temperature field
- heat flux field
- measurement-region overlay

## Quality Control

Set `qc_pass = true` only if all relevant checks pass.

Required checks:

1. solver converged
2. geometry is valid
3. measurement regions are inside the domain
4. material labels exist for the whole domain
5. no missing or NaN values in temperature field
6. energy-conservation relative error is below 1%
7. representative mesh-convergence error is below 2% for $\eta_T$

For early prototypes, if a check is not implemented yet, mark `qc_pass = false` or store a clear placeholder field explaining why. Do not silently pass incomplete samples into training data.

## Testing Expectations

When adding simulation or postprocessing code, include tests or reproducible checks for:

1. uniform material under `BC-001-TOP-ELECTRODE`
   - expected $\Delta T_{\parallel}$ should be near zero for symmetric uniform geometry

2. left-right mirrored structures
   - $x_{hot}$ and $x_{cold}$ should mirror around the center line
   - $\Delta T_{\parallel}$ and $\eta_T$ should remain approximately the same

3. measurement-region sensitivity
   - increasing sampling density should change $\overline{T}_L$, $\overline{T}_R$, and $\eta_T$ by less than 1%

4. mesh convergence
   - medium and fine meshes should change $\eta_T$ by less than 2% for representative structures

5. database round trip
   - a written HDF5 and metadata record can be read back and reproduce the same scalar metrics

## Plotting and Output

Every representative simulation should be able to generate:

- material map
- temperature field
- heat flux vector or streamline plot
- measurement-region overlay
- convergence or optimization history plot when relevant

Save figures under `results/figures/`.

Use project plotting utilities if available. Keep plot code reproducible and avoid manual edits to final figures.

## Machine Learning and Optimization Rules

Use ML as a surrogate or acquisition tool, not as final proof.

Allowed early-stage approaches:

- random or Latin hypercube initial sampling
- Gaussian process regression
- random forest
- XGBoost or LightGBM
- Bayesian optimization
- active learning
- CMA-ES for medium-dimensional parameter spaces

Avoid in early implementation:

- reinforcement learning
- large neural networks
- topology optimization without validated baseline FEM
- training on failed or mixed-boundary-condition samples

Every ML-recommended best structure must be re-simulated using the physics solver before being reported.

## Coding Style

- Prefer clear physical variable names: `thickness_h`, `length_L`, `k_low`, `k_high`, `x_hot_electrode`, `x_cold_electrode`, `delta_T_parallel`, `eta_T`.
- Keep units explicit in variable names or docstrings.
- Put formulas in docstrings for functions that compute physical targets.
- Avoid duplicating formulas across modules.
- Keep generated data paths configurable but default to `data/` and `results/`.
- Write deterministic code when possible; store random seeds for sampling and optimization.

## Do Not Do

- Do not optimize using single-point endpoint temperatures.
- Do not measure the primary objective at the film midline; the primary measurement is the best pair of top electrode-contact windows.
- Do not use single-point top maximum and minimum temperatures as the primary objective.
- Do not use a whole-top-surface fixed-temperature boundary when evaluating top-electrode temperature difference.
- Do not compare samples with different boundary conditions unless explicitly modeled.
- Do not report surrogate predictions as final optimized performance.
- Do not train on `qc_pass = false` samples.
- Do not generate structures with self-intersections or unresolved sub-mesh features.
- Do not save results without enough metadata to reproduce the calculation.
- Do not place generated source code, data, or figures in arbitrary root-level files.

## First Implementation Milestone

The first code milestone should build a minimal but complete loop:

1. Generate one uniform baseline and one simple two-material wedge structure.
2. Solve the 2D steady-state heat equation.
3. Search the best top hot/cold electrode-window pair and compute $\Delta T_{\parallel}$, $\eta_T$.
4. Save metadata and HDF5 field data.
5. Generate temperature, heat-flux, and measurement-region figures.
6. Verify that the uniform baseline gives near-zero in-plane temperature difference.

This milestone is more important than adding many structure types. It proves that the target definition, data format, and solver interface are correct.
