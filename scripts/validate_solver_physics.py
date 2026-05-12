import os
import sys

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from geometry.structured_library import generate_wedge_structure
from simulation.custom_solver import Custom3DFDMSolver


def _assert(condition, message):
    if not condition:
        raise AssertionError(message)


def validate_uniform_1d_case():
    Lx, Ly, h = 0.01, 0.01, 0.002
    k = 1.7
    T_hot = 350.0
    T_air = 298.15
    h_c = 10.0
    h_c_side = 0.0

    geom = {
        "geometry_type": "uniform",
        "Lx": Lx,
        "Ly": Ly,
        "h": h,
        "k_low": k,
        "k_high": k,
        "k_val": k,
    }

    solver = Custom3DFDMSolver(geom, T_hot, T_air, h_c, h_c_side, nx=30, ny=30, nz=12)
    _mesh, fields = solver.solve()
    surface = fields["temperature_surface"]

    heat_flux = (T_hot - T_air) / (h / k + 1.0 / h_c)
    expected_top = T_hot - heat_flux * h / k
    surface_range = float(np.max(surface) - np.min(surface))
    max_surface_error = float(np.max(np.abs(surface - expected_top)))

    _assert(fields["solver_bounds_pass"] == 1, "Uniform 1D case failed volume bounds check.")
    _assert(fields["surface_bounds_pass"] == 1, "Uniform 1D case failed surface bounds check.")
    _assert(
        fields["solver_relative_residual"] <= fields["solver_residual_tolerance"],
        "Uniform 1D case residual exceeded solver tolerance.",
    )
    _assert(
        surface_range < 1e-3,
        f"Uniform 1D case has lateral numerical noise {surface_range:.6g} K.",
    )
    _assert(
        max_surface_error < 2e-3,
        f"Uniform 1D case top temperature error is {max_surface_error:.6g} K.",
    )
    return {
        "surface_range_K": surface_range,
        "max_surface_error_K": max_surface_error,
        "relative_residual": fields["solver_relative_residual"],
        "solver_method_code": fields["solver_method_code"],
    }


def validate_heterogeneous_bounds_case():
    Lx, Ly, h = 0.01, 0.01, 0.0012
    T_hot = 360.0
    T_air = 300.0
    geom = generate_wedge_structure(
        Lx,
        Ly,
        h,
        k_low=0.2,
        k_high=5.0,
        nx=30,
        ny=30,
        nz=12,
        volume_fraction_target=0.45,
        wedge_slope=1.0,
        direction="x",
    )

    solver = Custom3DFDMSolver(geom, T_hot, T_air, h_c=5.0, h_c_side=3.0, nx=30, ny=30, nz=12)
    _mesh, fields = solver.solve()
    surface = fields["temperature_surface"]

    _assert(fields["solver_bounds_pass"] == 1, "Heterogeneous case failed volume bounds check.")
    _assert(fields["surface_bounds_pass"] == 1, "Heterogeneous case failed surface bounds check.")
    _assert(float(np.min(surface)) >= min(T_air, T_hot) - 1e-3, "Surface fell below thermal lower bound.")
    _assert(float(np.max(surface)) <= max(T_air, T_hot) + 1e-3, "Surface exceeded thermal upper bound.")
    return {
        "surface_min_K": float(np.min(surface)),
        "surface_max_K": float(np.max(surface)),
        "surface_range_K": float(np.max(surface) - np.min(surface)),
        "relative_residual": fields["solver_relative_residual"],
        "solver_method_code": fields["solver_method_code"],
    }


def main():
    results = {
        "uniform_1d": validate_uniform_1d_case(),
        "heterogeneous_bounds": validate_heterogeneous_bounds_case(),
    }
    print("Solver physics validation passed.")
    for name, metrics in results.items():
        print(f"{name}:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
