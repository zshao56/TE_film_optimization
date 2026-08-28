"""Sanity-check the planar_split geometry family before launching full data generation.

Runs a single small FDM solve for each structured family under identical physics and
prints the resulting in-plane temperature difference. The point of the check is the
claim that motivated adding planar_split: a full-thickness vertical interface should
beat the bottom-connected wedge families, because the wedge families always leave a
low-k cap over the whole top surface.

Usage (Linux, inside the venv, before the full run):

    python scripts/check_planar_split.py
    python scripts/check_planar_split.py --nx 50 --ny 50 --nz 20

This is a cheap check (a handful of solves), safe to run on any machine.
"""

import argparse
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from geometry.structured_library import (  # noqa: E402
    STRUCTURED_FAMILIES,
    generate_planar_split_structure,
    sample_structured_structure,
)
from postprocess.metrics import find_best_electrodes  # noqa: E402
from simulation.custom_solver import Custom3DFDMSolver  # noqa: E402


def solve_delta_t(geom, nx, ny, nz, T_hot, T_air, h_c):
    """Run one FDM solve and return the best-electrode in-plane delta T.

    Mirrors the measurement setup in src/main.py (window 5% of each in-plane
    dimension, minimum electrode gap 5% of Lx) so numbers are comparable with the
    database.
    """
    solver = Custom3DFDMSolver(geom, T_hot, T_air, h_c, h_c, nx=nx, ny=ny, nz=nz)
    try:
        mesh_data, field_data = solver.solve()
    except Exception as exc:  # noqa: BLE001 - diagnostic script, report and continue
        print(f"    solver raised: {type(exc).__name__}: {exc}")
        return None
    finally:
        solver.cleanup()

    if not field_data.get("solver_bounds_pass") or not field_data.get("surface_bounds_pass"):
        return None

    Lx = float(geom["Lx"])
    Ly = float(geom["Ly"])
    wx, wy = 0.05 * Lx, 0.05 * Ly
    s_min = 0.05 * Lx

    xv, yv = np.meshgrid(mesh_data["x"], mesh_data["y"], indexing="ij")
    result = find_best_electrodes(
        field_data["temperature_surface"], xv, yv, Lx, Ly, wx, wy, s_min
    )
    return None if result is None else result["delta_T_parallel"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=50)
    parser.add_argument("--ny", type=int, default=50)
    parser.add_argument("--nz", type=int, default=20)
    parser.add_argument("--Lx", type=float, default=0.01)
    parser.add_argument("--Ly", type=float, default=0.01)
    parser.add_argument("--thickness", type=float, default=0.001)
    parser.add_argument("--k-low", type=float, default=0.089)
    parser.add_argument("--k-high", type=float, default=4.492)
    parser.add_argument("--T-hot", type=float, default=350.0)
    parser.add_argument("--T-air", type=float, default=298.15)
    parser.add_argument("--h-c", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260513)
    args = parser.parse_args()

    geom_args = (args.Lx, args.Ly, args.thickness, args.k_low, args.k_high)
    grid = (args.nx, args.ny, args.nz)
    physics = dict(T_hot=args.T_hot, T_air=args.T_air, h_c=args.h_c)

    print(f"grid={grid}  Lx={args.Lx} Ly={args.Ly} h={args.thickness}")
    print(f"k_low={args.k_low} k_high={args.k_high} T_hot={args.T_hot} T_air={args.T_air} h_c={args.h_c}")

    print("\n=== planar_split: explicit configurations ===")
    explicit = [
        ("left/right, vertical", dict(split_fraction=0.5, interface_tilt=0.0, direction="x")),
        ("front/back, vertical", dict(split_fraction=0.5, interface_tilt=0.0, direction="y")),
        ("diagonal, vertical", dict(split_fraction=0.5, interface_tilt=0.0, direction="diagonal")),
        ("left/right, tilted +0.3", dict(split_fraction=0.5, interface_tilt=0.3, direction="x")),
        ("left/right, off-centre 0.35", dict(split_fraction=0.35, interface_tilt=0.0, direction="x")),
    ]

    planar_best = 0.0
    for label, kwargs in explicit:
        geom = generate_planar_split_structure(*geom_args, *grid, **kwargs)
        delta_t = solve_delta_t(geom, *grid, **physics)
        vf = geom["volume_fraction_actual"]
        if delta_t is None:
            print(f"  {label:32s} FAILED (no valid electrode pair)")
            continue
        planar_best = max(planar_best, delta_t)
        print(f"  {label:32s} delta_T={delta_t:8.4f} K   vol_frac={vf:.3f}")

    print("\n=== one random sample per family (same physics, seed-matched) ===")
    family_best = {}
    for family in STRUCTURED_FAMILIES:
        rng = np.random.default_rng(args.seed)
        geom = sample_structured_structure(*geom_args, *grid, family=family, rng=rng)
        delta_t = solve_delta_t(geom, *grid, **physics)
        family_best[family] = delta_t
        if delta_t is None:
            print(f"  {family:16s} FAILED (no valid electrode pair)")
        else:
            print(f"  {family:16s} delta_T={delta_t:8.4f} K   vol_frac={geom['volume_fraction_actual']:.3f}")

    print("\n=== verdict ===")
    wedge_like = [
        family_best.get(name)
        for name in ("wedge", "curved_wedge", "step", "arc")
        if family_best.get(name) is not None
    ]
    if not wedge_like or planar_best <= 0.0:
        print("  inconclusive: some solves failed, inspect the output above")
        return

    best_wedge = max(wedge_like)
    print(f"  best planar_split (explicit configs): {planar_best:.4f} K")
    print(f"  best wedge-like family (single sample): {best_wedge:.4f} K")
    if planar_best > best_wedge:
        print("  OK: planar_split exceeds the wedge-like families, matching the earlier observation.")
    else:
        print("  NOTE: planar_split did not win here. A single random sample per family is not a")
        print("  fair comparison (each family's own parameters are unoptimised), so this is not")
        print("  proof of a bug - but worth a look before committing to a full run.")


if __name__ == "__main__":
    main()
