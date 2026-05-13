from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from postprocess.metrics import find_best_electrodes
from simulation.custom_solver import Custom3DFDMSolver


SUMMARY_DIR = ROOT / "results" / "expanded_rebuild_v1_summary"
BENCH_DIR = (
    SUMMARY_DIR
    / "full_plot_inputs"
    / "results"
    / "real_world_benchmarks"
    / "expanded_rebuild_v1"
)
ASSET_DIR = (
    ROOT
    / "outputs"
    / "manual-20260513-173312"
    / "presentations"
    / "expanded-rebuild-v1-results"
    / "assets"
)


SCENARIOS = [
    ("battery_ac_half_cylinder", "Battery"),
    ("skin_slight_curve_uniform", "Skin"),
    ("glass_center_hot_natural", "Glass"),
    ("engine_forced_flat", "Engine"),
    ("phone_linear_natural", "Phone"),
]


def savefig(fig: plt.Figure, name: str) -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    for out_dir in (SUMMARY_DIR, ASSET_DIR):
        fig.savefig(out_dir / name, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def best_candidate_data(scenario_key: str, benchmark: pd.DataFrame):
    scenario_dir = BENCH_DIR / scenario_key
    best_id = benchmark.loc[scenario_key, "best_candidate_id"]
    top_df = pd.read_csv(scenario_dir / "top_candidates.csv")
    row = top_df[top_df["candidate_id"] == best_id].iloc[0].to_dict()
    geom = json.loads(row["geometry_parameters"])

    npz = np.load(scenario_dir / "top_candidate_masks.npz")
    ids = npz["candidate_ids"].astype(str)
    idx = int(np.where(ids == best_id)[0][0])
    geom["mask_3d"] = npz["masks"][idx].astype(bool)
    geom["T_hot_map"] = npz["hot_boundary_maps"][idx].astype(np.float64)
    return best_id, row, geom


def solve_electrodes(geom: dict):
    nx, ny, nz = geom["mask_3d"].shape
    solver = Custom3DFDMSolver(
        geom,
        T_hot=float(geom["T_hot"]),
        T_air=float(geom["T_air"]),
        h_c=float(geom["h_c"]),
        h_c_side=float(geom["h_c_side"]),
        nx=nx,
        ny=ny,
        nz=nz,
    )
    mesh, field = solver.solve()
    X, Y = np.meshgrid(mesh["x"], mesh["y"], indexing="ij")
    wx = 0.05 * float(geom["Lx"])
    wy = 0.05 * float(geom["Ly"])
    s_min = 0.05 * float(geom["Lx"])
    electrodes = find_best_electrodes(
        field["temperature_surface"],
        X,
        Y,
        float(geom["Lx"]),
        float(geom["Ly"]),
        wx,
        wy,
        s_min,
    )
    return mesh, field, electrodes, wx, wy, s_min


def plot_boundary_schematic() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    film = Rectangle((2.0, 1.7), 6.0, 1.45, facecolor="#dbeafe", edgecolor="#1e40af", linewidth=2)
    ax.add_patch(film)
    ax.text(5.0, 2.42, "TE film / high-low k 3D structure", ha="center", va="center", fontsize=13, weight="bold", color="#1e3a8a")

    ax.add_patch(Rectangle((2.0, 1.25), 6.0, 0.35, facecolor="#ef4444", edgecolor="#991b1b", linewidth=1.5))
    ax.text(5.0, 0.96, "Hot boundary: bottom surface z = 0\nT_hot_map(x,y) or uniform T_hot", ha="center", va="top", fontsize=11, color="#991b1b")

    ax.annotate("", xy=(2.4, 1.65), xytext=(2.4, 0.75), arrowprops=dict(arrowstyle="->", lw=2, color="#dc2626"))
    ax.annotate("", xy=(7.6, 1.65), xytext=(7.6, 0.75), arrowprops=dict(arrowstyle="->", lw=2, color="#dc2626"))

    for x in np.linspace(2.4, 7.6, 6):
        ax.annotate("", xy=(x, 3.45), xytext=(x, 3.15), arrowprops=dict(arrowstyle="->", lw=1.7, color="#2563eb"))
    ax.text(5.0, 3.82, "Cold-side cooling: top surface z = h convects to T_air", ha="center", fontsize=10.8, color="#1d4ed8")

    ax.annotate("", xy=(1.82, 2.45), xytext=(1.15, 2.45), arrowprops=dict(arrowstyle="->", lw=1.7, color="#2563eb"))
    ax.annotate("", xy=(8.18, 2.45), xytext=(8.85, 2.45), arrowprops=dict(arrowstyle="->", lw=1.7, color="#2563eb"))
    ax.text(1.1, 2.9, "side convection\nto T_air", ha="center", fontsize=10, color="#1d4ed8")
    ax.text(8.9, 2.9, "side convection\nto T_air", ha="center", fontsize=10, color="#1d4ed8")

    ax.add_patch(Rectangle((3.0, 3.18), 0.65, 0.26, facecolor="#f97316", edgecolor="#9a3412", linewidth=1.3))
    ax.add_patch(Rectangle((6.35, 3.18), 0.65, 0.26, facecolor="#38bdf8", edgecolor="#075985", linewidth=1.3))
    ax.text(2.75, 3.28, "hot electrode\non top surface", ha="right", va="center", fontsize=9.2, color="#9a3412")
    ax.text(7.25, 3.28, "cold electrode\non top surface", ha="left", va="center", fontsize=9.2, color="#075985")

    ax.text(
        5.0,
        4.95,
        "Boundary-condition and terminal definition used in the FDM target",
        ha="center",
        fontsize=16,
        weight="bold",
        color="#111827",
    )
    ax.text(
        5.0,
        4.58,
        "The thermal hot boundary is the bottom face. The reported Delta T_parallel is measured between two top-surface electrode windows.",
        ha="center",
        fontsize=10.5,
        color="#475569",
    )
    savefig(fig, "fig_hot_cold_boundary_definition.png")


def plot_electrode_locations(results: list[dict]) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(17.0, 3.9), constrained_layout=True)
    for ax, item in zip(axes, results):
        surf = item["surface"]
        geom = item["geom"]
        elec = item["electrodes"]
        wx_mm = item["wx"] * 1000.0
        wy_mm = item["wy"] * 1000.0
        extent = [0, float(geom["Ly"]) * 1000.0, 0, float(geom["Lx"]) * 1000.0]
        ax.imshow(surf, origin="lower", extent=extent, cmap="inferno", aspect="equal")
        hot_x = elec["x_hot"] * 1000.0
        hot_y = elec["y_hot"] * 1000.0
        cold_x = elec["x_cold"] * 1000.0
        cold_y = elec["y_cold"] * 1000.0

        ax.add_patch(
            Rectangle(
                (hot_y - wy_mm / 2, hot_x - wx_mm / 2),
                wy_mm,
                wx_mm,
                facecolor="none",
                edgecolor="#22c55e",
                linewidth=2.2,
            )
        )
        ax.add_patch(
            Rectangle(
                (cold_y - wy_mm / 2, cold_x - wx_mm / 2),
                wy_mm,
                wx_mm,
                facecolor="none",
                edgecolor="#38bdf8",
                linewidth=2.2,
            )
        )
        ax.scatter([hot_y], [hot_x], c="#22c55e", s=26, marker="o", edgecolors="white", linewidth=0.8)
        ax.scatter([cold_y], [cold_x], c="#38bdf8", s=26, marker="o", edgecolors="white", linewidth=0.8)
        ax.set_title(
            f"{item['label']}\nHot ({hot_x:.1f},{hot_y:.1f}) mm | Cold ({cold_x:.1f},{cold_y:.1f}) mm",
            fontsize=9.2,
        )
        ax.set_xlabel("y (mm)", fontsize=8)
        ax.set_ylabel("x (mm)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.text(
            0.03,
            0.05,
            f"Delta T={elec['delta_T_parallel']:.1f} K",
            transform=ax.transAxes,
            fontsize=8,
            color="white",
            bbox=dict(facecolor="black", alpha=0.42, edgecolor="none", pad=2),
        )

    fig.suptitle(
        "Recovered hot/cold terminal locations on the top surface",
        fontsize=15,
        fontweight="bold",
        x=0.02,
        ha="left",
    )
    fig.text(
        0.02,
        0.02,
        "Each panel is independently color-scaled by its top-surface temperature field. Green square = hot terminal; cyan square = cold terminal.",
        fontsize=10,
        color="#475569",
    )
    savefig(fig, "fig_best_hot_cold_terminal_locations.png")


def main() -> None:
    benchmark = pd.read_csv(BENCH_DIR / "benchmark_summary.csv").set_index("scenario_key")
    rows = []
    plot_items = []
    for key, label in SCENARIOS:
        print(f"Solving {key}...")
        best_id, row, geom = best_candidate_data(key, benchmark)
        mesh, field, electrodes, wx, wy, s_min = solve_electrodes(geom)
        if electrodes is None:
            raise RuntimeError(f"No electrode pair found for {key}")
        rows.append(
            {
                "scenario_key": key,
                "scenario_label": label,
                "candidate_id": best_id,
                "geometry_type": geom["geometry_type"],
                "Lx_mm": float(geom["Lx"]) * 1000.0,
                "Ly_mm": float(geom["Ly"]) * 1000.0,
                "h_mm": float(geom["h"]) * 1000.0,
                "hot_boundary_location": "bottom surface z=0",
                "cold_boundary_location": "top surface z=h and side surfaces convect to T_air",
                "terminal_surface": "top surface z=h",
                "electrode_window_x_mm": wx * 1000.0,
                "electrode_window_y_mm": wy * 1000.0,
                "minimum_electrode_gap_mm": s_min * 1000.0,
                "x_hot_electrode_mm": electrodes["x_hot"] * 1000.0,
                "y_hot_electrode_mm": electrodes["y_hot"] * 1000.0,
                "x_cold_electrode_mm": electrodes["x_cold"] * 1000.0,
                "y_cold_electrode_mm": electrodes["y_cold"] * 1000.0,
                "T_hot_electrode_avg_K": electrodes["T_hot_avg"],
                "T_cold_electrode_avg_K": electrodes["T_cold_avg"],
                "delta_T_parallel_K": electrodes["delta_T_parallel"],
            }
        )
        plot_items.append(
            {
                "key": key,
                "label": label,
                "geom": geom,
                "surface": field["temperature_surface"],
                "electrodes": electrodes,
                "wx": wx,
                "wy": wy,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(SUMMARY_DIR / "hot_cold_terminal_locations.csv", index=False)
    (SUMMARY_DIR / "hot_cold_terminal_locations.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    plot_boundary_schematic()
    plot_electrode_locations(plot_items)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
