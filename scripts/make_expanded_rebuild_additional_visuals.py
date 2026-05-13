from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize


ROOT = Path(__file__).resolve().parents[1]
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
    ("battery_ac_half_cylinder", "Battery", "#2563eb"),
    ("skin_slight_curve_uniform", "Skin", "#16a34a"),
    ("glass_center_hot_natural", "Glass", "#7c3aed"),
    ("engine_forced_flat", "Engine", "#dc2626"),
    ("phone_linear_natural", "Phone", "#ea580c"),
]


def savefig(fig: plt.Figure, name: str) -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    for out_dir in (SUMMARY_DIR, ASSET_DIR):
        fig.savefig(out_dir / name, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def relative_metrics() -> pd.DataFrame:
    rows = []
    for key, label, color in SCENARIOS:
        verified = pd.read_csv(BENCH_DIR / key / "verified_candidates.csv")
        err = verified["predicted_delta_T"] - verified["fdm_delta_T"]
        rel = err / verified["fdm_delta_T"]
        best_idx = verified["fdm_delta_T"].idxmax()
        rows.append(
            {
                "scenario_key": key,
                "scenario": label,
                "mean_fdm_delta_T": verified["fdm_delta_T"].mean(),
                "mae_K": err.abs().mean(),
                "rmse_K": np.sqrt(np.mean(err**2)),
                "bias_K": err.mean(),
                "relative_mae_pct": (err.abs() / verified["fdm_delta_T"]).mean() * 100,
                "signed_relative_bias_pct": rel.mean() * 100,
                "median_abs_relative_error_pct": rel.abs().median() * 100,
                "best_fdm_delta_T": verified.loc[best_idx, "fdm_delta_T"],
                "best_predicted_delta_T": verified.loc[best_idx, "predicted_delta_T"],
                "best_relative_error_pct": rel.loc[best_idx] * 100,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(SUMMARY_DIR / "real_world_relative_error_summary.csv", index=False)
    return df


def plot_relative_deviation(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    labels = df["scenario"].tolist()
    colors = [c for _, _, c in SCENARIOS]
    x = np.arange(len(df))

    ax = axes[0]
    ax.bar(x, df["mae_K"], color=colors, alpha=0.88)
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylabel("MAE against FDM (K)")
    ax.set_title("Absolute error")
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    for i, value in enumerate(df["mae_K"]):
        ax.text(i, value + max(df["mae_K"]) * 0.025, f"{value:.1f} K", ha="center", fontsize=9)

    ax = axes[1]
    ax.bar(x, df["relative_mae_pct"], color=colors, alpha=0.88)
    ax.axhline(20, color="#64748b", linestyle="--", linewidth=1)
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylabel("Mean |prediction - FDM| / FDM (%)")
    ax.set_title("Relative deviation")
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    for i, value in enumerate(df["relative_mae_pct"]):
        ax.text(i, value + max(df["relative_mae_pct"]) * 0.025, f"{value:.0f}%", ha="center", fontsize=9)

    fig.suptitle(
        "Relative error changes the interpretation of calibration",
        x=0.02,
        y=1.04,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.02,
        -0.02,
        "Engine has the largest Kelvin error, but its relative deviation is moderate because the true Delta T is large. "
        "Skin has small Kelvin error but high relative deviation because the true Delta T is small.",
        fontsize=10,
        color="#475569",
    )
    savefig(fig, "fig_benchmark_relative_deviation.png")


def plot_signed_relative_bias(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 4.7))
    y = np.arange(len(df))
    colors = ["#dc2626" if v < 0 else "#2563eb" for v in df["signed_relative_bias_pct"]]
    ax.barh(y, df["signed_relative_bias_pct"], color=colors, alpha=0.88)
    ax.axvline(0, color="#111827", linewidth=1)
    ax.set_yticks(y, df["scenario"])
    ax.set_xlabel("Mean signed relative bias: (prediction - FDM) / FDM (%)")
    ax.set_title("Signed relative bias separates overprediction from underprediction")
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    for i, value in enumerate(df["signed_relative_bias_pct"]):
        ha = "left" if value >= 0 else "right"
        offset = 1.5 if value >= 0 else -1.5
        ax.text(value + offset, i, f"{value:+.1f}%", va="center", ha=ha, fontsize=10)
    savefig(fig, "fig_benchmark_signed_relative_bias.png")


def draw_flow_box(ax, x, y, w, h, title, body, color):
    rect = plt.Rectangle((x, y), w, h, facecolor="white", edgecolor=color, linewidth=1.8)
    ax.add_patch(rect)
    ax.add_patch(plt.Rectangle((x, y + h - 0.16), w, 0.16, facecolor=color, edgecolor=color))
    ax.text(x + 0.035, y + h - 0.08, title, va="center", ha="left", fontsize=9.4, color="white", weight="bold")
    ax.text(x + 0.035, y + h - 0.19, body, va="top", ha="left", fontsize=8.2, color="#334155", linespacing=1.18)


def plot_workflow() -> None:
    fig, ax = plt.subplots(figsize=(13.6, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.01, 0.95, "Expanded surrogate-assisted inverse-design workflow", fontsize=16, weight="bold", color="#111827")
    ax.text(
        0.01,
        0.895,
        "The surrogate accelerates candidate discovery; final reported performance remains FDM-verified.",
        fontsize=10,
        color="#475569",
    )

    boxes = [
        (0.02, 0.52, 0.15, 0.30, "1. FDM data", "100k FDM samples\n3D material mask\nhot-boundary map", "#2563eb"),
        (0.20, 0.52, 0.15, 0.30, "2. Filter/split", "metadata QC\n80k train / 10k val\n10k test", "#0f766e"),
        (0.38, 0.52, 0.15, 0.30, "3. Surrogate", "3D CNN fusion\n2 field channels\n26 scalar descriptors", "#7c3aed"),
        (0.56, 0.52, 0.15, 0.30, "4. Evaluate", "MAE/RMSE/R2\nTop-10% recall\nrank correlation", "#ea580c"),
        (0.38, 0.14, 0.15, 0.30, "5. Screen", "50k candidates\nper scenario\nsurrogate ranking", "#0891b2"),
        (0.56, 0.14, 0.15, 0.30, "6. FDM verify", "top candidates only\n50 verified/scenario\nfinal Delta T claims", "#16a34a"),
        (0.75, 0.14, 0.20, 0.68, "Outputs", "Best verified structures\nRelative calibration error\nScenario-specific limits\nNext targeted data plan", "#111827"),
    ]
    for b in boxes:
        draw_flow_box(ax, *b)

    arrows = [
        ((0.17, 0.67), (0.20, 0.67)),
        ((0.35, 0.67), (0.38, 0.67)),
        ((0.53, 0.67), (0.56, 0.67)),
        ((0.455, 0.52), (0.455, 0.44)),
        ((0.53, 0.29), (0.56, 0.29)),
        ((0.71, 0.29), (0.75, 0.29)),
        ((0.71, 0.67), (0.75, 0.67)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.6, color="#64748b"))

    ax.text(0.395, 0.465, "trained model reused\nfor inverse design", fontsize=8.2, color="#475569")
    ax.text(0.025, 0.055, "Key guardrail: surrogate predictions are screening scores, not final physical claims.", fontsize=10.5, color="#dc2626", weight="bold")
    savefig(fig, "fig_workflow_overview_detailed.png")


def plot_3d_structures(df: pd.DataFrame) -> None:
    benchmark = pd.read_csv(BENCH_DIR / "benchmark_summary.csv").set_index("scenario_key")
    fig = plt.figure(figsize=(15.5, 5.9))
    for i, (key, label, color) in enumerate(SCENARIOS, start=1):
        npz = np.load(BENCH_DIR / key / "top_candidate_masks.npz")
        candidate_ids = npz["candidate_ids"].astype(str)
        best_id = benchmark.loc[key, "best_candidate_id"]
        idx = int(np.where(candidate_ids == best_id)[0][0])
        mask = npz["masks"][idx][::2, ::2, ::2]
        hot = npz["hot_boundary_maps"][idx][::2, ::2]
        row = df[df["scenario_key"] == key].iloc[0]

        ax = fig.add_subplot(1, 5, i, projection="3d")
        face = np.empty(mask.shape + (4,), dtype=float)
        rgba = matplotlib.colors.to_rgba(color, 0.58)
        face[mask] = rgba
        face[~mask] = (1, 1, 1, 0)
        ax.voxels(mask, facecolors=face, edgecolor=(1, 1, 1, 0.05), linewidth=0.1)

        nx, ny, nz = mask.shape
        xx, yy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        z = np.full_like(xx, nz + 0.5, dtype=float)
        norm = Normalize(vmin=float(hot.min()), vmax=float(hot.max()) if hot.max() > hot.min() else float(hot.max() + 1))
        ax.plot_surface(xx, yy, z, facecolors=cm.inferno(norm(hot)), shade=False, alpha=0.78, linewidth=0)

        ax.view_init(elev=25, azim=-55)
        ax.set_box_aspect((1, 1, 0.55))
        ax.set_axis_off()
        ax.set_title(
            f"{label}\n{benchmark.loc[key, 'best_geometry_type']} | FDM {row['best_fdm_delta_T']:.1f} K\nbest rel. err {row['best_relative_error_pct']:+.1f}%",
            fontsize=9.5,
            color="#111827",
            pad=1,
        )

    fig.suptitle(
        "Best FDM-verified 3D candidate structure for each real-world scenario",
        x=0.02,
        y=0.98,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.02,
        "Colored voxels represent the high-conductivity material mask. The top sheet shows the scenario hot-boundary map used by the model.",
        fontsize=10,
        color="#475569",
    )
    savefig(fig, "fig_scenario_best_3d_structures.png")


def plot_3d_structures_slide(df: pd.DataFrame) -> None:
    benchmark = pd.read_csv(BENCH_DIR / "benchmark_summary.csv").set_index("scenario_key")
    fig = plt.figure(figsize=(16.2, 4.2))
    for i, (key, label, color) in enumerate(SCENARIOS, start=1):
        npz = np.load(BENCH_DIR / key / "top_candidate_masks.npz")
        candidate_ids = npz["candidate_ids"].astype(str)
        best_id = benchmark.loc[key, "best_candidate_id"]
        idx = int(np.where(candidate_ids == best_id)[0][0])
        mask = npz["masks"][idx][::2, ::2, ::2]
        hot = npz["hot_boundary_maps"][idx][::2, ::2]
        row = df[df["scenario_key"] == key].iloc[0]

        ax = fig.add_subplot(1, 5, i, projection="3d")
        face = np.empty(mask.shape + (4,), dtype=float)
        rgba = matplotlib.colors.to_rgba(color, 0.62)
        face[mask] = rgba
        face[~mask] = (1, 1, 1, 0)
        ax.voxels(mask, facecolors=face, edgecolor=(1, 1, 1, 0.04), linewidth=0.08)

        nx, ny, nz = mask.shape
        xx, yy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        z = np.full_like(xx, nz + 0.5, dtype=float)
        norm = Normalize(vmin=float(hot.min()), vmax=float(hot.max()) if hot.max() > hot.min() else float(hot.max() + 1))
        ax.plot_surface(xx, yy, z, facecolors=cm.inferno(norm(hot)), shade=False, alpha=0.78, linewidth=0)

        ax.view_init(elev=25, azim=-55)
        ax.set_box_aspect((1, 1, 0.58))
        ax.set_axis_off()
        ax.set_title(
            f"{label} | FDM {row['best_fdm_delta_T']:.1f} K\n{benchmark.loc[key, 'best_geometry_type']} | rel. err {row['best_relative_error_pct']:+.1f}%",
            fontsize=12,
            color="#111827",
            pad=0,
        )

    fig.subplots_adjust(left=0.01, right=0.99, top=0.82, bottom=0.02, wspace=0.02)
    savefig(fig, "fig_scenario_best_3d_structures_slide.png")


def main() -> None:
    df = relative_metrics()
    plot_relative_deviation(df)
    plot_signed_relative_bias(df)
    plot_workflow()
    plot_3d_structures(df)
    plot_3d_structures_slide(df)
    print(json.dumps({"rows": df.to_dict(orient="records")}, indent=2))


if __name__ == "__main__":
    main()
