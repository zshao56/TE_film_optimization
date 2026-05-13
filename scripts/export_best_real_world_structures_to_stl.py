from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = (
    ROOT
    / "results"
    / "expanded_rebuild_v1_summary"
    / "full_plot_inputs"
    / "results"
    / "real_world_benchmarks"
    / "expanded_rebuild_v1"
)
OUT_DIR = ROOT / "results" / "expanded_rebuild_v1_summary" / "stl_structures"


SCENARIO_LABELS = {
    "battery_ac_half_cylinder": "Battery surface cooling",
    "skin_slight_curve_uniform": "Skin patch",
    "glass_center_hot_natural": "Glass panel",
    "engine_forced_flat": "Automotive engine surface",
    "phone_linear_natural": "Phone surface",
}


FACES = [
    ((-1, 0, 0), (-1.0, 0.0, 0.0), [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)]),
    ((1, 0, 0), (1.0, 0.0, 0.0), [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]),
    ((0, -1, 0), (0.0, -1.0, 0.0), [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)]),
    ((0, 1, 0), (0.0, 1.0, 0.0), [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)]),
    ((0, 0, -1), (0.0, 0.0, -1.0), [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)]),
    ((0, 0, 1), (0.0, 0.0, 1.0), [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]),
]


def is_occupied(mask: np.ndarray, i: int, j: int, k: int) -> bool:
    if i < 0 or j < 0 or k < 0:
        return False
    if i >= mask.shape[0] or j >= mask.shape[1] or k >= mask.shape[2]:
        return False
    return bool(mask[i, j, k])


def voxel_triangles(mask: np.ndarray, spacing_mm: tuple[float, float, float]):
    sx, sy, sz = spacing_mm
    occupied = np.argwhere(mask)
    for i, j, k in occupied:
        base = np.array([i * sx, j * sy, k * sz], dtype=np.float32)
        for neighbor_delta, normal, corners in FACES:
            di, dj, dk = neighbor_delta
            if is_occupied(mask, int(i + di), int(j + dj), int(k + dk)):
                continue
            points = []
            for cx, cy, cz in corners:
                points.append(base + np.array([cx * sx, cy * sy, cz * sz], dtype=np.float32))
            yield normal, points[0], points[1], points[2]
            yield normal, points[0], points[2], points[3]


def write_binary_stl(path: Path, triangles, name: str) -> int:
    triangles = list(triangles)
    header = f"{name} | units: millimeters | voxel-surface export".encode("ascii", errors="ignore")[:80]
    header = header + b" " * (80 - len(header))
    with path.open("wb") as f:
        f.write(header)
        f.write(struct.pack("<I", len(triangles)))
        for normal, v1, v2, v3 in triangles:
            data = (*normal, *map(float, v1), *map(float, v2), *map(float, v3))
            f.write(struct.pack("<12fH", *data, 0))
    return len(triangles)


def candidate_dimensions_mm(row: pd.Series) -> tuple[float, float, float]:
    params = json.loads(row["geometry_parameters"])
    lx_m = float(row.get("length_Lx", params.get("Lx", 0.01)))
    ly_m = float(row.get("length_Ly", params.get("Ly", 0.01)))
    h_m = float(row.get("thickness_h", params.get("h", 0.001)))
    return lx_m * 1000.0, ly_m * 1000.0, h_m * 1000.0


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    benchmark = pd.read_csv(BENCH_DIR / "benchmark_summary.csv")
    records = []

    for row in benchmark.itertuples(index=False):
        scenario_key = row.scenario_key
        best_id = row.best_candidate_id
        scenario_dir = BENCH_DIR / scenario_key
        top_candidates = pd.read_csv(scenario_dir / "top_candidates.csv")
        cand_row = top_candidates[top_candidates["candidate_id"] == best_id].iloc[0]

        masks_npz = np.load(scenario_dir / "top_candidate_masks.npz")
        candidate_ids = masks_npz["candidate_ids"].astype(str)
        mask_idx = int(np.where(candidate_ids == best_id)[0][0])
        mask = masks_npz["masks"][mask_idx].astype(bool)

        lx_mm, ly_mm, h_mm = candidate_dimensions_mm(cand_row)
        nx, ny, nz = mask.shape
        spacing_mm = (lx_mm / nx, ly_mm / ny, h_mm / nz)

        geometry_type = str(row.best_geometry_type)
        file_stem = f"{scenario_key}_best_fdm_{geometry_type}"
        stl_path = OUT_DIR / f"{file_stem}.stl"
        triangle_count = write_binary_stl(stl_path, voxel_triangles(mask, spacing_mm), file_stem)

        records.append(
            {
                "scenario_key": scenario_key,
                "scenario_label": SCENARIO_LABELS.get(scenario_key, scenario_key),
                "candidate_id": best_id,
                "geometry_type": geometry_type,
                "stl_file": stl_path.name,
                "units": "millimeters",
                "size_x_mm": lx_mm,
                "size_y_mm": ly_mm,
                "size_z_mm": h_mm,
                "voxel_shape": "x".join(map(str, mask.shape)),
                "voxel_spacing_x_mm": spacing_mm[0],
                "voxel_spacing_y_mm": spacing_mm[1],
                "voxel_spacing_z_mm": spacing_mm[2],
                "occupied_voxels": int(mask.sum()),
                "triangles": triangle_count,
                "best_fdm_delta_T": float(row.best_fdm_delta_T),
                "best_predicted_delta_T": float(row.best_predicted_delta_T),
                "best_surrogate_rank": int(row.best_surrogate_rank),
            }
        )
        print(f"Wrote {stl_path} ({triangle_count} triangles)")

    metadata = pd.DataFrame(records)
    metadata.to_csv(OUT_DIR / "metadata.csv", index=False)
    (OUT_DIR / "metadata.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (OUT_DIR / "README.md").write_text(
        "# Best real-world candidate STL exports\n\n"
        "These STL files export the high-conductivity material masks for the FDM-best verified "
        "candidate in each real-world benchmark scenario.\n\n"
        "- Units: millimeters. STL itself is unitless, but coordinates are scaled to mm.\n"
        "- Geometry source: `top_candidate_masks.npz` from `expanded_rebuild_v1`.\n"
        "- Export method: binary STL generated from exposed voxel faces; internal faces are omitted.\n"
        "- Note: curved scenario metadata is preserved in `metadata.csv`, but the STL is the voxel mask "
        "in the model grid coordinate system, not a post-warped curved substrate.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
