import argparse
import json
from pathlib import Path

import pandas as pd


def _pct(value, total):
    return 0.0 if total == 0 else 100.0 * value / total


def check_dataset(metadata_path, config_path=None):
    metadata_path = Path(metadata_path)
    df = pd.read_csv(metadata_path, low_memory=False)
    total = len(df)
    print(f"Metadata: {metadata_path}")
    print(f"Rows: {total}")

    if "qc_pass" in df.columns:
        qc_count = int((df["qc_pass"] == True).sum())
        print(f"QC pass: {qc_count} ({_pct(qc_count, total):.2f}%)")

    if "thickness_h" in df.columns:
        h = pd.to_numeric(df["thickness_h"], errors="coerce")
        outside = int(((h < 0.0004) | (h > 0.004)).sum())
        print(f"Thickness range: min={h.min():.6g}, max={h.max():.6g}, outside_v2={outside}")

    if "curvature_type" in df.columns:
        print("Curvature types:")
        print(df["curvature_type"].value_counts(dropna=False).to_string())

    if "delta_T_parallel" in df.columns:
        delta = pd.to_numeric(df["delta_T_parallel"], errors="coerce").dropna()
        bins = pd.cut(
            delta,
            bins=[-float("inf"), 10.0, 15.0, 50.0, 100.0, float("inf")],
            labels=["<=10K", "10-15K", "15-50K", "50-100K", ">100K"],
        )
        counts = bins.value_counts().sort_index()
        print("\nDelta_T_parallel bins:")
        for label, count in counts.items():
            print(f"  {label}: {int(count)} ({_pct(int(count), len(delta)):.2f}%)")
        print(
            f"\nDelta_T stats: min={delta.min():.4f}, mean={delta.mean():.4f}, "
            f"p90={delta.quantile(0.90):.4f}, p95={delta.quantile(0.95):.4f}, max={delta.max():.4f}"
        )

    if "geometry_type" in df.columns:
        print("\nGeometry distribution:")
        print(df["geometry_type"].value_counts(dropna=False).to_string())

    if config_path is not None:
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        scenarios = config.get("real_world_benchmark", {}).get("scenarios", [])
        engine = [scenario for scenario in scenarios if scenario.get("key") == "engine_forced_flat"]
        if engine:
            hot = engine[0]["hot_boundary"]
            print(
                "\nEngine scenario from config: "
                f"center={hot.get('center_temp')} K, edge={hot.get('edge_temp')} K, "
                f"curvature={engine[0].get('curvature_level')}"
            )


def build_parser():
    parser = argparse.ArgumentParser(description="Check whether a generated dataset matches the v2 sampling intent.")
    parser.add_argument("--metadata", default="data/simulations/metadata_clean.csv")
    parser.add_argument("--config", default="configs/v2_flat_unified_thickness.json")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    check_dataset(args.metadata, args.config)
