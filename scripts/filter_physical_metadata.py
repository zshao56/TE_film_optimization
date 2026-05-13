import argparse
import os

import pandas as pd


def _ensure_parent(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _numeric(df, column):
    if column not in df.columns:
        return pd.Series(float("nan"), index=df.index)
    return pd.to_numeric(df[column], errors="coerce")


def filter_metadata(args):
    metadata_path = os.path.abspath(args.metadata)
    output_path = os.path.abspath(args.output)
    bad_output_path = os.path.abspath(args.bad_output)
    report_path = os.path.abspath(args.report)

    df = pd.read_csv(metadata_path, low_memory=False)

    T_air = _numeric(df, "T_air")
    T_hot = _numeric(df, "T_hot")
    T_hot_min = _numeric(df, "T_hot_min").fillna(T_hot)
    T_hot_max = _numeric(df, "T_hot_max").fillna(T_hot)
    delta = _numeric(df, "delta_T_parallel")
    hot_electrode = _numeric(df, "T_hot_electrode_avg")
    cold_electrode = _numeric(df, "T_cold_electrode_avg")
    solver_residual = _numeric(df, "solver_relative_residual")

    lower_bound = pd.concat([T_air, T_hot_min], axis=1).min(axis=1)
    upper_bound = pd.concat([T_air, T_hot_max], axis=1).max(axis=1)
    max_allowed_delta = upper_bound - lower_bound

    bad_delta = delta > max_allowed_delta + args.tolerance
    bad_hot = hot_electrode > upper_bound + args.tolerance
    bad_cold = cold_electrode < lower_bound - args.tolerance

    bad_solver = pd.Series(False, index=df.index)
    for col in ["solver_bounds_pass", "surface_bounds_pass"]:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            bad_solver |= values.notna() & (values < 1)
    bad_residual = solver_residual.notna() & (solver_residual > args.max_relative_residual)

    bad_any = bad_delta | bad_hot | bad_cold | bad_solver | bad_residual
    if args.require_qc_pass and "qc_pass" in df.columns:
        qc_pass = df["qc_pass"].astype(str).str.lower().isin(["true", "1", "yes"])
        bad_any |= ~qc_pass

    annotated = df.copy()
    annotated["physical_lower_bound_K"] = lower_bound
    annotated["physical_upper_bound_K"] = upper_bound
    annotated["physical_max_allowed_delta_T_K"] = max_allowed_delta
    annotated["bad_delta_bound"] = bad_delta
    annotated["bad_hot_electrode_bound"] = bad_hot
    annotated["bad_cold_electrode_bound"] = bad_cold
    annotated["bad_solver_bounds"] = bad_solver
    annotated["bad_solver_residual"] = bad_residual
    annotated["bad_any"] = bad_any

    clean_df = df.loc[~bad_any].copy()
    bad_df = annotated.loc[bad_any].copy()

    _ensure_parent(output_path)
    _ensure_parent(bad_output_path)
    _ensure_parent(report_path)
    clean_df.to_csv(output_path, index=False)
    bad_df.to_csv(bad_output_path, index=False)

    summary = (
        annotated.assign(bad_any=bad_any)
        .groupby("scenario_id", dropna=False)["bad_any"]
        .agg(["count", "sum", "mean"])
        .sort_values("mean", ascending=False)
        .reset_index()
    )
    summary.to_csv(report_path, index=False)

    print(f"Input rows: {len(df)}")
    print(f"Clean rows: {len(clean_df)}")
    print(f"Bad rows: {int(bad_any.sum())}")
    print(f"Bad ratio: {float(bad_any.mean()):.6f}")
    print(f"bad_delta_bound: {int(bad_delta.sum())}")
    print(f"bad_hot_electrode_bound: {int(bad_hot.sum())}")
    print(f"bad_cold_electrode_bound: {int(bad_cold.sum())}")
    print(f"bad_solver_bounds: {int(bad_solver.sum())}")
    print(f"bad_solver_residual: {int(bad_residual.sum())}")
    print("\nWorst scenarios:")
    print(summary.head(args.head).to_string(index=False))
    print(f"\nWrote clean metadata to: {output_path}")
    print(f"Wrote bad rows to: {bad_output_path}")
    print(f"Wrote scenario report to: {report_path}")


def build_parser():
    parser = argparse.ArgumentParser(description="Filter metadata rows that violate thermal maximum-principle bounds.")
    parser.add_argument("--metadata", default=os.path.join("data", "simulations", "metadata.csv"))
    parser.add_argument("--output", default=os.path.join("data", "simulations", "metadata_clean.csv"))
    parser.add_argument("--bad-output", default=os.path.join("results", "metadata", "bad_physical_rows.csv"))
    parser.add_argument("--report", default=os.path.join("results", "metadata", "physical_sanity_by_scenario.csv"))
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--max-relative-residual", type=float, default=1e-8)
    parser.add_argument("--head", type=int, default=20)
    parser.add_argument("--require-qc-pass", action="store_true", help="Also drop rows where qc_pass is not true.")
    return parser


if __name__ == "__main__":
    filter_metadata(build_parser().parse_args())
