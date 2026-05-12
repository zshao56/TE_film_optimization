import argparse
import json
import os
import sys
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch


current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)
if current_dir not in sys.path:
    sys.path.append(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from dataset import TEFilmDataset
from generate_database import (
    HOT_BOUNDARY_TYPE_CODES,
    _sample_geometry,
    _sample_materials,
    _sample_thickness,
)
from inverse_design import (
    _candidate_record,
    _convection_regime_for_h,
    _device_from_arg,
    _load_checkpoint,
    _make_inputs,
    _run_verification_task,
    _save_verification_rows,
    _update_top_heap,
    VERIFY_COLUMNS,
)
from models import ThermoNetFusion


def _normalized_xy(nx, ny):
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    return np.meshgrid(x, y, indexing="ij")


def _center_hot_map(nx, ny, edge_temp, center_temp, sigma=0.23):
    x, y = _normalized_xy(nx, ny)
    r2 = (x - 0.5) ** 2 + (y - 0.5) ** 2
    raw = np.exp(-0.5 * r2 / (sigma ** 2))
    raw_min = float(raw.min())
    raw_max = float(raw.max())
    normalized = (raw - raw_min) / max(raw_max - raw_min, 1e-12)
    return edge_temp + (center_temp - edge_temp) * normalized


def _linear_hot_map(nx, ny, cold_temp, hot_temp, direction="x"):
    x, y = _normalized_xy(nx, ny)
    coord = x if direction == "x" else y
    return cold_temp + (hot_temp - cold_temp) * coord


def _uniform_hot_map(nx, ny, temperature):
    return np.full((nx, ny), temperature, dtype=float)


def _hot_map_from_config(nx, ny, hot_config):
    kind = hot_config.get("kind", hot_config.get("type", "uniform"))
    if kind == "uniform":
        return _uniform_hot_map(nx, ny, float(hot_config["temperature"]))
    if kind in {"center_hotspot", "gaussian_hotspot"}:
        return _center_hot_map(
            nx,
            ny,
            edge_temp=float(hot_config["edge_temp"]),
            center_temp=float(hot_config["center_temp"]),
            sigma=float(hot_config.get("sigma", 0.23)),
        )
    if kind == "linear_gradient":
        return _linear_hot_map(
            nx,
            ny,
            cold_temp=float(hot_config["cold_temp"]),
            hot_temp=float(hot_config["hot_temp"]),
            direction=hot_config.get("direction", "x"),
        )
    raise ValueError(f"Unsupported real-world hot boundary kind: {kind}")


def _boundary_type_from_hot_config(hot_config):
    kind = hot_config.get("kind", hot_config.get("type", "uniform"))
    if kind == "center_hotspot":
        return "gaussian_hotspot"
    if kind in HOT_BOUNDARY_TYPE_CODES:
        return kind
    if kind == "linear_gradient":
        return "linear_gradient"
    raise ValueError(f"Unsupported real-world hot boundary kind: {kind}")


def _hot_boundary_metadata(hot_map, boundary_type, gradient_direction_code=0):
    return {
        "T_hot": float(np.mean(hot_map)),
        "T_hot_map": hot_map.astype(float),
        "hot_boundary_type": boundary_type,
        "hot_boundary_type_code": HOT_BOUNDARY_TYPE_CODES[boundary_type],
        "T_hot_min": float(np.min(hot_map)),
        "T_hot_max": float(np.max(hot_map)),
        "T_hot_amplitude": float(np.max(hot_map) - np.min(hot_map)),
        "gradient_direction_code": gradient_direction_code,
        "hotspot_x": 0.5 if boundary_type == "gaussian_hotspot" else 0.0,
        "hotspot_y": 0.5 if boundary_type == "gaussian_hotspot" else 0.0,
        "hotspot_sigma": 0.23 if boundary_type == "gaussian_hotspot" else 0.0,
    }


def _curvature_metadata(level, Lx, Ly, bend_axis="x"):
    if level <= 1e-12:
        return {
            "curvature_type": "flat",
            "curvature_level": 0.0,
            "arc_angle": 0.0,
            "bend_axis": "none",
            "bend_axis_code": 0,
            "bend_radius": 0.0,
            "arc_length": Lx,
            "projected_length": Lx,
            "projected_Lx": Lx,
            "projected_Ly": Ly,
        }

    bend_axis_code = 0 if bend_axis == "x" else 1
    arc_angle = float(level * np.pi)
    arc_length = float(Lx if bend_axis == "x" else Ly)
    radius = float(arc_length / arc_angle)
    projected_length = float(2.0 * radius * np.sin(arc_angle / 2.0))
    return {
        "curvature_type": "cylindrical_arc",
        "curvature_level": float(level),
        "arc_angle": arc_angle,
        "bend_axis": bend_axis,
        "bend_axis_code": bend_axis_code,
        "bend_radius": radius,
        "arc_length": arc_length,
        "projected_length": projected_length,
        "projected_Lx": projected_length if bend_axis == "x" else Lx,
        "projected_Ly": projected_length if bend_axis == "y" else Ly,
    }


def _scenario_from_config(config, nx, ny):
    hot_config = config["hot_boundary"]
    return {
        "key": config["key"],
        "label": config.get("label", config["key"]),
        "description": config.get("description", ""),
        "T_air": float(config.get("T_air", 298.15)),
        "h_c": float(config["h_c"]),
        "h_c_side": float(config.get("h_c_side", config["h_c"])),
        "hot_map": _hot_map_from_config(nx, ny, hot_config),
        "hot_boundary_type": _boundary_type_from_hot_config(hot_config),
        "curvature_level": float(config.get("curvature_level", 0.0)),
        "bend_axis": config.get("bend_axis", "x"),
        "gradient_direction_code": 0 if hot_config.get("direction", "x") == "x" else 1,
    }


def _scenario_definitions(nx, ny, scenario_configs=None):
    if scenario_configs is not None:
        return [_scenario_from_config(config, nx, ny) for config in scenario_configs]

    return [
        {
            "key": "battery_ac_half_cylinder",
            "label": "Battery surface cooling",
            "description": "Battery shell, center hotspot 390 K, edge 360 K, half-cylinder curvature, strong air-conditioner convection.",
            "T_air": 298.15,
            "h_c": 180.0,
            "h_c_side": 180.0,
            "hot_map": _center_hot_map(nx, ny, edge_temp=360.0, center_temp=390.0, sigma=0.24),
            "hot_boundary_type": "gaussian_hotspot",
            "curvature_level": 1.0,
            "bend_axis": "x",
        },
        {
            "key": "skin_slight_curve_uniform",
            "label": "Skin patch",
            "description": "Skin-like patch, uniform 37 C body temperature, slight curvature, natural convection.",
            "T_air": 298.15,
            "h_c": 8.0,
            "h_c_side": 8.0,
            "hot_map": _uniform_hot_map(nx, ny, temperature=310.15),
            "hot_boundary_type": "uniform",
            "curvature_level": 0.10,
            "bend_axis": "x",
        },
        {
            "key": "glass_center_hot_natural",
            "label": "Glass panel",
            "description": "Glass-like panel, center 70 C, edge 50 C, flat geometry, natural convection.",
            "T_air": 298.15,
            "h_c": 8.0,
            "h_c_side": 8.0,
            "hot_map": _center_hot_map(nx, ny, edge_temp=323.15, center_temp=343.15, sigma=0.25),
            "hot_boundary_type": "gaussian_hotspot",
            "curvature_level": 0.0,
            "bend_axis": "x",
        },
        {
            "key": "engine_forced_flat",
            "label": "Automotive engine surface",
            "description": "Engine-like hot surface, flat geometry, very high center temperature, strong forced convection from driving airflow.",
            "T_air": 303.15,
            "h_c": 300.0,
            "h_c_side": 300.0,
            "hot_map": _center_hot_map(nx, ny, edge_temp=420.0, center_temp=520.0, sigma=0.22),
            "hot_boundary_type": "gaussian_hotspot",
            "curvature_level": 0.0,
            "bend_axis": "x",
        },
        {
            "key": "phone_linear_natural",
            "label": "Phone surface",
            "description": "Phone back cover, one hot end and one cooler end, flat geometry, natural convection.",
            "T_air": 298.15,
            "h_c": 8.0,
            "h_c_side": 8.0,
            "hot_map": _linear_hot_map(nx, ny, cold_temp=303.15, hot_temp=333.15, direction="x"),
            "hot_boundary_type": "linear_gradient",
            "curvature_level": 0.0,
            "bend_axis": "x",
            "gradient_direction_code": 0,
        },
    ]


def _scenario_environment(scenario, nx, ny):
    hot_map = np.asarray(scenario["hot_map"], dtype=float)
    env = {
        "T_air": float(scenario["T_air"]),
        "h_c": float(scenario["h_c"]),
        "h_c_side": float(scenario["h_c_side"]),
    }
    regime, code = _convection_regime_for_h(env["h_c"])
    env["convection_regime"] = regime
    env["convection_regime_code"] = code
    env.update(
        _hot_boundary_metadata(
            hot_map,
            scenario["hot_boundary_type"],
            gradient_direction_code=int(scenario.get("gradient_direction_code", 0)),
        )
    )
    return env


def _sample_benchmark_candidate(args, scenario, seed, candidate_index):
    rng = np.random.default_rng(seed)
    h = _sample_thickness(rng, "expanded")
    k_low, k_high = _sample_materials(rng, "expanded")
    env_params = _scenario_environment(scenario, args.nx, args.ny)
    geom = _sample_geometry(
        args.Lx,
        args.Ly,
        h,
        k_low,
        k_high,
        args.nx,
        args.ny,
        args.nz,
        env_params,
        rng,
        args.mode,
        args.structured_ratio,
    )
    geom.update(_curvature_metadata(scenario["curvature_level"], args.Lx, args.Ly, scenario["bend_axis"]))
    geom["database_profile"] = "real_world_benchmark"
    geom["scenario_id"] = scenario["key"]
    geom["scenario_label"] = scenario["label"]
    geom["k_ratio"] = float(k_high / (k_low + 1e-15))
    geom["sample_seed"] = int(seed)
    return f"{scenario['key']}_cand_{candidate_index:08d}", geom


def _save_screen_result(output_dir, scenario, all_records, top_items, checkpoint_meta, args):
    os.makedirs(output_dir, exist_ok=True)

    all_df = pd.DataFrame(all_records)
    all_path = os.path.join(output_dir, "screened_candidates.csv")
    all_df.to_csv(all_path, index=False)

    top_items = sorted(top_items, key=lambda item: item[0], reverse=True)
    top_records = []
    top_masks = []
    top_hot_maps = []
    for rank, (_score, _candidate_id, record, mask, hot_map) in enumerate(top_items, start=1):
        enriched = dict(record)
        enriched["surrogate_rank"] = rank
        top_records.append(enriched)
        top_masks.append(mask)
        top_hot_maps.append(hot_map)

    top_df = pd.DataFrame(top_records)
    top_path = os.path.join(output_dir, "top_candidates.csv")
    top_df.to_csv(top_path, index=False)

    masks_path = os.path.join(output_dir, "top_candidate_masks.npz")
    np.savez_compressed(
        masks_path,
        candidate_ids=top_df["candidate_id"].to_numpy(dtype=str),
        masks=np.stack(top_masks, axis=0).astype(bool),
        hot_boundary_maps=np.stack(top_hot_maps, axis=0).astype(np.float32),
    )

    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scenario": {key: value for key, value in scenario.items() if key != "hot_map"},
        "hot_boundary_summary": {
            "T_hot_min": float(np.min(scenario["hot_map"])),
            "T_hot_max": float(np.max(scenario["hot_map"])),
            "T_hot_mean": float(np.mean(scenario["hot_map"])),
        },
        "model_path": args.model_path,
        "num_candidates": args.num_candidates,
        "top_k": args.top_k,
        "verify_count": args.verify_count,
        "mode": args.mode,
        "structured_ratio": args.structured_ratio,
        "seed": args.seed,
        "checkpoint_meta": {
            key: value
            for key, value in checkpoint_meta.items()
            if key not in {"state_dict", "model_state_dict"}
        },
    }
    with open(os.path.join(output_dir, "scenario_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    return top_df, masks_path


def _screen_scenario(args, scenario, model_context, scenario_index, output_dir):
    model, checkpoint_meta, scalar_cols, scalar_mean, scalar_std, include_boundary_channel, hot_mean, hot_std, device = model_context
    seed_sequence = np.random.SeedSequence([args.seed, scenario_index])
    child_seeds = seed_sequence.spawn(args.num_candidates)
    all_records = []
    top_heap = []

    with torch.no_grad():
        for batch_start in range(0, args.num_candidates, args.batch_size):
            batch_end = min(batch_start + args.batch_size, args.num_candidates)
            geoms = []
            records = []
            for idx in range(batch_start, batch_end):
                seed = int(child_seeds[idx].generate_state(1)[0])
                candidate_id, geom = _sample_benchmark_candidate(args, scenario, seed, idx)
                records.append(_candidate_record(candidate_id, geom))
                geoms.append(geom)

            masks, scalars = _make_inputs(
                geoms,
                scalar_cols,
                scalar_mean,
                scalar_std,
                include_boundary_channel,
                hot_mean,
                hot_std,
                device,
            )
            outputs = model(masks, scalars).detach().cpu().numpy().reshape(-1)
            if checkpoint_meta.get("normalize_target", False):
                outputs = outputs * float(checkpoint_meta["target_std"]) + float(checkpoint_meta["target_mean"])

            for record, geom, pred in zip(records, geoms, outputs):
                record["predicted_delta_T"] = float(pred)
                all_records.append(record)
                _update_top_heap(
                    top_heap,
                    args.top_k,
                    pred,
                    record,
                    geom["mask_3d"],
                    geom["T_hot_map"],
                )

            if batch_end % max(args.batch_size * 10, 1) == 0 or batch_end == args.num_candidates:
                print(f"[{scenario['key']}] Screened {batch_end}/{args.num_candidates} candidates")

    return _save_screen_result(output_dir, scenario, all_records, top_heap, checkpoint_meta, args)


def _verify_scenario(args, scenario_dir):
    top_path = os.path.join(scenario_dir, "top_candidates.csv")
    mask_path = os.path.join(scenario_dir, "top_candidate_masks.npz")
    verified_path = os.path.join(scenario_dir, "verified_candidates.csv")

    top_df = pd.read_csv(top_path).head(args.verify_count).copy()
    mask_data = np.load(mask_path, allow_pickle=False)
    ids = mask_data["candidate_ids"].astype(str)
    masks = {candidate_id: mask_data["masks"][idx].astype(bool) for idx, candidate_id in enumerate(ids)}
    hot_maps = {candidate_id: mask_data["hot_boundary_maps"][idx].astype(np.float32) for idx, candidate_id in enumerate(ids)}

    tasks = []
    for _, row in top_df.iterrows():
        row_dict = row.to_dict()
        candidate_id = row_dict["candidate_id"]
        tasks.append((row_dict, masks[candidate_id], hot_maps[candidate_id]))

    rows = []
    print(f"Verifying {len(tasks)} candidates in {scenario_dir} with {args.verify_workers} worker(s)")
    if args.verify_workers <= 1:
        for idx, (row_dict, mask, hot_map) in enumerate(tasks, start=1):
            rows.append(_run_verification_task(row_dict, mask, hot_map))
            if idx % args.save_every == 0 or idx == len(tasks):
                _save_verification_rows(pd.DataFrame(columns=VERIFY_COLUMNS), rows, verified_path)
                print(f"Verified {idx}/{len(tasks)}")
    else:
        with ProcessPoolExecutor(max_workers=args.verify_workers) as executor:
            futures = [executor.submit(_run_verification_task, row, mask, hot) for row, mask, hot in tasks]
            for completed, future in enumerate(as_completed(futures), start=1):
                rows.append(future.result())
                if completed % args.save_every == 0 or completed == len(tasks):
                    _save_verification_rows(pd.DataFrame(columns=VERIFY_COLUMNS), rows, verified_path)
                    print(f"Verified {completed}/{len(tasks)}")

    verified_df = _save_verification_rows(pd.DataFrame(columns=VERIFY_COLUMNS), rows, verified_path)
    return verified_df


def _summarize_verified(scenario, verified_df):
    clean = verified_df.dropna(subset=["fdm_delta_T"]).copy()
    if clean.empty:
        return {
            "scenario_key": scenario["key"],
            "scenario_label": scenario["label"],
            "verified": 0,
        }
    err = clean["predicted_delta_T"] - clean["fdm_delta_T"]
    best = clean.sort_values("fdm_delta_T", ascending=False).iloc[0]
    return {
        "scenario_key": scenario["key"],
        "scenario_label": scenario["label"],
        "verified": int(len(clean)),
        "best_fdm_delta_T": float(clean["fdm_delta_T"].max()),
        "mean_fdm_delta_T": float(clean["fdm_delta_T"].mean()),
        "median_fdm_delta_T": float(clean["fdm_delta_T"].median()),
        "mae": float(err.abs().mean()),
        "bias": float(err.mean()),
        "rmse": float((err.pow(2).mean()) ** 0.5),
        "best_candidate_id": best["candidate_id"],
        "best_surrogate_rank": int(best["surrogate_rank"]),
        "best_geometry_type": best["geometry_type"],
        "best_predicted_delta_T": float(best["predicted_delta_T"]),
    }


def _write_scenario_table(scenarios, output_dir):
    rows = []
    for scenario in scenarios:
        regime, code = _convection_regime_for_h(float(scenario["h_c"]))
        rows.append(
            {
                "scenario_key": scenario["key"],
                "label": scenario["label"],
                "description": scenario["description"],
                "T_hot_min_K": float(np.min(scenario["hot_map"])),
                "T_hot_max_K": float(np.max(scenario["hot_map"])),
                "T_hot_mean_K": float(np.mean(scenario["hot_map"])),
                "T_air_K": float(scenario["T_air"]),
                "h_c": float(scenario["h_c"]),
                "h_c_side": float(scenario["h_c_side"]),
                "convection_regime": regime,
                "curvature_level": float(scenario["curvature_level"]),
                "hot_boundary_type": scenario["hot_boundary_type"],
            }
        )
    path = os.path.join(output_dir, "scenario_definitions.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def run_benchmark(args):
    if args.verify_workers < 1:
        raise ValueError("--verify-workers must be >= 1")
    if args.save_every < 1:
        raise ValueError("--save-every must be >= 1")

    device = _device_from_arg(args.device)
    print(f"Using device: {device}")

    state_dict, checkpoint_meta = _load_checkpoint(args.model_path, device)
    scalar_cols = checkpoint_meta.get("scalar_cols", ["thickness_h", "k_low", "k_high", "T_hot", "T_air"])
    input_channels = int(checkpoint_meta.get("input_channels", 1))
    include_boundary_channel = bool(checkpoint_meta.get("include_boundary_channel", input_channels > 1))

    metadata_csv = args.metadata_csv or os.path.join(project_root, "data", "simulations", "metadata.csv")
    root_dir = args.root_dir or os.path.join(project_root, "data", "simulations")
    dataset = TEFilmDataset(
        metadata_csv=metadata_csv,
        root_dir=root_dir,
        include_boundary_channel=include_boundary_channel,
        scalar_cols=scalar_cols,
        check_field_files=False,
    )

    model = ThermoNetFusion(scalar_dim=len(scalar_cols), input_channels=input_channels).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model scalar_dim={len(scalar_cols)}, input_channels={input_channels}")

    output_dir = args.output_dir or os.path.join(
        project_root,
        "results",
        "real_world_benchmarks",
        f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(output_dir, exist_ok=True)

    scenarios = _scenario_definitions(args.nx, args.ny, getattr(args, "scenarios", None))
    scenario_path = _write_scenario_table(scenarios, output_dir)
    print(f"Saved scenario definitions to {scenario_path}")

    model_context = (
        model,
        checkpoint_meta,
        scalar_cols,
        dataset.scalar_mean.astype(np.float32),
        dataset.scalar_std.astype(np.float32),
        include_boundary_channel,
        float(dataset.hot_boundary_mean),
        float(dataset.hot_boundary_std),
        device,
    )

    summary_rows = []
    for scenario_idx, scenario in enumerate(scenarios):
        print(f"\n=== Scenario: {scenario['label']} ({scenario['key']}) ===")
        scenario_dir = os.path.join(output_dir, scenario["key"])
        _screen_scenario(args, scenario, model_context, scenario_idx, scenario_dir)
        if args.skip_verify:
            continue
        verified_df = _verify_scenario(args, scenario_dir)
        summary = _summarize_verified(scenario, verified_df)
        summary_rows.append(summary)
        print(json.dumps(summary, indent=2))

    if summary_rows:
        summary_path = os.path.join(output_dir, "benchmark_summary.csv")
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        print(f"\nSaved benchmark summary to {summary_path}")
        print(pd.DataFrame(summary_rows).to_string(index=False))


def build_parser():
    parser = argparse.ArgumentParser(description="Run fair real-world scenario benchmarks with surrogate screening plus FDM verification.")
    parser.add_argument("--config", type=str, default=None, help="Optional JSON pipeline/config file. Uses the real_world_benchmark section when present.")
    parser.add_argument("--model-path", type=str, default=os.path.join(project_root, "results", "models", "best_thermonet.pth"))
    parser.add_argument("--metadata-csv", type=str, default=None)
    parser.add_argument("--root-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--num-candidates", type=int, default=50000, help="Candidates screened per scenario.")
    parser.add_argument("--top-k", type=int, default=500, help="Top surrogate candidates saved per scenario.")
    parser.add_argument("--verify-count", type=int, default=50, help="Top candidates verified by FDM per scenario.")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--verify-workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260512)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--mode", choices=["structured", "mixed", "random"], default="mixed")
    parser.add_argument("--structured-ratio", type=float, default=0.9)
    parser.add_argument("--Lx", type=float, default=0.01)
    parser.add_argument("--Ly", type=float, default=0.01)
    parser.add_argument("--nx", type=int, default=50)
    parser.add_argument("--ny", type=int, default=50)
    parser.add_argument("--nz", type=int, default=20)
    parser.add_argument("--skip-verify", action="store_true", help="Only run surrogate screening.")
    return parser


def _apply_config(args):
    if args.config is None:
        return args
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
    benchmark_config = config.get("real_world_benchmark", config)
    for key, value in benchmark_config.items():
        if key == "scenarios":
            setattr(args, "scenarios", value)
        elif hasattr(args, key):
            setattr(args, key, value)
    return args


if __name__ == "__main__":
    run_benchmark(_apply_config(build_parser().parse_args()))
