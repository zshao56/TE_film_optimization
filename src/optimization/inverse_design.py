import argparse
import heapq
import json
import os
import math
import re
import sys
import tempfile
import uuid
from datetime import datetime

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
from generate_database import _sample_environment, _sample_geometry
from main import run_simulation_pipeline
from models import ThermoNetFusion


DEFAULT_MODEL_PATH = os.path.join(
    project_root,
    "results",
    "experiments",
    "thermonet_auto_adaptive_under_0p2_bs128",
    "best_thermonet.pth",
)


def _device_from_arg(device_arg):
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_checkpoint(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"], checkpoint
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"], checkpoint
    return checkpoint, {"normalize_target": False}


def _json_safe_geometry(geom):
    safe = {}
    for key, value in geom.items():
        if key == "mask_3d":
            continue
        if isinstance(value, (np.integer, np.floating, np.bool_)):
            safe[key] = value.item()
        else:
            safe[key] = value
    return safe


def _candidate_record(candidate_id, geom):
    safe_geom = _json_safe_geometry(geom)
    return {
        "candidate_id": candidate_id,
        "geometry_type": geom["geometry_type"],
        "thickness_h": geom["h"],
        "length_Lx": geom["Lx"],
        "length_Ly": geom["Ly"],
        "k_low": geom["k_low"],
        "k_high": geom["k_high"],
        "T_hot": geom.get("T_hot"),
        "T_air": geom.get("T_air"),
        "h_c": geom.get("h_c"),
        "h_c_side": geom.get("h_c_side"),
        "volume_fraction_actual": geom.get("volume_fraction_actual"),
        "sample_seed": geom.get("sample_seed"),
        "geometry_parameters": json.dumps(safe_geom, sort_keys=True),
    }


def _sample_candidate(args, seed, candidate_index):
    rng = np.random.default_rng(seed)
    Lx, Ly = args.Lx, args.Ly
    nx, ny, nz = args.nx, args.ny, args.nz

    h = args.fixed_h if args.fixed_h is not None else float(rng.uniform(args.h_min, args.h_max))
    k_low = args.fixed_k_low if args.fixed_k_low is not None else float(rng.uniform(args.k_low_min, args.k_low_max))
    k_high = args.fixed_k_high if args.fixed_k_high is not None else float(rng.uniform(args.k_high_min, args.k_high_max))
    env_params = _sample_environment(rng)
    if args.fixed_T_hot is not None:
        env_params["T_hot"] = float(args.fixed_T_hot)
    if args.fixed_T_air is not None:
        env_params["T_air"] = float(args.fixed_T_air)
    if args.fixed_h_c is not None:
        env_params["h_c"] = float(args.fixed_h_c)
    if args.fixed_h_c_side is not None:
        env_params["h_c_side"] = float(args.fixed_h_c_side)
    geom = _sample_geometry(
        Lx,
        Ly,
        h,
        k_low,
        k_high,
        nx,
        ny,
        nz,
        env_params,
        rng,
        args.mode,
        args.structured_ratio,
    )
    geom["sample_seed"] = int(seed)
    return f"cand_{candidate_index:08d}", geom


def _make_inputs(geoms, scalar_mean, scalar_std, device):
    masks = []
    scalars = []
    for geom in geoms:
        mask = np.asarray(geom["mask_3d"], dtype=np.float32)
        scalar = np.array(
            [geom["h"], geom["k_low"], geom["k_high"], geom["T_hot"], geom["T_air"]],
            dtype=np.float32,
        )
        scalars.append((scalar - scalar_mean) / scalar_std)
        masks.append(mask)

    mask_tensor = torch.from_numpy(np.stack(masks, axis=0)).unsqueeze(1).to(device)
    scalar_tensor = torch.from_numpy(np.stack(scalars, axis=0).astype(np.float32)).to(device)
    return mask_tensor, scalar_tensor


def _update_top_heap(heap, top_k, score, record, mask):
    item = (float(score), record["candidate_id"], record, np.asarray(mask, dtype=bool))
    if len(heap) < top_k:
        heapq.heappush(heap, item)
    elif score > heap[0][0]:
        heapq.heapreplace(heap, item)


def _save_screen_outputs(args, output_dir, all_records, top_items, checkpoint_meta):
    os.makedirs(output_dir, exist_ok=True)

    all_df = pd.DataFrame(all_records)
    all_path = os.path.join(output_dir, "screened_candidates.csv")
    all_df.to_csv(all_path, index=False)

    top_items = sorted(top_items, key=lambda item: item[0], reverse=True)
    top_records = []
    top_masks = []
    for rank, (_score, _candidate_id, record, mask) in enumerate(top_items, start=1):
        enriched = dict(record)
        enriched["surrogate_rank"] = rank
        top_records.append(enriched)
        top_masks.append(mask)

    top_df = pd.DataFrame(top_records)
    top_path = os.path.join(output_dir, "top_candidates.csv")
    top_df.to_csv(top_path, index=False)

    masks_path = os.path.join(output_dir, "top_candidate_masks.npz")
    np.savez_compressed(
        masks_path,
        candidate_ids=top_df["candidate_id"].to_numpy(dtype=str),
        masks=np.stack(top_masks, axis=0).astype(bool),
    )

    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_path": args.model_path,
        "num_candidates": args.num_candidates,
        "top_k": args.top_k,
        "mode": args.mode,
        "structured_ratio": args.structured_ratio,
        "fixed_values": {
            "h": args.fixed_h,
            "k_low": args.fixed_k_low,
            "k_high": args.fixed_k_high,
            "T_hot": args.fixed_T_hot,
            "T_air": args.fixed_T_air,
            "h_c": args.fixed_h_c,
            "h_c_side": args.fixed_h_c_side,
        },
        "seed": args.seed,
        "checkpoint_meta": {
            key: value
            for key, value in checkpoint_meta.items()
            if key not in {"state_dict", "model_state_dict"}
        },
    }
    config_path = os.path.join(output_dir, "screen_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    return all_path, top_path, masks_path, config_path


def screen_candidates(args):
    device = _device_from_arg(args.device)
    print(f"Using device: {device}")

    metadata_csv = args.metadata_csv or os.path.join(project_root, "data", "simulations", "metadata.csv")
    root_dir = args.root_dir or os.path.join(project_root, "data", "simulations")
    dataset = TEFilmDataset(metadata_csv=metadata_csv, root_dir=root_dir)
    scalar_mean = dataset.scalar_mean.astype(np.float32)
    scalar_std = dataset.scalar_std.astype(np.float32)

    model = ThermoNetFusion(scalar_dim=5).to(device)
    state_dict, checkpoint_meta = _load_checkpoint(args.model_path, device)
    model.load_state_dict(state_dict)
    model.eval()

    output_dir = args.output_dir or os.path.join(
        project_root,
        "results",
        "inverse_design",
        f"screen_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )

    seed_sequence = np.random.SeedSequence(args.seed)
    child_seeds = seed_sequence.spawn(args.num_candidates)
    all_records = []
    top_heap = []

    with torch.no_grad():
        for batch_start in range(0, args.num_candidates, args.batch_size):
            batch_end = min(batch_start + args.batch_size, args.num_candidates)
            records = []
            geoms = []
            for idx in range(batch_start, batch_end):
                seed = int(child_seeds[idx].generate_state(1)[0])
                candidate_id, geom = _sample_candidate(args, seed, idx)
                records.append(_candidate_record(candidate_id, geom))
                geoms.append(geom)

            masks, scalars = _make_inputs(geoms, scalar_mean, scalar_std, device)
            outputs = model(masks, scalars).detach().cpu().numpy().reshape(-1)
            if checkpoint_meta.get("normalize_target", False):
                outputs = outputs * float(checkpoint_meta["target_std"]) + float(checkpoint_meta["target_mean"])

            for record, geom, pred in zip(records, geoms, outputs):
                record["predicted_delta_T"] = float(pred)
                all_records.append(record)
                _update_top_heap(top_heap, args.top_k, pred, record, geom["mask_3d"])

            if batch_end % max(args.batch_size * 10, 1) == 0 or batch_end == args.num_candidates:
                print(f"Screened {batch_end}/{args.num_candidates} candidates")

    all_path, top_path, masks_path, config_path = _save_screen_outputs(
        args,
        output_dir,
        all_records,
        top_heap,
        checkpoint_meta,
    )
    print("\nSaved inverse-design screening outputs:")
    print(f"- {all_path}")
    print(f"- {top_path}")
    print(f"- {masks_path}")
    print(f"- {config_path}")


def _load_top_masks(screen_dir):
    masks_path = os.path.join(screen_dir, "top_candidate_masks.npz")
    data = np.load(masks_path, allow_pickle=False)
    candidate_ids = data["candidate_ids"].astype(str)
    masks = data["masks"].astype(bool)
    return {candidate_id: masks[idx] for idx, candidate_id in enumerate(candidate_ids)}


VERIFY_COLUMNS = [
    "candidate_id",
    "simulation_id",
    "surrogate_rank",
    "predicted_delta_T",
    "fdm_delta_T",
    "residual",
    "geometry_type",
    "geometry_parameters",
]


def _load_existing_verifications(out_path):
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        return pd.DataFrame(columns=VERIFY_COLUMNS), set()

    existing_df = pd.read_csv(out_path)
    if "candidate_id" not in existing_df.columns:
        raise ValueError(f"Existing verification CSV has no candidate_id column: {out_path}")

    verified_ids = set(existing_df["candidate_id"].dropna().astype(str))
    return existing_df, verified_ids


def verify_candidates(args):
    top_path = os.path.join(args.screen_dir, "top_candidates.csv")
    if not os.path.exists(top_path):
        raise FileNotFoundError(f"Top candidate CSV not found: {top_path}")
    if args.start_rank < 1:
        raise ValueError("--start-rank must be >= 1")
    if args.verify_count < 1:
        raise ValueError("--verify-count must be >= 1")

    top_df = pd.read_csv(top_path)
    mask_map = _load_top_masks(args.screen_dir)
    out_path = args.output_csv or os.path.join(args.screen_dir, "verified_candidates.csv")
    existing_df, verified_ids = _load_existing_verifications(out_path)

    rank_end = args.start_rank + args.verify_count - 1
    verify_df = top_df[
        (top_df["surrogate_rank"] >= args.start_rank) & (top_df["surrogate_rank"] <= rank_end)
    ].copy()
    if verify_df.empty:
        print(f"No candidates found for surrogate ranks {args.start_rank}-{rank_end}")
        return

    before_skip = len(verify_df)
    verify_df = verify_df[~verify_df["candidate_id"].astype(str).isin(verified_ids)].copy()
    skipped_count = before_skip - len(verify_df)
    if skipped_count:
        print(f"Skipping {skipped_count} already verified candidates in ranks {args.start_rank}-{rank_end}")
    if verify_df.empty:
        print(f"No new candidates to verify. Existing results are in {out_path}")
        return

    rows = []

    for _, row in verify_df.iterrows():
        candidate_id = row["candidate_id"]
        if candidate_id not in mask_map:
            raise KeyError(f"Missing mask for candidate {candidate_id}")

        geom = json.loads(row["geometry_parameters"])
        geom["mask_3d"] = mask_map[candidate_id]
        sim_id = f"inverse_{candidate_id}_{uuid.uuid4().hex[:8]}"
        metadata = run_simulation_pipeline(geom, sim_id)
        actual = metadata.get("delta_T_parallel")
        predicted = float(row["predicted_delta_T"])
        rows.append(
            {
                "candidate_id": candidate_id,
                "simulation_id": sim_id,
                "surrogate_rank": int(row["surrogate_rank"]),
                "predicted_delta_T": predicted,
                "fdm_delta_T": actual,
                "residual": None if actual is None else predicted - actual,
                "geometry_type": row["geometry_type"],
                "geometry_parameters": row["geometry_parameters"],
            }
        )

    new_df = pd.DataFrame(rows, columns=VERIFY_COLUMNS)
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=["candidate_id"], keep="first")
    if "surrogate_rank" in combined_df.columns:
        combined_df = combined_df.sort_values("surrogate_rank", kind="stable")
    combined_df.to_csv(out_path, index=False)
    print(f"\nSaved verified candidate results to {out_path}")
    print(f"Verified {len(new_df)} new candidates; total rows in CSV: {len(combined_df)}")


def _safe_filename(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _plot_mask_on_axis(ax, mask, row, title_prefix, elev, azim):
    colors = np.empty(mask.shape, dtype=object)
    colors[:] = "#00000000"
    colors[mask] = "#4ECDC4"

    ax.voxels(mask, facecolors=colors, edgecolor="#1F2933", linewidth=0.08, alpha=0.92)
    ax.set_title(
        (
            f"{title_prefix} | rank {int(row['surrogate_rank'])} | "
            f"FDM {float(row['fdm_delta_T']):.3f} K\n"
            f"{row['candidate_id']} | {row['geometry_type']}"
        ),
        fontsize=8,
    )
    ax.set_xlabel("X", labelpad=-4)
    ax.set_ylabel("Y", labelpad=-4)
    ax.set_zlabel("Z", labelpad=-4)
    ax.tick_params(labelsize=6, pad=-2)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect(mask.shape)


def plot_top_verified(args):
    if args.top_n < 1:
        raise ValueError("--top-n must be >= 1")

    os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "te_film_matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    verified_csv = args.verified_csv or os.path.join(args.screen_dir, "verified_candidates.csv")
    if not os.path.exists(verified_csv):
        raise FileNotFoundError(f"Verified candidate CSV not found: {verified_csv}")

    verified_df = pd.read_csv(verified_csv)
    required_columns = {"candidate_id", "surrogate_rank", "fdm_delta_T", "geometry_type"}
    missing_columns = required_columns - set(verified_df.columns)
    if missing_columns:
        raise ValueError(f"Verified CSV is missing required columns: {sorted(missing_columns)}")

    verified_df["fdm_delta_T"] = pd.to_numeric(verified_df["fdm_delta_T"], errors="coerce")
    top_df = verified_df.dropna(subset=["fdm_delta_T"]).sort_values("fdm_delta_T", ascending=False).head(args.top_n)
    if top_df.empty:
        raise ValueError(f"No numeric fdm_delta_T values found in {verified_csv}")

    mask_map = _load_top_masks(args.screen_dir)
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(verified_csv))
    os.makedirs(output_dir, exist_ok=True)

    saved_paths = []
    if args.save_individual:
        for top_idx, (_, row) in enumerate(top_df.iterrows(), start=1):
            candidate_id = str(row["candidate_id"])
            if candidate_id not in mask_map:
                raise KeyError(f"Missing mask for candidate {candidate_id}")

            fig = plt.figure(figsize=(7.2, 5.6), dpi=180)
            ax = fig.add_subplot(111, projection="3d")
            _plot_mask_on_axis(ax, mask_map[candidate_id], row, f"Top {top_idx}", args.elev, args.azim)
            fig.tight_layout()

            filename = (
                f"top{top_idx:02d}_fdm_{float(row['fdm_delta_T']):.3f}_"
                f"rank{int(row['surrogate_rank']):03d}_{_safe_filename(candidate_id)}.png"
            )
            save_path = os.path.join(output_dir, filename)
            fig.savefig(save_path, bbox_inches="tight")
            plt.close(fig)
            saved_paths.append(save_path)

    ncols = min(5, len(top_df))
    nrows = math.ceil(len(top_df) / ncols)
    fig = plt.figure(figsize=(4.2 * ncols, 3.8 * nrows), dpi=180)
    for top_idx, (_, row) in enumerate(top_df.iterrows(), start=1):
        candidate_id = str(row["candidate_id"])
        if candidate_id not in mask_map:
            raise KeyError(f"Missing mask for candidate {candidate_id}")

        ax = fig.add_subplot(nrows, ncols, top_idx, projection="3d")
        _plot_mask_on_axis(ax, mask_map[candidate_id], row, f"Top {top_idx}", args.elev, args.azim)
    fig.tight_layout()

    overview_path = os.path.join(output_dir, f"top{len(top_df):02d}_verified_structures.png")
    fig.savefig(overview_path, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(overview_path)

    print("\nSaved FDM-ranked structure figures:")
    for path in saved_paths:
        print(f"- {path}")


def build_parser():
    parser = argparse.ArgumentParser(description="Surrogate-assisted inverse design for TE film structures.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    screen = subparsers.add_parser("screen", help="Generate many candidates and rank them with the surrogate.")
    screen.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    screen.add_argument("--metadata-csv", type=str, default=None)
    screen.add_argument("--root-dir", type=str, default=None)
    screen.add_argument("--output-dir", type=str, default=None)
    screen.add_argument("--num-candidates", type=int, default=10000)
    screen.add_argument("--top-k", type=int, default=500)
    screen.add_argument("--batch-size", type=int, default=256)
    screen.add_argument("--seed", type=int, default=20260511)
    screen.add_argument("--device", type=str, default="auto")
    screen.add_argument("--mode", choices=["structured", "mixed", "random"], default="mixed")
    screen.add_argument("--structured-ratio", type=float, default=0.9)
    screen.add_argument("--Lx", type=float, default=0.01)
    screen.add_argument("--Ly", type=float, default=0.01)
    screen.add_argument("--nx", type=int, default=50)
    screen.add_argument("--ny", type=int, default=50)
    screen.add_argument("--nz", type=int, default=20)
    screen.add_argument("--h-min", type=float, default=0.0005)
    screen.add_argument("--h-max", type=float, default=0.002)
    screen.add_argument("--k-low-min", type=float, default=0.08)
    screen.add_argument("--k-low-max", type=float, default=0.5)
    screen.add_argument("--k-high-min", type=float, default=1.0)
    screen.add_argument("--k-high-max", type=float, default=5.0)
    screen.add_argument("--fixed-h", type=float, default=None, help="Fix film thickness during candidate screening.")
    screen.add_argument("--fixed-k-low", type=float, default=None, help="Fix low-conductivity material value.")
    screen.add_argument("--fixed-k-high", type=float, default=None, help="Fix high-conductivity material value.")
    screen.add_argument("--fixed-T-hot", type=float, default=None, help="Fix hot-side temperature.")
    screen.add_argument("--fixed-T-air", type=float, default=None, help="Fix ambient/cold-side air temperature.")
    screen.add_argument("--fixed-h-c", type=float, default=None, help="Fix top convection coefficient.")
    screen.add_argument("--fixed-h-c-side", type=float, default=None, help="Fix side convection coefficient.")
    screen.set_defaults(func=screen_candidates)

    verify = subparsers.add_parser("verify", help="Run FDM verification for screened top candidates.")
    verify.add_argument("--screen-dir", type=str, required=True)
    verify.add_argument("--verify-count", type=int, default=50)
    verify.add_argument("--start-rank", type=int, default=1, help="First surrogate rank to verify, using 1-based ranks.")
    verify.add_argument("--output-csv", type=str, default=None)
    verify.set_defaults(func=verify_candidates)

    plot_top = subparsers.add_parser("plot-top", help="Plot the top FDM-verified structures from a screen directory.")
    plot_top.add_argument("--screen-dir", type=str, required=True)
    plot_top.add_argument("--verified-csv", type=str, default=None)
    plot_top.add_argument("--top-n", type=int, default=10)
    plot_top.add_argument("--output-dir", type=str, default=None, help="Defaults to the folder containing verified_candidates.csv.")
    plot_top.add_argument("--elev", type=float, default=30.0)
    plot_top.add_argument("--azim", type=float, default=45.0)
    plot_top.add_argument("--save-individual", action="store_true", help="Also save one PNG per top candidate.")
    plot_top.set_defaults(func=plot_top_verified)

    return parser


if __name__ == "__main__":
    parsed_args = build_parser().parse_args()
    parsed_args.func(parsed_args)
