import argparse
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset, random_split

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "te_film_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(tempfile.gettempdir(), "te_film_cache"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
project_root = os.path.dirname(os.path.dirname(current_dir))

from dataset import TEFilmDataset
from models import ThermoNetFusion


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


def _make_split(dataset, split_name, seed):
    total_samples = len(dataset)
    train_size = int(0.8 * total_samples)
    val_size = int(0.1 * total_samples)
    test_size = total_samples - train_size - val_size

    splits = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )
    split_map = {"train": splits[0], "val": splits[1], "test": splits[2], "all": dataset}
    selected = split_map[split_name]
    if isinstance(selected, Subset):
        indices = list(selected.indices)
    else:
        indices = list(range(total_samples))
    return selected, indices


def _metrics(y_true, y_pred):
    residual = y_pred - y_true
    mae = np.mean(np.abs(residual))
    rmse = np.sqrt(np.mean(residual**2))
    bias = np.mean(residual)
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = np.nan if denom <= 0 else 1.0 - np.sum(residual**2) / denom
    return {
        "count": int(len(y_true)),
        "mae_K": float(mae),
        "rmse_K": float(rmse),
        "bias_K": float(bias),
        "r2": float(r2) if not np.isnan(r2) else None,
    }


def _ranking_metrics(predictions, top_quantile):
    true_cutoff = predictions["delta_T_true"].quantile(top_quantile)
    pred_cutoff = predictions["delta_T_pred"].quantile(top_quantile)
    true_top = predictions["delta_T_true"] >= true_cutoff
    pred_top = predictions["delta_T_pred"] >= pred_cutoff
    overlap = true_top & pred_top
    spearman = predictions[["delta_T_true", "delta_T_pred"]].corr(method="spearman").iloc[0, 1]

    return {
        "quantile": float(top_quantile),
        "true_delta_T_cutoff_K": float(true_cutoff),
        "pred_delta_T_cutoff_K": float(pred_cutoff),
        "true_top_count": int(true_top.sum()),
        "pred_top_count": int(pred_top.sum()),
        "top_overlap_count": int(overlap.sum()),
        "top_recall": float(overlap.sum() / max(true_top.sum(), 1)),
        "top_precision": float(overlap.sum() / max(pred_top.sum(), 1)),
        "spearman_rank_corr": float(spearman) if not np.isnan(spearman) else None,
    }


def _save_scatter(predictions, output_dir, split_name):
    y_true = predictions["delta_T_true"].to_numpy()
    y_pred = predictions["delta_T_pred"].to_numpy()
    lo = min(float(np.min(y_true)), float(np.min(y_pred)))
    hi = max(float(np.max(y_true)), float(np.max(y_pred)))
    pad = max((hi - lo) * 0.05, 1e-6)

    fig, ax = plt.subplots(figsize=(6.5, 6.0), dpi=160)
    ax.scatter(y_true, y_pred, s=10, alpha=0.45, edgecolors="none")
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", linewidth=1.2)
    ax.set_xlabel("FDM delta_T_parallel (K)")
    ax.set_ylabel("Predicted delta_T_parallel (K)")
    ax.set_title(f"ThermoNet predictions: {split_name}")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    fig.tight_layout()
    path = os.path.join(output_dir, f"prediction_scatter_{split_name}.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def _save_residual_histogram(predictions, output_dir, split_name):
    residual = predictions["delta_T_pred"].to_numpy() - predictions["delta_T_true"].to_numpy()

    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=160)
    ax.hist(residual, bins=50, color="#4f83cc", alpha=0.85)
    ax.axvline(0.0, color="black", linewidth=1.1)
    ax.set_xlabel("Prediction residual (K)")
    ax.set_ylabel("Count")
    ax.set_title(f"Residual distribution: {split_name}")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    path = os.path.join(output_dir, f"residual_histogram_{split_name}.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def _evaluate_loader(model, loader, device):
    y_true_batches = []
    y_pred_batches = []

    model.eval()
    with torch.no_grad():
        for masks, scalars, targets in loader:
            masks = masks.to(device)
            scalars = scalars.to(device)
            outputs = model(masks, scalars)
            y_true_batches.append(targets.cpu().numpy().reshape(-1))
            y_pred_batches.append(outputs.cpu().numpy().reshape(-1))

    return np.concatenate(y_true_batches), np.concatenate(y_pred_batches)


def evaluate_model(args):
    device = _device_from_arg(args.device)
    print(f"Using device: {device}")

    metadata_csv = args.metadata_csv or os.path.join(project_root, "data", "simulations", "metadata.csv")
    root_dir = args.root_dir or os.path.join(project_root, "data", "simulations")
    model_path = args.model_path or os.path.join(project_root, "results", "models", "best_thermonet.pth")
    output_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    state_dict, checkpoint_meta = _load_checkpoint(model_path, device)
    include_boundary_channel = bool(
        checkpoint_meta.get("include_boundary_channel", args.include_boundary_channel)
    )
    checkpoint_scalar_cols = checkpoint_meta.get("scalar_cols")
    dataset = TEFilmDataset(
        metadata_csv=metadata_csv,
        root_dir=root_dir,
        include_boundary_channel=include_boundary_channel,
        scalar_cols=checkpoint_scalar_cols,
    )
    if len(dataset) == 0:
        raise RuntimeError("No successful samples were found in metadata.csv.")

    split_dataset, split_indices = _make_split(dataset, args.split, args.seed)
    print(f"Evaluating split: {args.split} ({len(split_indices)} samples)")

    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    loader = DataLoader(split_dataset, **loader_kwargs)

    input_channels = int(checkpoint_meta.get("input_channels", dataset.input_channels))
    scalar_dim = int(checkpoint_meta.get("scalar_dim", len(checkpoint_scalar_cols or dataset.scalar_cols)))
    print(f"Scalar inputs ({len(dataset.scalar_cols)}): {dataset.scalar_cols}")
    print(f"Model scalar_dim={scalar_dim}, input_channels={input_channels}")
    model = ThermoNetFusion(scalar_dim=scalar_dim, input_channels=input_channels).to(device)
    model.load_state_dict(state_dict)

    y_true, y_pred = _evaluate_loader(model, loader, device)
    if checkpoint_meta.get("normalize_target", False):
        target_mean = float(checkpoint_meta["target_mean"])
        target_std = float(checkpoint_meta["target_std"])
        y_pred = y_pred * target_std + target_mean
        print(f"Checkpoint output unnormalized with target mean={target_mean:.6g}, std={target_std:.6g}")

    metadata = dataset.data_frame.iloc[split_indices].reset_index(drop=True).copy()
    prediction_cols = [
        col for col in [
            "simulation_id", "geometry_type", "database_profile", "scenario_id",
            "thickness_h", "k_low", "k_high", "k_ratio", "T_hot", "T_air",
            "h_c", "h_c_side", "convection_regime", "hot_boundary_type",
            "curvature_type", "curvature_level",
        ]
        if col in metadata.columns
    ]
    predictions = metadata[prediction_cols].copy()
    predictions["delta_T_true"] = y_true
    predictions["delta_T_pred"] = y_pred
    predictions["residual"] = predictions["delta_T_pred"] - predictions["delta_T_true"]
    predictions["abs_error"] = predictions["residual"].abs()

    overall = _metrics(y_true, y_pred)
    per_family = (
        predictions.groupby("geometry_type", dropna=False)
        .apply(lambda group: pd.Series(_metrics(group["delta_T_true"].to_numpy(), group["delta_T_pred"].to_numpy())))
        .reset_index()
        .sort_values(["rmse_K", "mae_K"], ascending=False)
    )
    per_scenario = None
    if "scenario_id" in predictions.columns:
        per_scenario = (
            predictions.groupby("scenario_id", dropna=False)
            .apply(lambda group: pd.Series(_metrics(group["delta_T_true"].to_numpy(), group["delta_T_pred"].to_numpy())))
            .reset_index()
            .sort_values(["rmse_K", "mae_K"], ascending=False)
        )

    top_quantile_cutoff = predictions["delta_T_true"].quantile(args.top_quantile)
    top_predictions = predictions[predictions["delta_T_true"] >= top_quantile_cutoff]
    top_metrics = _metrics(
        top_predictions["delta_T_true"].to_numpy(),
        top_predictions["delta_T_pred"].to_numpy(),
    )
    top_metrics["true_delta_T_cutoff_K"] = float(top_quantile_cutoff)
    top_metrics["quantile"] = float(args.top_quantile)
    ranking = _ranking_metrics(predictions, args.top_quantile)

    predictions_path = os.path.join(output_dir, f"predictions_{args.split}.csv")
    per_family_path = os.path.join(output_dir, f"per_family_metrics_{args.split}.csv")
    per_scenario_path = os.path.join(output_dir, f"per_scenario_metrics_{args.split}.csv")
    metrics_path = os.path.join(output_dir, f"metrics_{args.split}.json")
    scatter_path = _save_scatter(predictions, output_dir, args.split)
    residual_path = _save_residual_histogram(predictions, output_dir, args.split)

    predictions.to_csv(predictions_path, index=False)
    per_family.to_csv(per_family_path, index=False)
    if per_scenario is not None:
        per_scenario.to_csv(per_scenario_path, index=False)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {"overall": overall, "top_delta_T_region": top_metrics, "ranking": ranking},
            f,
            indent=2,
        )

    print("\nOverall metrics:")
    print(json.dumps(overall, indent=2))
    print(f"\nTop delta_T region metrics (true delta_T >= {top_quantile_cutoff:.6g} K):")
    print(json.dumps(top_metrics, indent=2))
    print("\nRanking metrics:")
    print(json.dumps(ranking, indent=2))
    print("\nPer-family metrics:")
    print(per_family.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    if per_scenario is not None:
        print("\nWorst scenario metrics:")
        print(per_scenario.head(15).to_string(index=False, float_format=lambda value: f"{value:.6g}"))

    print("\nSaved outputs:")
    print(f"- {metrics_path}")
    print(f"- {per_family_path}")
    if per_scenario is not None:
        print(f"- {per_scenario_path}")
    print(f"- {predictions_path}")
    print(f"- {scatter_path}")
    print(f"- {residual_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ThermoNet on a held-out split.")
    parser.add_argument("--model-path", type=str, default=None, help="Path to best_thermonet.pth.")
    parser.add_argument("--metadata-csv", type=str, default=None, help="Path to metadata.csv.")
    parser.add_argument("--root-dir", type=str, default=None, help="Directory containing the fields/ folder.")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--seed", type=int, default=42, help="Must match the training split seed.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, cuda:0, or mps.")
    parser.add_argument("--output-dir", type=str, default=os.path.join("results", "evaluation"))
    parser.add_argument("--include-boundary-channel", action="store_true", help="Use hot-boundary map channel when evaluating checkpoints that were trained with it.")
    parser.add_argument(
        "--top-quantile",
        type=float,
        default=0.9,
        help="Evaluate the high-performance region above this true delta_T quantile.",
    )
    evaluate_model(parser.parse_args())
