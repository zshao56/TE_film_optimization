import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))


DEFAULT_PENALTIES = [0.05, 0.1, 0.15, 0.2, 0.25]


def _project_path(*parts):
    return os.path.join(project_root, *parts)


def _penalty_label(value):
    return f"{value:g}".replace(".", "p")


def _run_command(command, dry_run=False):
    print("\n$ " + " ".join(command), flush=True)
    if dry_run:
        return 0
    process = subprocess.Popen(
        command,
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in process.stdout:
        print(line, end="", flush=True)
    return process.wait()


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _checkpoint_path():
    return _project_path("results", "models", "best_thermonet.pth")


def _experiment_dir(output_root, run_name):
    return _project_path(output_root, run_name)


def _metrics_path(output_root, run_name, split):
    return os.path.join(_experiment_dir(output_root, run_name), "evaluation", f"metrics_{split}.json")


def _saved_checkpoint_path(output_root, run_name):
    return os.path.join(_experiment_dir(output_root, run_name), "best_thermonet.pth")


def _command_to_string(command):
    return " ".join(command)


def _build_train_command(args, run_name, penalty):
    command = [
        sys.executable,
        os.path.join("src", "optimization", "train.py"),
        "--batch-size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--seed",
        str(args.seed),
        "--workers",
        str(args.workers),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--patience",
        str(args.patience),
        "--normalize-target",
        "--top-quantile",
        str(args.top_quantile),
        "--underpredict-penalty",
        str(penalty),
        "--run-name",
        run_name,
        "--metadata-report-dir",
        os.path.join(args.output_root, run_name, "metadata"),
    ]
    if args.underpredict_quantile is not None:
        command.extend(["--underpredict-quantile", str(args.underpredict_quantile)])
    return command


def _build_eval_command(args, run_name):
    return [
        sys.executable,
        os.path.join("src", "optimization", "evaluate.py"),
        "--model-path",
        _saved_checkpoint_path(args.output_root, run_name),
        "--split",
        args.split,
        "--seed",
        str(args.seed),
        "--batch-size",
        str(args.eval_batch_size),
        "--workers",
        str(args.workers),
        "--top-quantile",
        str(args.top_quantile),
        "--output-dir",
        os.path.join(args.output_root, run_name, "evaluation"),
    ]


def _copy_current_checkpoint(args, run_name):
    src = _checkpoint_path()
    dst = _saved_checkpoint_path(args.output_root, run_name)
    if not os.path.exists(src):
        raise FileNotFoundError(f"Checkpoint not found: {src}")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    print(f"[INFO] Checkpoint copied to {dst}")
    return dst


def _score_metrics(metrics):
    overall = metrics["overall"]
    top = metrics["top_delta_T_region"]
    ranking = metrics["ranking"]

    overall_r2 = overall.get("r2") or 0.0
    top_mae = top["mae_K"]
    top_bias_abs = abs(top["bias_K"])
    top_recall = ranking["top_recall"]
    top_precision = ranking["top_precision"]
    spearman = ranking.get("spearman_rank_corr") or 0.0

    score = 0.0
    score += 100.0 * overall_r2
    score += 30.0 * top_recall
    score += 30.0 * top_precision
    score += 20.0 * spearman
    score -= 8.0 * top_mae
    score -= 5.0 * top_bias_abs
    if overall_r2 >= 0.88:
        score += 10.0
    if top_mae <= 1.5:
        score += 10.0
    if top_bias_abs <= 0.8:
        score += 10.0
    if top_recall >= 0.82 and top_precision >= 0.82:
        score += 10.0
    return score


def _leaderboard_row(output_root, run_name, split):
    metrics_file = _metrics_path(output_root, run_name, split)
    if not os.path.exists(metrics_file):
        return None
    metrics = _read_json(metrics_file)
    overall = metrics["overall"]
    top = metrics["top_delta_T_region"]
    ranking = metrics["ranking"]
    return {
        "run_name": run_name,
        "score": _score_metrics(metrics),
        "overall_mae_K": overall["mae_K"],
        "overall_rmse_K": overall["rmse_K"],
        "overall_bias_K": overall["bias_K"],
        "overall_r2": overall["r2"],
        "top_mae_K": top["mae_K"],
        "top_rmse_K": top["rmse_K"],
        "top_bias_K": top["bias_K"],
        "top_r2": top["r2"],
        "top_recall": ranking["top_recall"],
        "top_precision": ranking["top_precision"],
        "top_overlap_count": ranking["top_overlap_count"],
        "spearman_rank_corr": ranking["spearman_rank_corr"],
        "metrics_path": metrics_file,
    }


def _update_leaderboard(args):
    if args.dry_run:
        print("[DRY-RUN] Leaderboard update skipped.")
        return []

    root = _project_path(args.output_root)
    rows = []
    if os.path.isdir(root):
        for name in sorted(os.listdir(root)):
            if not os.path.isdir(os.path.join(root, name)):
                continue
            row = _leaderboard_row(args.output_root, name, args.split)
            if row is not None:
                rows.append(row)

    rows.sort(key=lambda row: row["score"], reverse=True)
    leaderboard_path = _project_path(args.output_root, "leaderboard.csv")
    os.makedirs(os.path.dirname(leaderboard_path), exist_ok=True)

    fieldnames = [
        "rank",
        "run_name",
        "score",
        "overall_mae_K",
        "overall_rmse_K",
        "overall_bias_K",
        "overall_r2",
        "top_mae_K",
        "top_rmse_K",
        "top_bias_K",
        "top_r2",
        "top_recall",
        "top_precision",
        "top_overlap_count",
        "spearman_rank_corr",
        "metrics_path",
    ]
    with open(leaderboard_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow({"rank": rank, **row})

    _write_json(_project_path(args.output_root, "leaderboard.json"), rows)
    print(f"\n[INFO] Leaderboard written to {leaderboard_path}")
    if rows:
        print("[INFO] Current best:")
        best = rows[0]
        print(
            f"  {best['run_name']} | score={best['score']:.3f} | "
            f"overall_r2={best['overall_r2']:.4f} | top_mae={best['top_mae_K']:.4f} | "
            f"top_bias={best['top_bias_K']:.4f} | top_recall={best['top_recall']:.4f}"
        )
    return rows


def _save_run_metadata(args, run_name, payload):
    metadata_path = os.path.join(_experiment_dir(args.output_root, run_name), "run_metadata.json")
    if args.dry_run:
        print(f"[DRY-RUN] Run metadata would be written to {metadata_path}")
        return

    payload = {
        "run_name": run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        **payload,
    }
    _write_json(metadata_path, payload)


def import_current_checkpoint(args):
    run_name = args.import_current_run
    if not run_name:
        return

    eval_command = _build_eval_command(args, run_name)
    if args.dry_run:
        print(f"[DRY-RUN] Current checkpoint would be copied into run {run_name}")
    else:
        _copy_current_checkpoint(args, run_name)
    _save_run_metadata(
        args,
        run_name,
        {
            "mode": "import_current_checkpoint",
            "source_checkpoint": _checkpoint_path(),
            "evaluate_command": _command_to_string(eval_command),
        },
    )
    if _run_command(eval_command, args.dry_run) != 0:
        raise RuntimeError(f"Evaluation failed for imported run: {run_name}")
    _update_leaderboard(args)


def run_experiment(args, run_name, penalty):
    exp_dir = _experiment_dir(args.output_root, run_name)
    metrics_file = _metrics_path(args.output_root, run_name, args.split)
    checkpoint_file = _saved_checkpoint_path(args.output_root, run_name)
    if os.path.exists(metrics_file) and os.path.exists(checkpoint_file) and not args.force:
        print(f"[SKIP] {run_name} already has metrics and checkpoint. Use --force to rerun.")
        return

    train_command = _build_train_command(args, run_name, penalty)
    eval_command = _build_eval_command(args, run_name)
    _save_run_metadata(
        args,
        run_name,
        {
            "mode": "train_and_evaluate",
            "underpredict_penalty": penalty,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "epochs": args.epochs,
            "seed": args.seed,
            "top_quantile": args.top_quantile,
            "underpredict_quantile": args.underpredict_quantile,
            "train_command": _command_to_string(train_command),
            "evaluate_command": _command_to_string(eval_command),
        },
    )

    print(f"\n===== Running {run_name} =====")
    if _run_command(train_command, args.dry_run) != 0:
        raise RuntimeError(f"Training failed for run: {run_name}")
    if not args.dry_run:
        _copy_current_checkpoint(args, run_name)
    if _run_command(eval_command, args.dry_run) != 0:
        raise RuntimeError(f"Evaluation failed for run: {run_name}")
    _update_leaderboard(args)


def main(args):
    os.makedirs(_project_path(args.output_root), exist_ok=True)

    if args.import_current_run:
        import_current_checkpoint(args)
        if args.no_sweep:
            return

    for penalty in args.penalties:
        run_name = f"{args.run_prefix}_under_{_penalty_label(penalty)}_bs{args.batch_size}"
        run_experiment(args, run_name, penalty)

    _update_leaderboard(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ThermoNet training/evaluation sweeps and maintain a leaderboard.")
    parser.add_argument("--penalties", type=float, nargs="+", default=DEFAULT_PENALTIES, help="Underprediction penalty values to sweep.")
    parser.add_argument("--run-prefix", type=str, default="thermonet_auto", help="Prefix used to build run names.")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=128, help="Evaluation batch size.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--top-quantile", type=float, default=0.9)
    parser.add_argument("--underpredict-quantile", type=float, default=None)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--output-root", type=str, default=os.path.join("results", "experiments"))
    parser.add_argument("--import-current-run", type=str, default=None, help="Copy current results/models/best_thermonet.pth into this run and evaluate it.")
    parser.add_argument("--no-sweep", action="store_true", help="Only import/evaluate --import-current-run; do not run the penalty sweep.")
    parser.add_argument("--force", action="store_true", help="Rerun experiments even if their outputs already exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands and metadata without running training/evaluation.")
    main(parser.parse_args())
