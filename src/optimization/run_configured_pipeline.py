import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent
project_root = src_dir.parent


def _load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _stage_enabled(config, key):
    return bool(config.get("run", {}).get(key, False))


def _env_for_stage(config, stage_config):
    env = os.environ.copy()
    base_env = config.get("environment", {})
    for key, value in base_env.items():
        env[str(key).upper()] = str(value)
    for key, value in stage_config.get("environment", {}).items():
        env[str(key).upper()] = str(value)
    cuda = stage_config.get("cuda_visible_devices", base_env.get("cuda_visible_devices"))
    if cuda is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda)
    return env


def _run_command(command, log_path=None, env=None):
    print("\n$ " + " ".join(command))
    log_file = None
    if log_path is not None:
        log_path = project_root / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "w", encoding="utf-8")
        print(f"Logging to {log_path}")

    process = subprocess.Popen(
        command,
        cwd=project_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        for line in process.stdout:
            print(line, end="")
            if log_file is not None:
                log_file.write(line)
                log_file.flush()
    finally:
        if log_file is not None:
            log_file.close()

    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def _clear_database_if_requested(config):
    data_config = config.get("data_generation", {})
    if not data_config.get("clear_existing", False):
        return
    if not data_config.get("allow_delete_data", False):
        raise ValueError("Refusing to clear data unless data_generation.allow_delete_data is true.")

    simulations_dir = project_root / "data" / "simulations"
    metadata_path = simulations_dir / "metadata.csv"
    fields_dir = simulations_dir / "fields"

    if metadata_path.exists():
        metadata_path.unlink()
        print(f"Deleted {metadata_path}")
    if fields_dir.exists():
        for h5_path in fields_dir.glob("*.h5"):
            h5_path.unlink()
        print(f"Deleted HDF5 fields under {fields_dir}")
    fields_dir.mkdir(parents=True, exist_ok=True)


def _bool_arg(command, condition, flag):
    if condition:
        command.append(flag)


def _append_option(command, config, key, flag):
    if key in config and config[key] is not None:
        command.extend([flag, str(config[key])])


def run_data_generation(config, config_path):
    data_config = config.get("data_generation", {})
    _clear_database_if_requested(config)
    command = [
        sys.executable,
        "src/generate_database.py",
        "--config",
        str(config_path),
    ]
    _run_command(command, data_config.get("log_path", "logs/01_generate_configured.log"), env=_env_for_stage(config, data_config))


def run_metadata_filter(config):
    filter_config = config.get("metadata_filter", {})
    command = [sys.executable, "scripts/filter_physical_metadata.py"]
    option_map = {
        "metadata": "--metadata",
        "output": "--output",
        "bad_output": "--bad-output",
        "report": "--report",
        "tolerance": "--tolerance",
        "max_relative_residual": "--max-relative-residual",
        "head": "--head",
    }
    for key, flag in option_map.items():
        _append_option(command, filter_config, key, flag)
    _bool_arg(command, filter_config.get("require_qc_pass", False), "--require-qc-pass")
    _run_command(command, filter_config.get("log_path", "logs/01b_filter_metadata_configured.log"), env=_env_for_stage(config, filter_config))


def run_training(config):
    train_config = config.get("training", {})
    command = [sys.executable, "src/optimization/train.py"]
    option_map = {
        "metadata_csv": "--metadata-csv",
        "root_dir": "--root-dir",
        "batch_size": "--batch-size",
        "epochs": "--epochs",
        "lr": "--lr",
        "weight_decay": "--weight-decay",
        "patience": "--patience",
        "min_delta": "--min-delta",
        "lr_patience": "--lr-patience",
        "lr_factor": "--lr-factor",
        "min_lr": "--min-lr",
        "grad_clip": "--grad-clip",
        "seed": "--seed",
        "run_name": "--run-name",
        "metadata_report_dir": "--metadata-report-dir",
        "workers": "--workers",
        "top_quantile": "--top-quantile",
        "top_weight": "--top-weight",
        "underpredict_penalty": "--underpredict-penalty",
        "underpredict_quantile": "--underpredict-quantile",
    }
    for key, flag in option_map.items():
        _append_option(command, train_config, key, flag)
    _bool_arg(command, train_config.get("normalize_target", False), "--normalize-target")
    _bool_arg(command, train_config.get("include_boundary_channel", False), "--include-boundary-channel")
    _run_command(command, train_config.get("log_path", "logs/02_train_configured.log"), env=_env_for_stage(config, train_config))


def run_evaluation(config):
    eval_config = config.get("evaluation", {})
    splits = eval_config.get("splits", ["test"])
    for split in splits:
        command = [sys.executable, "src/optimization/evaluate.py", "--split", str(split)]
        option_map = {
            "model_path": "--model-path",
            "metadata_csv": "--metadata-csv",
            "root_dir": "--root-dir",
            "seed": "--seed",
            "batch_size": "--batch-size",
            "workers": "--workers",
            "device": "--device",
            "top_quantile": "--top-quantile",
        }
        for key, flag in option_map.items():
            _append_option(command, eval_config, key, flag)
        output_dir = eval_config.get("output_dir")
        if output_dir:
            command.extend(["--output-dir", str(Path(output_dir) / str(split))])
        _bool_arg(command, eval_config.get("include_boundary_channel", False), "--include-boundary-channel")
        log_template = eval_config.get("log_path_template", "logs/03_eval_{split}.log")
        _run_command(command, log_template.format(split=split), env=_env_for_stage(config, eval_config))


def run_real_world_benchmark(config, config_path):
    benchmark_config = config.get("real_world_benchmark", {})
    command = [
        sys.executable,
        "src/optimization/real_world_benchmark.py",
        "--config",
        str(config_path),
    ]
    _run_command(command, benchmark_config.get("log_path", "logs/04_real_world_benchmark.log"), env=_env_for_stage(config, benchmark_config))


def run_pipeline(config_path):
    config_path = Path(config_path).resolve()
    config = _load_config(config_path)
    if _stage_enabled(config, "data_generation"):
        run_data_generation(config, config_path)
    if _stage_enabled(config, "metadata_filter"):
        run_metadata_filter(config)
    if _stage_enabled(config, "training"):
        run_training(config)
    if _stage_enabled(config, "evaluation"):
        run_evaluation(config)
    if _stage_enabled(config, "real_world_benchmark"):
        run_real_world_benchmark(config, config_path)


def build_parser():
    parser = argparse.ArgumentParser(description="Run the TE film workflow from one JSON config file.")
    parser.add_argument("--config", required=True, help="Path to pipeline JSON config.")
    return parser


if __name__ == "__main__":
    run_pipeline(build_parser().parse_args().config)
