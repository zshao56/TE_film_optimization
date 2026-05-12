import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent
project_root = src_dir.parent
AUTO_CUDA_VALUES = {"auto", "least_busy", "most_free"}


def _load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _stage_enabled(config, key):
    return bool(config.get("run", {}).get(key, False))


def _cuda_value(config, stage_config):
    base_env = config.get("environment", {})
    return stage_config.get("cuda_visible_devices", base_env.get("cuda_visible_devices"))


def _is_auto_cuda(value):
    return isinstance(value, str) and value.strip().lower() in AUTO_CUDA_VALUES


def _parse_cuda_candidates(config):
    candidates = config.get("environment", {}).get("auto_cuda_devices", [0, 1])
    return {str(candidate) for candidate in candidates}


def _query_cuda_devices():
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.free,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    devices = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        index, memory_free, memory_used, utilization = parts
        devices.append(
            {
                "index": index,
                "memory_free_mib": int(memory_free),
                "memory_used_mib": int(memory_used),
                "utilization_pct": int(utilization),
            }
        )
    if not devices:
        raise RuntimeError("nvidia-smi returned no CUDA devices.")
    return devices


def _select_cuda_device(config):
    candidates = _parse_cuda_candidates(config)
    devices = [device for device in _query_cuda_devices() if device["index"] in candidates]
    if not devices:
        raise RuntimeError(f"No CUDA devices matched configured candidates: {sorted(candidates)}")

    selected = sorted(
        devices,
        key=lambda device: (
            -device["memory_free_mib"],
            device["utilization_pct"],
            device["memory_used_mib"],
            int(device["index"]),
        ),
    )[0]
    print("CUDA device status:")
    for device in devices:
        selected_marker = " <- selected" if device["index"] == selected["index"] else ""
        print(
            f"  GPU {device['index']}: free={device['memory_free_mib']} MiB, "
            f"used={device['memory_used_mib']} MiB, util={device['utilization_pct']}%"
            f"{selected_marker}"
        )
    return selected["index"]


def _resolve_cuda_for_stage(config, stage_config, resolve_auto=False):
    cuda = _cuda_value(config, stage_config)
    if not _is_auto_cuda(cuda):
        return cuda
    if not resolve_auto:
        return None
    if "_resolved_cuda_visible_devices" not in config:
        config["_resolved_cuda_visible_devices"] = _select_cuda_device(config)
    return config["_resolved_cuda_visible_devices"]


def _env_for_stage(config, stage_config, resolve_cuda=False):
    env = os.environ.copy()
    base_env = config.get("environment", {})
    for key, value in base_env.items():
        if str(key).lower() in {"cuda_visible_devices", "auto_cuda_devices"}:
            continue
        env[str(key).upper()] = str(value)
    for key, value in stage_config.get("environment", {}).items():
        if str(key).lower() in {"cuda_visible_devices", "auto_cuda_devices"}:
            continue
        env[str(key).upper()] = str(value)
    cuda = _resolve_cuda_for_stage(config, stage_config, resolve_auto=resolve_cuda)
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
    _run_command(
        command,
        data_config.get("log_path", "logs/01_generate_configured.log"),
        env=_env_for_stage(config, data_config, resolve_cuda=False),
    )


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
    _run_command(
        command,
        filter_config.get("log_path", "logs/01b_filter_metadata_configured.log"),
        env=_env_for_stage(config, filter_config, resolve_cuda=False),
    )


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
    _run_command(
        command,
        train_config.get("log_path", "logs/02_train_configured.log"),
        env=_env_for_stage(config, train_config, resolve_cuda=True),
    )


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
        _run_command(command, log_template.format(split=split), env=_env_for_stage(config, eval_config, resolve_cuda=True))


def run_real_world_benchmark(config, config_path):
    benchmark_config = config.get("real_world_benchmark", {})
    command = [
        sys.executable,
        "src/optimization/real_world_benchmark.py",
        "--config",
        str(config_path),
    ]
    _run_command(
        command,
        benchmark_config.get("log_path", "logs/04_real_world_benchmark.log"),
        env=_env_for_stage(config, benchmark_config, resolve_cuda=True),
    )


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
