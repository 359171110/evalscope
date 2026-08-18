from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from src.evalscope_model_api import (
    file_sha256,
    register_static_expert_profile_api,
    validate_static_profile_artifacts,
)


DEFAULT_DATASETS = ["mmlu_pro", "arc", "hellaswag", "gsm8k", "math_500", "ifeval"]
BOXED_ANSWER_PROMPT = (
    "{question}\nPlease reason step by step. End with "
    "a non-empty LaTeX \\boxed expression containing the computed final number "
    "inside its braces. Do not use an empty box or a placeholder."
)


def parse_json_object(value: str) -> dict:
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise argparse.ArgumentTypeError("value must be a JSON object.")
    return payload


def parse_limit(value: str) -> int | float:
    stripped = value.strip()
    if not stripped:
        raise argparse.ArgumentTypeError("limit must be non-empty.")
    if any(marker in stripped.lower() for marker in (".", "e")):
        parsed = float(stripped)
        if not 0.0 < parsed <= 1.0:
            raise argparse.ArgumentTypeError("fractional limit must be in (0, 1].")
        return parsed
    parsed = int(stripped)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("integer limit must be positive.")
    return parsed


def parse_dataset_limits(value: str) -> dict[str, int | float]:
    payload = parse_json_object(value)
    limits = {}
    for dataset, limit in payload.items():
        if not isinstance(dataset, str) or not dataset.strip():
            raise argparse.ArgumentTypeError("dataset limit keys must be non-empty strings.")
        if isinstance(limit, bool):
            raise argparse.ArgumentTypeError("dataset limits must be positive integers or fractions.")
        try:
            limits[dataset] = parse_limit(str(limit))
        except (TypeError, ValueError, argparse.ArgumentTypeError) as exc:
            raise argparse.ArgumentTypeError(f"invalid limit for dataset '{dataset}': {limit}") from exc
    return limits


def parse_sha256(value: str) -> str:
    normalized = value.strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", normalized) is None:
        raise argparse.ArgumentTypeError("SHA256 must contain exactly 64 hexadecimal characters.")
    return normalized


def source_tree_identity(repository: Path, *, pathspecs: tuple[str, ...]) -> dict:
    root = repository.expanduser().resolve()
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    files_output = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            *pathspecs,
        ],
        check=True,
        capture_output=True,
    ).stdout
    relative_paths = sorted(
        path.decode("utf-8")
        for path in files_output.split(b"\0")
        if path
    )
    digest = hashlib.sha256()
    for relative_path in relative_paths:
        path = root / relative_path
        if not path.is_file():
            continue
        path_bytes = relative_path.encode("utf-8")
        digest.update(len(path_bytes).to_bytes(8, byteorder="big"))
        digest.update(path_bytes)
        digest.update(bytes.fromhex(file_sha256(path)))
    status = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            *pathspecs,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "repository": str(root),
        "commit": commit,
        "runtime_pathspecs": list(pathspecs),
        "runtime_tree_dirty": bool(status.strip()),
        "runtime_tree_sha256": digest.hexdigest(),
        "runtime_file_count": len(relative_paths),
    }


def evalscope_source_root() -> Path:
    spec = importlib.util.find_spec("evalscope")
    if spec is None or spec.origin is None:
        raise RuntimeError("Unable to locate the EvalScope source package.")
    return Path(spec.origin).resolve().parent.parent


def validate_visible_gpus(value: str | None) -> list[str]:
    if value is None:
        raise ValueError("CUDA_VISIBLE_DEVICES must explicitly select physical GPU IDs.")
    devices = [item.strip() for item in value.split(",") if item.strip()]
    if (
        not devices
        or len(devices) != len(set(devices))
        or any(not item.isdigit() for item in devices)
    ):
        raise ValueError("CUDA_VISIBLE_DEVICES must be a unique comma-separated list of physical GPU IDs.")
    return devices


def normalize_dataset_args(
    datasets: list[str],
    dataset_args: dict,
    dataset_limits: dict[str, int | float] | None = None,
) -> dict:
    normalized = {
        str(name): dict(values)
        for name, values in dataset_args.items()
    }
    dataset_limits = dataset_limits or {}
    unknown_datasets = set(dataset_limits) - set(datasets)
    if unknown_datasets:
        unknown = ', '.join(sorted(unknown_datasets))
        raise ValueError(f"dataset limits specified for datasets not selected: {unknown}")
    for dataset in datasets:
        if dataset in dataset_limits:
            normalized.setdefault(dataset, {})['limit'] = dataset_limits[dataset]
        if dataset not in {"gsm8k", "math_500"}:
            continue
        values = normalized.setdefault(dataset, {})
        values.setdefault("prompt_template", BOXED_ANSWER_PROMPT)
        if dataset == "gsm8k":
            values.setdefault("few_shot_num", 0)
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a frozen static expert profile with EvalScope.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-family", default=None)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--channel-cache", type=Path, required=True)
    parser.add_argument("--expected-profile-file-sha256", type=parse_sha256, required=True)
    parser.add_argument("--expected-channel-file-sha256", type=parse_sha256, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path, default=None)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--dataset-args", type=parse_json_object, default={})
    parser.add_argument("--dataset-limits", type=parse_dataset_limits, default={})
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--dataset-hub", default="modelscope")
    parser.add_argument("--generation-config", type=parse_json_object, default={})
    parser.add_argument("--limit", type=parse_limit, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--correction-mode", default="none")
    parser.add_argument("--max-correction-ratio", type=float, default=0.20)
    parser.add_argument("--moe-backend", choices=("torch", "torch_index_add"), default="torch_index_add")
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sandbox", type=parse_json_object, default=None)
    parser.add_argument("--use-cache", type=Path, default=None)
    parser.add_argument("--rerun-review", action="store_true")
    parser.add_argument("--no-timestamp", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    visible_gpus = validate_visible_gpus(os.environ.get("CUDA_VISIBLE_DEVICES"))
    if args.eval_batch_size <= 0:
        raise ValueError("eval-batch-size must be positive.")
    profile_path = args.profile.expanduser().resolve()
    channel_path = args.channel_cache.expanduser().resolve()
    profile_hash = file_sha256(profile_path)
    channel_hash = file_sha256(channel_path)
    profile, _, _, _ = validate_static_profile_artifacts(
        model_path=args.model_path,
        profile_path=profile_path,
        channel_cache_path=channel_path,
        expected_profile_file_sha256=args.expected_profile_file_sha256,
        expected_channel_file_sha256=args.expected_channel_file_sha256,
    )
    stats_path = (
        args.stats_path.expanduser().resolve()
        if args.stats_path is not None
        else args.work_dir.expanduser().resolve() / "static_profile_runtime.json"
    )
    generation_config = {
        "do_sample": False,
        "temperature": 0.0,
        "max_tokens": 4096,
        **args.generation_config,
    }
    dataset_args = normalize_dataset_args(list(args.datasets), args.dataset_args, getattr(args, "dataset_limits", {}))
    task_kwargs = {
        "model": str(Path(args.model_path).expanduser().resolve()),
        "model_id": args.model_id,
        "eval_type": "static_expert_profile",
        "datasets": list(args.datasets),
        "dataset_args": dataset_args,
        "dataset_hub": args.dataset_hub,
        "generation_config": generation_config,
        "model_args": {
            "model_path": str(Path(args.model_path).expanduser().resolve()),
            "model_family": args.model_family,
            "profile_path": str(profile_path),
            "channel_cache_path": str(channel_path),
            "expected_profile_file_sha256": profile_hash,
            "expected_channel_file_sha256": channel_hash,
            "device_map": "auto" if len(visible_gpus) > 1 else {"": "cuda:0"},
            "correction_mode": args.correction_mode,
            "max_correction_ratio": args.max_correction_ratio,
            "moe_backend": args.moe_backend,
            "enable_thinking": args.enable_thinking,
            "stats_path": str(stats_path),
        },
        "eval_batch_size": args.eval_batch_size,
        "limit": args.limit,
        "seed": args.seed,
        "work_dir": str(args.work_dir.expanduser().resolve()),
        "use_cache": None if args.use_cache is None else str(args.use_cache.expanduser().resolve()),
        "rerun_review": args.rerun_review,
        "no_timestamp": args.no_timestamp,
    }
    if args.dataset_dir is not None:
        task_kwargs["dataset_dir"] = args.dataset_dir
    if args.sandbox is not None:
        task_kwargs["sandbox"] = args.sandbox
    args.work_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_identity": {
            "static_moe_prunning": source_tree_identity(
                Path(__file__).resolve().parents[2],
                pathspecs=("code/src", "code/scripts/run_evalscope_static_profile.py"),
            ),
            "evalscope": source_tree_identity(
                evalscope_source_root(),
                pathspecs=("evalscope",),
            ),
        },
        "cuda_visible_devices": visible_gpus,
        "profile_method": profile.get("method"),
        "profile_mode": profile.get("mode"),
        "profile_file_sha256": profile_hash,
        "expected_profile_file_sha256": args.expected_profile_file_sha256,
        "channel_file_sha256": channel_hash,
        "expected_channel_file_sha256": args.expected_channel_file_sha256,
        "task_config": task_kwargs,
        "test_metrics_used_for_profile": profile.get("test_metrics_used_for_profile"),
        "preflight_only": args.preflight_only,
    }
    manifest_path = args.work_dir / "evalscope_static_profile_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.preflight_only:
        print(manifest_path.resolve())
        return 0
    register_static_expert_profile_api()
    from evalscope import TaskConfig, run_task

    run_task(TaskConfig(**task_kwargs))
    print(manifest_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())