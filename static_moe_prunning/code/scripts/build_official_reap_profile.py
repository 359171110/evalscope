from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

from src.calibration_data import calibration_batches_from_payload
from src.channel_runtime import channel_table_from_payload
from src.model_loading import load_supported_moe
from src.model_structure import iter_moe_layer_bindings
from src.reap_bridge import build_reap_profile_payload
from src.source_identity import source_tree_identity


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an official REAP whole-expert profile from a shared token artifact."
    )
    parser.add_argument("--official-reap-root", type=Path, required=True)
    parser.add_argument("--official-reap-commit", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-family", default="qwen3")
    parser.add_argument("--calibration-cache", type=Path, required=True)
    parser.add_argument("--channel-cache", type=Path, required=True)
    parser.add_argument("--output-observer", type=Path, required=True)
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument("--experts-to-prune-per-layer", type=int, required=True)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--batch-group-size", type=int, default=8)
    parser.add_argument("--device-map", default="cpu")
    return parser.parse_args()


def _git_commit(repository: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_is_clean(repository: Path) -> bool:
    status = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain=v1"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return not bool(status.strip())


def _load_official_observer_classes(repository: Path):
    source_root = repository / "src"
    if not source_root.is_dir():
        raise FileNotFoundError(f"official REAP source directory does not exist: {source_root}")
    sys.path.insert(0, str(source_root))
    from reap.layerwise_observer import LayerwiseMoEObserver
    from reap.observer import OBSERVER_CONFIG_REGISTRY

    return LayerwiseMoEObserver, OBSERVER_CONFIG_REGISTRY


def main() -> int:
    args = parse_args()
    reap_root = args.official_reap_root.expanduser().resolve()
    actual_commit = _git_commit(reap_root)
    if actual_commit != args.official_reap_commit:
        raise ValueError(
            f"official REAP commit {actual_commit} does not match frozen commit "
            f"{args.official_reap_commit}."
        )
    if not _git_is_clean(reap_root):
        raise ValueError("official REAP checkout must be clean.")
    if args.sequence_length <= 0:
        raise ValueError("official REAP requires a positive sequence length.")
    if args.batch_group_size <= 0:
        raise ValueError("batch-group-size must be positive.")

    calibration_path = args.calibration_cache.expanduser().resolve()
    channel_path = args.channel_cache.expanduser().resolve()
    calibration_payload = torch.load(calibration_path, map_location="cpu", weights_only=True)
    if int(calibration_payload.get("sequence_length", -1)) != int(args.sequence_length):
        raise ValueError("calibration cache sequence length does not match REAP arguments.")
    calibration_batches = calibration_batches_from_payload(
        calibration_payload,
        required_sequence_length=args.sequence_length,
        model_path=args.model_path,
        require_identity=True,
    )
    channel_payload = torch.load(channel_path, map_location="cpu", weights_only=True)
    if channel_payload.get("split") != "train":
        raise ValueError("REAP runtime channel topology must come from a train-only cache.")
    if int(channel_payload.get("sequence_length", -1)) != int(args.sequence_length):
        raise ValueError("channel cache sequence length does not match REAP calibration.")
    channel_table = channel_table_from_payload(channel_payload["table"])
    block_counts = {int(layer.block_sizes.numel()) for layer in channel_table.values()}
    if len(block_counts) != 1:
        raise ValueError("REAP profile requires a uniform channel block count.")
    num_blocks = next(iter(block_counts))

    model, _ = load_supported_moe(
        args.model_path,
        device_map=args.device_map,
        model_family=args.model_family,
    )
    bindings = list(iter_moe_layer_bindings(model))
    if not bindings:
        raise ValueError("No supported MoE layers found for official REAP observation.")
    top_k_values = {int(binding.top_k) for binding in bindings}
    if len(top_k_values) != 1:
        raise ValueError("official REAP profile requires a uniform top-k.")
    top_k = next(iter(top_k_values))

    LayerwiseMoEObserver, observer_registry = _load_official_observer_classes(reap_root)
    model_class_name = model.__class__.__name__
    hook_config_class = observer_registry.get(model_class_name)
    if hook_config_class is None:
        raise ValueError(
            f"official REAP commit {actual_commit} has no observer config for "
            f"{model_class_name}."
        )
    hook_config = hook_config_class(
        renormalize_router_weights=True,
        record_pruning_metrics_only=True,
    )
    observer = LayerwiseMoEObserver(model=model, hook_config=hook_config)
    observer_data = observer.record_all_blocks(
        data_batches=calibration_batches,
        batch_group_size=args.batch_group_size,
    )

    observer_payload = {
        "schema_version": 1,
        "method": "official_reap_observer",
        "official_reap_commit": actual_commit,
        "official_reap_checkout_clean": True,
        "model_path": str(Path(args.model_path).expanduser().resolve()),
        "calibration_cache_path": str(calibration_path),
        "calibration_cache_sha256": file_sha256(calibration_path),
        "input_ids_sha256": calibration_payload.get("input_ids_sha256"),
        "sequence_length": args.sequence_length,
        "calibration_sequences": len(calibration_batches),
        "calibration_tokens": sum(
            int(batch["attention_mask"].sum().item()) for batch in calibration_batches
        ),
        "renormalize_router_weights": True,
        "record_pruning_metrics_only": True,
        "bridge_source_identity": source_tree_identity(
            Path(__file__).resolve().parents[2],
            pathspecs=("code/src", "code/scripts/build_official_reap_profile.py"),
        ),
        "observer_data": observer_data,
    }
    args.output_observer.parent.mkdir(parents=True, exist_ok=True)
    torch.save(observer_payload, args.output_observer)
    observer_hash = file_sha256(args.output_observer)

    profile = build_reap_profile_payload(
        observer_data=observer_data,
        model_path=str(Path(args.model_path).expanduser().resolve()),
        calibration_payload=calibration_payload,
        calibration_file_sha256=file_sha256(calibration_path),
        channel_file_sha256=file_sha256(channel_path),
        official_reap_commit=actual_commit,
        num_blocks=num_blocks,
        experts_to_prune_per_layer=args.experts_to_prune_per_layer,
        top_k=top_k,
        renormalize_router_weights=True,
    )
    profile["cache_provenance"]["observer"] = {
        "path": str(args.output_observer.resolve()),
        "sha256": observer_hash,
        "official_reap_commit": actual_commit,
    }
    profile["bridge_source_identity"] = observer_payload["bridge_source_identity"]
    profile["cache_provenance"]["channel"]["path"] = str(channel_path)
    args.output_profile.parent.mkdir(parents=True, exist_ok=True)
    torch.save(profile, args.output_profile)
    summary = {
        key: value
        for key, value in profile.items()
        if key not in {"profile_widths", "retained_expert_mask", "official_reap_saliency"}
    }
    summary["profile_file_sha256"] = file_sha256(args.output_profile)
    summary["retained_expert_ids_by_layer"] = {
        str(layer_id): torch.nonzero(profile["retained_expert_mask"][row], as_tuple=False)
        .flatten()
        .tolist()
        for row, layer_id in enumerate(profile["layer_ids"])
    }
    args.output_profile.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(args.output_observer.resolve())
    print(args.output_profile.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())