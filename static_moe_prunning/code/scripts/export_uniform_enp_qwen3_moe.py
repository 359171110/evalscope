from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a uniformly ENP-pruned Qwen3 MoE HF checkpoint.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--channel-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--retained-channels", type=int, required=True)
    parser.add_argument("--expected-protocol-name", required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expert_indices(name: str) -> tuple[int, int] | None:
    parts = name.split(".")
    if len(parts) < 7 or parts[0] != "model" or parts[1] != "layers" or parts[3:5] != ["mlp", "experts"]:
        return None
    return int(parts[2]), int(parts[5])


def prune_tensor(name: str, tensor: torch.Tensor, retained: torch.Tensor) -> torch.Tensor:
    if name.endswith(("gate_proj.weight", "up_proj.weight")):
        return tensor.index_select(0, retained)
    if name.endswith("down_proj.weight"):
        return tensor.index_select(1, retained)
    return tensor


def validate_enp_artifacts(
    profile: dict[str, object],
    channel_cache: dict[str, object],
    *,
    retained_channels: int,
    expected_protocol_name: str,
    channel_cache_path: Path,
) -> torch.Tensor:
    if profile.get("method") != "enp" or profile.get("mode") != "uniform_expert_neuron_pruning":
        raise ValueError("profile must be a uniform ENP profile.")
    if profile.get("profile_construction") != "calibrated" or profile.get("test_metrics_used_for_profile") is not False:
        raise ValueError("ENP profile must be train-calibrated without test metrics.")
    calibration = profile.get("cache_provenance", {}).get("calibration", {})
    if calibration.get("protocol_name") != expected_protocol_name:
        raise ValueError("profile calibration protocol does not match --expected-protocol-name.")
    channel = profile.get("cache_provenance", {}).get("channel", {})
    if channel.get("sha256") != file_sha256(channel_cache_path):
        raise ValueError("profile channel-cache SHA256 does not match the supplied channel cache.")
    if channel_cache.get("purpose") != "enp_tenp_signed_projection_channel_ranking":
        raise ValueError("channel cache must contain ENP/TENP signed-projection rankings.")
    if channel_cache.get("split") != "train" or channel_cache.get("test_metrics_used") is not False:
        raise ValueError("channel cache must be train-only and independent of test metrics.")
    block_size = int(profile["channel_block_size"])
    if retained_channels % block_size:
        raise ValueError("retained_channels must align with the profile channel block size.")
    expected_width = retained_channels // block_size
    widths = profile["profile_widths"].to(torch.long)
    if not torch.all(widths == expected_width):
        raise ValueError("ENP profile must retain the same requested width for every routed expert.")
    if profile.get("enp", {}).get("zero_token_policy") == "keep_full":
        raise ValueError("keep_full profiles cannot be exported as strict uniform ENP checkpoints.")
    return widths


def main() -> int:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    profile_path = args.profile.expanduser().resolve()
    channel_cache_path = args.channel_cache.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    profile = torch.load(profile_path, map_location="cpu", weights_only=True)
    channel_cache = torch.load(channel_cache_path, map_location="cpu", weights_only=True)
    retained_channels = int(args.retained_channels)
    validate_enp_artifacts(
        profile,
        channel_cache,
        retained_channels=retained_channels,
        expected_protocol_name=args.expected_protocol_name,
        channel_cache_path=channel_cache_path,
    )
    retained_by_expert = {
        (int(layer_id), expert_id): row[:retained_channels].to(torch.long)
        for layer_id, table in channel_cache["table"].items()
        for expert_id, row in enumerate(table["ranked_indices"])
    }

    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    source_moe_intermediate_size = int(config["moe_intermediate_size"])
    if not 0 < retained_channels < source_moe_intermediate_size:
        raise ValueError("retained_channels must be positive and smaller than the source MoE intermediate size.")

    index_path = model_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    total_size = 0
    shape_changes: dict[str, dict[str, list[int]]] = {}
    exported_shards: dict[str, str] = {}
    for shard_name in sorted(set(index["weight_map"].values())):
        tensors = {}
        with safe_open(model_path / shard_name, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                tensor = handle.get_tensor(name)
                source_shape = list(tensor.shape)
                ids = expert_indices(name)
                if ids is not None:
                    tensor = prune_tensor(name, tensor, retained_by_expert[ids])
                exported_shape = list(tensor.shape)
                if exported_shape != source_shape:
                    shape_changes[name] = {"source": source_shape, "exported": exported_shape}
                tensors[name] = tensor.contiguous()
                total_size += tensor.numel() * tensor.element_size()
        shard_path = output_dir / shard_name
        save_file(tensors, shard_path, metadata={"format": "pt"})
        exported_shards[shard_name] = file_sha256(shard_path)
        print(f"Exported {shard_name}", flush=True)

    for source in model_path.iterdir():
        if source.name.startswith("model-") and source.suffix == ".safetensors":
            continue
        if source.name in {"config.json", "model.safetensors.index.json"}:
            continue
        target = output_dir / source.name
        if source.is_file():
            shutil.copy2(source, target)
        elif source.is_dir():
            shutil.copytree(source, target)

    config["moe_intermediate_size"] = retained_channels
    exported_config_path = output_dir / "config.json"
    exported_config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    index.setdefault("metadata", {})["total_size"] = total_size
    exported_index_path = output_dir / "model.safetensors.index.json"
    exported_index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "method": "enp",
        "source_model": str(model_path),
        "source_config_sha256": file_sha256(model_path / "config.json"),
        "source_weight_index_sha256": file_sha256(index_path),
        "profile": str(profile_path),
        "profile_sha256": file_sha256(profile_path),
        "channel_cache": str(channel_cache_path),
        "channel_cache_sha256": file_sha256(channel_cache_path),
        "calibration_protocol_name": args.expected_protocol_name,
        "calibration_cache_sha256": profile["cache_provenance"]["calibration"]["sha256"],
        "calibration_input_ids_sha256": profile["cache_provenance"]["calibration"]["input_ids_sha256"],
        "export_script": str(Path(__file__).resolve()),
        "export_script_sha256": file_sha256(Path(__file__).resolve()),
        "retained_channels": retained_channels,
        "source_moe_intermediate_size": source_moe_intermediate_size,
        "routed_param_retention": profile["routed_param_retention"],
        "target_pruning_ratio": profile["target_pruning_ratio"],
        "exported_weight_bytes": total_size,
        "exported_config_sha256": file_sha256(exported_config_path),
        "exported_weight_index_sha256": file_sha256(exported_index_path),
        "exported_shards": exported_shards,
        "shape_changes": shape_changes,
        "validation_status": {
            "transformers_greedy_smoke": "pending",
            "vllm_health_and_chat": "pending",
            "transformers_vllm_consistency": "pending",
        },
    }
    (output_dir / "pruning_export_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())