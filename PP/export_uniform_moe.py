from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from PP.pure_pseudo_model_adapter import PurePseudoModelAdapter


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a uniformly Pure-Pseudo-pruned MoE checkpoint.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--channel-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--retained-channels", type=int, required=True)
    return parser.parse_args()


def _config_with_text(config: dict) -> dict:
    return config.get("text_config", config)


def _expert_tensor_kind(name: str, adapter: PurePseudoModelAdapter) -> str | None:
    for layer_id in range(adapter.num_layers):
        if name == adapter.expert_gate_up_name(layer_id):
            return "gate_up"
        if name == adapter.expert_down_name(layer_id):
            return "down"
    return None


def _prune_expert_tensor(
    name: str,
    tensor: torch.Tensor,
    adapter: PurePseudoModelAdapter,
    retained_by_layer_expert: dict[tuple[int, int], torch.Tensor],
) -> torch.Tensor:
    kind = _expert_tensor_kind(name, adapter)
    if kind is None:
        return tensor
    parts = name.split(".")
    layer_id = next(int(part) for index, part in enumerate(parts) if part == "layers" and index + 1 < len(parts) for part in [parts[index + 1]])
    if tensor.ndim != 3:
        raise ValueError(f"Expected fused expert tensor to be rank 3: {name}, shape={tuple(tensor.shape)}")
    output = []
    for expert_id in range(tensor.shape[0]):
        retained = retained_by_layer_expert[(layer_id, expert_id)]
        if kind == "gate_up":
            indices = torch.cat((retained, retained + adapter.intermediate_size), dim=0)
            output.append(tensor[expert_id].index_select(0, indices))
        else:
            output.append(tensor[expert_id].index_select(1, retained))
    return torch.stack(output, dim=0)


def main() -> int:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = args.profile.expanduser().resolve()
    cache_path = args.channel_cache.expanduser().resolve()
    profile = torch.load(profile_path, map_location="cpu", weights_only=True)
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    profile_method = str(profile.get("method", ""))
    cache_purpose = str(cache.get("purpose", ""))
    supported_method = profile_method in {
        "pure_pseudo",
        "random",
        "enp",
        "aimer_channel",
        "aimer_gauge_balanced",
        "shape_aimer",
        "stable_concat_aimer",
    } or profile_method.endswith("_pp")
    supported_cache = cache_purpose in {
        "pure_pseudo_channel_ranking",
        "random_channel_ranking",
        "enp_tenp_signed_projection_channel_ranking",
        "aimer_weight_only_channel_ranking",
    } or "pseudo_protection" in cache
    if not supported_method or not supported_cache:
        raise ValueError("profile and channel cache must be a supported fixed-width or PP-composed artifact pair.")
    retained_channels = int(args.retained_channels)
    block_size = int(profile["channel_block_size"])
    if retained_channels % block_size:
        raise ValueError("retained_channels must align with the profile channel block size.")
    widths = profile["profile_widths"].to(torch.long)
    if not torch.all(widths == retained_channels // block_size):
        raise ValueError("profile widths do not match retained_channels.")
    index_path = model_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = {str(name): str(shard) for name, shard in index["weight_map"].items()}
    adapter = PurePseudoModelAdapter.from_checkpoint(model_path, weight_map)
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    text_config = _config_with_text(config)
    if not 0 < retained_channels < adapter.intermediate_size:
        raise ValueError("retained_channels must be positive and smaller than the source expert width.")
    retained_by_layer_expert = {
        (int(layer_id), expert_id): row[:retained_channels].to(torch.long)
        for layer_id, table in cache["table"].items()
        for expert_id, row in enumerate(table["ranked_indices"])
    }
    total_size = 0
    shape_changes = {}
    for shard_name in sorted(set(weight_map.values())):
        tensors = {}
        with safe_open(model_path / shard_name, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                tensor = handle.get_tensor(name)
                pruned = _prune_expert_tensor(name, tensor, adapter, retained_by_layer_expert)
                if list(pruned.shape) != list(tensor.shape):
                    shape_changes[name] = {"source": list(tensor.shape), "exported": list(pruned.shape)}
                tensors[name] = pruned.contiguous()
                total_size += pruned.numel() * pruned.element_size()
        save_file(tensors, output_dir / shard_name, metadata={"format": "pt"})
    for source in model_path.iterdir():
        if source.name.endswith(".safetensors") or source.name in {"config.json", "model.safetensors.index.json"}:
            continue
        target = output_dir / source.name
        if source.is_file():
            shutil.copy2(source, target)
        elif source.is_dir():
            shutil.copytree(source, target)
    text_config["moe_intermediate_size"] = retained_channels
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    index.setdefault("metadata", {})["total_size"] = total_size
    (output_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "method": str(profile_method),
        "model_family": adapter.model_family,
        "source_model": str(model_path),
        "profile": str(profile_path),
        "channel_cache": str(cache_path),
        "retained_channels": retained_channels,
        "source_expert_width": adapter.intermediate_size,
        "exported_weight_bytes": total_size,
        "shape_changes": shape_changes,
        "validation_status": {"transformers_greedy_smoke": "pending", "vllm_health_and_chat": "pending"},
        "source_config_sha256": file_sha256(model_path / "config.json"),
        "source_weight_index_sha256": file_sha256(index_path),
    }
    (output_dir / "pruning_export_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
