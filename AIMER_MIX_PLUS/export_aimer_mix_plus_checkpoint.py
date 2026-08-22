from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from AIMER_Mix.export_aimer_mix_checkpoint import (
    fused_shared_expert_width,
    patch_deepseek_remote_code,
    prune_routed_tensor,
    retained_table,
)
from AIMER_Mix.mix_core import file_sha256, validate_rankings
from AIMER_Mix.model_adapter import AIMERMixModelAdapter
from static_moe_prunning.code.src.static_expert_pruning import validate_static_profile_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a uniformly pruned AIMER-Mix-Plus checkpoint.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--channel-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    profile_path = args.profile.expanduser().resolve()
    channel_path = args.channel_cache.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = torch.load(profile_path, map_location="cpu", weights_only=True)
    cache = torch.load(channel_path, map_location="cpu", weights_only=True)
    validate_static_profile_payload(profile)
    if profile.get("method") != "aimer_mix_plus" or cache.get("purpose") != "aimer_mix_plus_ranking":
        raise ValueError("Expected AIMER-Mix-Plus profile and channel ranking cache")
    if profile["cache_provenance"]["channel"]["sha256"] != file_sha256(channel_path):
        raise ValueError("AIMER-Mix-Plus channel cache SHA256 does not match the profile")

    index_path = model_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = {str(name): str(shard) for name, shard in index["weight_map"].items()}
    adapter = AIMERMixModelAdapter.from_checkpoint(model_path, weight_map)
    architecture = adapter.architecture
    if (
        Path(str(profile["model_path"])).expanduser().resolve() != model_path
        or Path(str(cache["model_path"])).expanduser().resolve() != model_path
    ):
        raise ValueError("AIMER-Mix-Plus artifacts were built for a different model path")
    provenance = cache.get("model_provenance", {})
    if provenance.get("config_sha256") != file_sha256(model_path / "config.json"):
        raise ValueError("Checkpoint config changed after AMP ranking construction")
    if provenance.get("weight_index_sha256") != file_sha256(index_path):
        raise ValueError("Checkpoint weight index changed after AMP ranking construction")
    widths = profile["profile_widths"].long()
    if not bool((widths == widths.flatten()[0]).all()):
        raise ValueError("A standard HF checkpoint requires one uniform expert width")
    retained_channels = int(widths.flatten()[0].item()) * int(profile["channel_block_size"])
    if int(cache.get("retained_channels", -1)) != retained_channels:
        raise ValueError("AMP cache retained width does not match the profile")
    architecture.validate_width(retained_channels)
    validate_rankings(
        cache["table"],
        len(architecture.moe_layer_ids()),
        architecture.num_experts,
        architecture.intermediate_size,
        layer_ids=architecture.moe_layer_ids(),
    )
    retained = retained_table(cache, retained_channels)

    changed_tensors = 0
    shape_changes: dict[str, Any] = {}
    total_size = 0
    for shard_name in sorted(set(weight_map.values())):
        tensors = {}
        with safe_open(model_path / shard_name, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                source = handle.get_tensor(name)
                changed, did_change = prune_routed_tensor(name, source, adapter, retained)
                if did_change:
                    changed_tensors += 1
                tensors[name] = changed.contiguous()
                total_size += changed.numel() * changed.element_size()
                if list(changed.shape) != list(source.shape):
                    shape_changes[name] = {"source": list(source.shape), "exported": list(changed.shape)}
        save_file(tensors, output_dir / shard_name, metadata={"format": "pt"})
        print(f"exported_shard={shard_name}", flush=True)
    tensors_per_layer = 3 * architecture.num_experts if architecture.tensor_codec == "separate" else 2
    expected_changed = len(architecture.moe_layer_ids()) * tensors_per_layer
    if changed_tensors != expected_changed:
        raise RuntimeError(f"Changed {changed_tensors} routed tensors, expected {expected_changed}")

    for source in model_path.iterdir():
        if source.suffix == ".safetensors" or source.name in {
            "config.json",
            "model.safetensors.index.json",
            ".git",
            ".cache",
        }:
            continue
        target = output_dir / source.name
        shutil.copytree(source, target) if source.is_dir() else shutil.copy2(source, target)
    if architecture.model_family == "deepseek_v2":
        patch_deepseek_remote_code(output_dir)

    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    text_config = config.get("text_config", config)
    shared_width = fused_shared_expert_width(text_config)
    text_config["moe_intermediate_size"] = retained_channels
    if architecture.model_family == "deepseek_v2" and shared_width is not None:
        text_config["shared_expert_intermediate_size"] = shared_width
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    index.setdefault("metadata", {})["total_size"] = total_size
    (output_dir / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "method": "aimer_mix_plus",
        "model_family": architecture.model_family,
        "source_model": str(model_path),
        "source_config_sha256": file_sha256(model_path / "config.json"),
        "source_weight_index_sha256": file_sha256(index_path),
        "profile": str(profile_path),
        "profile_sha256": file_sha256(profile_path),
        "channel_cache": str(channel_path),
        "channel_cache_sha256": file_sha256(channel_path),
        "retained_channels": retained_channels,
        "source_expert_width": architecture.intermediate_size,
        "exported_moe_intermediate_size": retained_channels,
        "exported_shared_expert_intermediate_size": shared_width,
        "export_layout": "slice_uniform_width",
        "actual_structural_pruning_ratio": 1.0 - retained_channels / architecture.intermediate_size,
        "pseudo_sources": cache["aimer_mix_plus"]["sources"],
        "fusion_config": cache["aimer_mix_plus"]["fusion_config"],
        "diagnostic_summary": cache["aimer_mix_plus"]["diagnostic_summary"],
        "changed_routed_expert_tensors": changed_tensors,
        "exported_weight_bytes": total_size,
        "shape_changes": shape_changes,
        "preserved_scope": "routers, shared experts, dense MLPs, multimodal modules, and auxiliary/MTP tensors",
        "validation_status": {
            "transformers_greedy_smoke": "pending",
            "vllm_health_and_chat": "pending",
            "downstream": "pending",
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