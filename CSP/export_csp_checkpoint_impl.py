from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from CSP.csp_core import file_sha256, validate_rankings
from CSP.model_adapter import CSPModelAdapter
from static_moe_prunning.code.src.static_expert_pruning import validate_static_profile_payload

_SHARED_WIDTH_PRODUCT = "config.moe_intermediate_size * config.n_shared_experts"
_SHARED_WIDTH_BLOCK = (
    "            shared_width = int(getattr(config, \"shared_expert_intermediate_size\", 0) or 0)\n"
    "            intermediate_size = shared_width or (\n"
    "                config.moe_intermediate_size * config.n_shared_experts)\n"
)


def layer_id_from_name(name: str) -> int:
    """Extract the decoder layer index from a tensor name."""

    marker = ".layers."
    if marker not in name:
        raise ValueError(f"Cannot parse layer from tensor name: {name}")
    return int(name.split(marker, 1)[1].split(".", 1)[0])


def expert_id_from_name(name: str) -> int:
    """Extract a physical expert index from a separate expert tensor name."""

    marker = ".experts."
    if marker not in name:
        raise ValueError(f"Cannot parse expert from tensor name: {name}")
    return int(name.split(marker, 1)[1].split(".", 1)[0])


def retained_table(
    cache: dict[str, Any], profile: dict[str, Any]
) -> dict[tuple[int, int], torch.Tensor]:
    """Materialize each expert's logical retained CSP prefix."""

    widths = profile["profile_widths"].to(dtype=torch.long)
    layer_ids = [int(layer_id) for layer_id in profile["layer_ids"]]
    block_size = int(profile["channel_block_size"])
    return {
        (layer_id, expert_id): cache["table"][layer_id]["ranked_indices"][expert_id, :int(widths[row, expert_id]) * block_size].long()
        for row, layer_id in enumerate(layer_ids)
        for expert_id in range(int(profile["num_experts"]))
    }


def pad_rows(source: torch.Tensor, width: int) -> torch.Tensor:
    """Pad the first tensor dimension with exact zeros."""

    if int(source.shape[0]) > int(width):
        raise ValueError("source rows exceed the requested padded width.")
    if int(source.shape[0]) == int(width):
        return source
    return torch.cat((source, torch.zeros((int(width) - int(source.shape[0]), *source.shape[1:]), dtype=source.dtype)), dim=0)


def pad_columns(source: torch.Tensor, width: int) -> torch.Tensor:
    """Pad the last tensor dimension with exact zeros."""

    if int(source.shape[-1]) > int(width):
        raise ValueError("source columns exceed the requested padded width.")
    if int(source.shape[-1]) == int(width):
        return source
    shape = (*source.shape[:-1], int(width) - int(source.shape[-1]))
    return torch.cat((source, torch.zeros(shape, dtype=source.dtype)), dim=-1)


def fused_shared_expert_width(text_config: dict[str, Any]) -> int | None:
    """Return the source DeepSeek fused shared-expert width, if applicable."""

    n_shared = text_config.get("n_shared_experts")
    if n_shared is None:
        return None
    return int(text_config["moe_intermediate_size"]) * int(n_shared)


def patch_deepseek_remote_code(output_dir: Path) -> None:
    """Allow copied DeepSeek remote code to distinguish routed and shared widths."""

    config_py = output_dir / "configuration_deepseek.py"
    if config_py.is_file():
        text = config_py.read_text(encoding="utf-8")
        if "self.shared_expert_intermediate_size" not in text:
            old_arg = "        moe_intermediate_size = 1407,\n"
            new_arg = "        moe_intermediate_size = 1407,\n        shared_expert_intermediate_size = None,\n"
            old_assign = "        self.moe_intermediate_size = moe_intermediate_size\n"
            new_assign = (
                "        self.moe_intermediate_size = moe_intermediate_size\n"
                "        self.shared_expert_intermediate_size = shared_expert_intermediate_size\n"
            )
            if old_arg not in text or old_assign not in text:
                raise RuntimeError(f"Cannot patch DeepSeek config class: {config_py}")
            text = text.replace(old_arg, new_arg, 1).replace(old_assign, new_assign, 1)
            config_py.write_text(text, encoding="utf-8")
    modeling_py = output_dir / "modeling_deepseek.py"
    if modeling_py.is_file():
        text = modeling_py.read_text(encoding="utf-8")
        if _SHARED_WIDTH_PRODUCT in text:
            text = text.replace(
                f"            intermediate_size = {_SHARED_WIDTH_PRODUCT}\n",
                _SHARED_WIDTH_BLOCK,
                1,
            )
            modeling_py.write_text(text, encoding="utf-8")
        elif "shared_expert_intermediate_size" not in text:
            raise RuntimeError(f"Cannot patch DeepSeek shared MLP width: {modeling_py}")


def prune_routed_tensor(
    name: str,
    source: torch.Tensor,
    adapter: CSPModelAdapter,
    retained: dict[tuple[int, int], torch.Tensor],
    padded_width: int | None = None,
) -> tuple[torch.Tensor, bool]:
    """Slice routed gate/up rows and down columns; preserve every other tensor."""

    architecture = adapter.architecture
    moe_layers = set(architecture.moe_layer_ids())
    if architecture.tensor_codec == "packed":
        if ".layers." not in name:
            return source, False
        try:
            layer_id = layer_id_from_name(name)
        except ValueError:
            return source, False
        if layer_id not in moe_layers:
            return source, False
        if name == adapter.gate_up_name(layer_id):
            selected = []
            for expert_id in range(architecture.num_experts):
                indices = retained[(layer_id, expert_id)]
                packed_indices = torch.cat((indices, indices + architecture.intermediate_size))
                expert = source[expert_id].index_select(0, packed_indices)
                selected.append(pad_rows(expert, 2 * int(padded_width)) if padded_width is not None else expert)
            return torch.stack(selected), True
        if name == adapter.down_name(layer_id):
            selected = [source[expert_id].index_select(1, retained[(layer_id, expert_id)]) for expert_id in range(architecture.num_experts)]
            if padded_width is not None:
                selected = [pad_columns(expert, int(padded_width)) for expert in selected]
            return torch.stack(selected), True
        return source, False

    if ".experts." not in name:
        return source, False
    try:
        layer_id = layer_id_from_name(name)
        expert_id = expert_id_from_name(name)
    except ValueError:
        return source, False
    if layer_id not in moe_layers or not 0 <= expert_id < architecture.num_experts:
        return source, False
    indices = retained[(layer_id, expert_id)]
    if name in {adapter.gate_name(layer_id, expert_id), adapter.up_name(layer_id, expert_id)}:
        selected = source.index_select(0, indices)
        return pad_rows(selected, padded_width) if padded_width is not None else selected, True
    if name == adapter.down_name(layer_id, expert_id):
        selected = source.index_select(1, indices)
        return pad_columns(selected, padded_width) if padded_width is not None else selected, True
    return source, False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a CSP-pruned MoE checkpoint.")
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
    if profile.get("method") not in {"csp", "hsp"} or cache.get("purpose") != "csp_channel_ranking":
        raise ValueError("Expected a CSP/HSP profile and CSP channel ranking cache.")
    if profile["cache_provenance"]["channel"]["sha256"] != file_sha256(channel_path):
        raise ValueError("CSP channel cache SHA256 does not match the profile.")
    index_path = model_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = {str(name): str(shard) for name, shard in index["weight_map"].items()}
    adapter = CSPModelAdapter.from_checkpoint(model_path, weight_map)
    architecture = adapter.architecture
    if Path(str(profile["model_path"])).resolve() != model_path or Path(str(cache["model_path"])).resolve() != model_path:
        raise ValueError("CSP artifacts were built for a different model path.")
    model_provenance = cache.get("model_provenance", {})
    if model_provenance.get("config_sha256") != file_sha256(model_path / "config.json"):
        raise ValueError("Checkpoint config changed after CSP ranking construction.")
    if model_provenance.get("weight_index_sha256") != file_sha256(index_path):
        raise ValueError("Checkpoint weight index changed after CSP ranking construction.")
    widths = profile["profile_widths"].long()
    heterogeneous = profile.get("allocation_scope") == "per_layer_expert_sp_quantiles"
    if heterogeneous:
        if architecture.model_family not in {"qwen3", "qwen3.6", "gemma4", "deepseek_v2"}:
            raise ValueError("heterogeneous CSP export does not support this model family.")
        if cache.get("csp", {}).get("canonicalization") is not False:
            raise ValueError("heterogeneous CSP export requires raw Expert-SP and raw Channel-SP scoring.")
        padded_width = int(profile.get("padded_intermediate_size", 0))
        width_options = [int(width) for width in profile.get("width_options", [])]
        if not width_options or padded_width != max(width_options):
            raise ValueError("heterogeneous CSP profile must declare width_options and padded_intermediate_size.")
        if bool((widths * int(profile["channel_block_size"]) > padded_width).any()):
            raise ValueError("heterogeneous logical widths cannot exceed padded width.")
        retained_channels = int(profile.get("budget_reference_width", 0))
    else:
        if not bool((widths == widths.flatten()[0]).all()):
            raise ValueError("A standard HF checkpoint requires one uniform expert width.")
        retained_channels = int(widths.flatten()[0].item()) * int(profile["channel_block_size"])
        padded_width = None
        width_options = [retained_channels]
    architecture.validate_width(retained_channels)
    validate_rankings(cache["table"], len(architecture.moe_layer_ids()), architecture.num_experts, architecture.intermediate_size, layer_ids=architecture.moe_layer_ids())
    retained = retained_table(cache, profile)

    changed_tensors = 0
    shape_changes: dict[str, Any] = {}
    total_size = 0
    for shard_name in sorted(set(weight_map.values())):
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(model_path / shard_name, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                source = handle.get_tensor(name)
                changed, did_change = prune_routed_tensor(name, source, adapter, retained, padded_width)
                changed_tensors += int(did_change)
                tensors[name] = changed.contiguous()
                total_size += changed.numel() * changed.element_size()
                if list(changed.shape) != list(source.shape):
                    shape_changes[name] = {"source": list(source.shape), "exported": list(changed.shape)}
        save_file(tensors, output_dir / shard_name, metadata={"format": "pt"})
        print(f"exported_shard={shard_name}", flush=True)

    tensors_per_layer = 3 * architecture.num_experts if architecture.tensor_codec == "separate" else 2
    expected_changed = len(architecture.moe_layer_ids()) * tensors_per_layer
    if changed_tensors != expected_changed:
        raise RuntimeError(f"Changed {changed_tensors} routed-expert tensors, expected {expected_changed}.")
    for source in model_path.iterdir():
        if source.suffix == ".safetensors" or source.name in {"config.json", "model.safetensors.index.json", ".git", ".cache"}:
            continue
        target = output_dir / source.name
        shutil.copytree(source, target) if source.is_dir() else shutil.copy2(source, target)

    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    text_config = config.get("text_config", config)
    exported_shared_width = fused_shared_expert_width(text_config) if architecture.model_family == "deepseek_v2" else None
    exported_width = int(padded_width) if heterogeneous else retained_channels
    text_config["moe_intermediate_size"] = exported_width
    if exported_shared_width is not None:
        text_config["shared_expert_intermediate_size"] = exported_shared_width
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if architecture.model_family == "deepseek_v2":
        patch_deepseek_remote_code(output_dir)
    index.setdefault("metadata", {})["total_size"] = total_size
    (output_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "method": "hsp" if heterogeneous else "csp",
        "model_family": architecture.model_family,
        "source_model": str(model_path),
        "source_config_sha256": file_sha256(model_path / "config.json"),
        "source_weight_index_sha256": file_sha256(index_path),
        "profile": str(profile_path),
        "profile_sha256": file_sha256(profile_path),
        "channel_cache": str(channel_path),
        "channel_cache_sha256": file_sha256(channel_path),
        "score_mode": cache["score_mode"],
        "input_scale_mode": cache["csp"]["input_scale_mode"],
        "retained_channels": retained_channels,
        "logical_widths_by_layer": (widths * int(profile["channel_block_size"])).tolist() if heterogeneous else None,
        "width_options": width_options,
        "padded_intermediate_size": exported_width,
        "allocation_scope": profile.get("allocation_scope"),
        "source_expert_width": architecture.intermediate_size,
        "exported_moe_intermediate_size": exported_width,
        "exported_shared_expert_intermediate_size": exported_shared_width,
        "export_layout": "slice_uniform_width_padded" if heterogeneous else "slice_uniform_width",
        "actual_structural_pruning_ratio": 1.0 - retained_channels / architecture.intermediate_size,
        "changed_routed_expert_tensors": changed_tensors,
        "exported_weight_bytes": total_size,
        "shape_changes": shape_changes,
        "preserved_scope": "routers, shared experts, dense MLPs, multimodal modules, and auxiliary/MTP tensors",
        "validation_status": {
            "transformers_greedy_smoke": "pending",
            "vllm_health_and_chat": "pending",
            "full8_v1": "pending",
        },
    }
    (output_dir / "pruning_export_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
