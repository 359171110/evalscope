from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

from NAPS_v2.build_naps_v2_artifacts import file_sha256, load_weight_map
from NAPS_v2.channel_merge import apply_channel_merge_plan
from NAPS_v2.model_adapter import PurePseudoModelAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a padded homogeneous NAPS-v2 heterogeneous Mask checkpoint."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--channel-merge-plan", type=Path)
    return parser.parse_args()


def pad_rows(tensor: torch.Tensor, padded_width: int) -> torch.Tensor:
    if tensor.ndim != 2 or tensor.shape[0] > padded_width:
        raise ValueError("tensor rows exceed padded width")
    if tensor.shape[0] == padded_width:
        return tensor.contiguous()
    output = torch.zeros((padded_width, tensor.shape[1]), dtype=tensor.dtype)
    output[: tensor.shape[0]] = tensor
    return output


def pad_columns(tensor: torch.Tensor, padded_width: int) -> torch.Tensor:
    if tensor.ndim != 2 or tensor.shape[1] > padded_width:
        raise ValueError("tensor columns exceed padded width")
    if tensor.shape[1] == padded_width:
        return tensor.contiguous()
    output = torch.zeros((tensor.shape[0], padded_width), dtype=tensor.dtype)
    output[:, : tensor.shape[1]] = tensor
    return output


def swiglu_expert_output(
    hidden_states: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    activation: str = "silu",
) -> torch.Tensor:
    if hidden_states.ndim != 2 or gate.ndim != 2 or up.shape != gate.shape:
        raise ValueError("hidden states, gate, and up tensors have incompatible shapes")
    if down.ndim != 2 or down.shape[1] != gate.shape[0] or hidden_states.shape[1] != gate.shape[1]:
        raise ValueError("expert projections are not channel-aligned")
    gate_output = hidden_states.float() @ gate.float().transpose(0, 1)
    if activation == "silu":
        activated_gate = F.silu(gate_output)
    elif activation == "gelu_pytorch_tanh":
        activated_gate = F.gelu(gate_output, approximate="tanh")
    else:
        raise ValueError(f"Unsupported heterogeneous export activation: {activation!r}")
    expert_output = activated_gate * (hidden_states.float() @ up.float().transpose(0, 1))
    return expert_output @ down.float().transpose(0, 1)


def padded_swiglu_expert_output(
    hidden_states: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    retained: torch.Tensor,
    padded_width: int,
    activation: str = "silu",
) -> torch.Tensor:
    retained = retained.to(device=gate.device, dtype=torch.long)
    if retained.ndim != 1 or retained.numel() == 0 or retained.numel() > padded_width:
        raise ValueError("retained channels must be a non-empty one-dimensional subset of padded width")
    padded_gate = pad_rows(gate.index_select(0, retained), padded_width)
    padded_up = pad_rows(up.index_select(0, retained), padded_width)
    padded_down = pad_columns(down.index_select(1, retained), padded_width)
    return swiglu_expert_output(hidden_states, padded_gate, padded_up, padded_down, activation=activation)


def layer_id_from_name(name: str) -> int:
    marker = ".layers."
    if marker not in name:
        raise ValueError(f"Cannot parse layer from tensor name: {name}")
    return int(name.split(marker, 1)[1].split(".", 1)[0])


def expert_id_from_name(name: str) -> int:
    marker = ".experts."
    if marker not in name:
        raise ValueError(f"Cannot parse expert from tensor name: {name}")
    return int(name.split(marker, 1)[1].split(".", 1)[0])


def load_width_metadata(
    artifact_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], torch.Tensor, tuple[int, ...], int]:
    rankings = torch.load(artifact_dir / "rankings.pt", map_location="cpu", weights_only=True)
    profile = torch.load(artifact_dir / "profile.pt", map_location="cpu", weights_only=True)
    widths_by_layer = profile["profile_widths"].to(torch.long) * int(profile["channel_block_size"])
    width_options = tuple(int(value) for value in profile["width_options"].tolist())
    padded_width = int(profile["padded_intermediate_size"])
    if int(widths_by_layer.max().item()) > padded_width:
        raise ValueError("profile width exceeds padded width")
    if padded_width % int(profile["channel_block_size"]):
        raise ValueError("padded width is not block-aligned")
    return rankings, profile, widths_by_layer, width_options, padded_width


def export_method(profile: dict[str, Any]) -> str:
    method = str(profile.get("method", "naps_v2_heterogeneous_mask_padded"))
    return method if method.endswith("_padded") else f"{method}_padded"


def selected_indices(
    rankings: dict[str, Any],
    layer_id: int,
    expert_id: int,
    width: int,
) -> torch.Tensor:
    table = rankings["table"].get(layer_id, rankings["table"].get(str(layer_id)))
    if table is None:
        raise KeyError(f"Missing ranking table for layer {layer_id}")
    options = [int(value) for value in table["width_options"].tolist()]
    try:
        width_slot = options.index(int(width))
    except ValueError as error:
        raise ValueError(f"Width {width} is not present in the ranking table") from error
    ids = table["ranked_indices_by_width"][expert_id, width_slot, :width].to(torch.long)
    if not torch.equal(torch.unique(ids).sort().values, ids.sort().values):
        raise ValueError("selected ranking contains duplicate channels")
    return ids


def export_qwen3_separate_tensor(
    tensor: torch.Tensor,
    *,
    layer_id: int,
    expert_id: int,
    width: int,
    padded_width: int,
    rankings: dict[str, Any],
    down_projection: bool,
    merge_plan: dict[str, Any] | None = None,
) -> torch.Tensor:
    ids = selected_indices(rankings, layer_id, expert_id, width)
    selected = (
        apply_channel_merge_plan(tensor, ids, merge_plan)
        if down_projection and merge_plan is not None else tensor.index_select(1 if down_projection else 0, ids)
    )
    return pad_columns(selected, padded_width) if down_projection else pad_rows(selected, padded_width)


def export_qwen36_gate_up(
    tensor: torch.Tensor,
    *,
    layer_id: int,
    widths: torch.Tensor,
    padded_width: int,
    rankings: dict[str, Any],
) -> torch.Tensor:
    source_width = int(tensor.shape[1] // 2)
    outputs = []
    for expert_id, width_tensor in enumerate(widths):
        width = int(width_tensor.item())
        ids = selected_indices(rankings, layer_id, expert_id, width)
        gate = pad_rows(tensor[expert_id, :source_width].index_select(0, ids), padded_width)
        up = pad_rows(tensor[expert_id, source_width:].index_select(0, ids), padded_width)
        outputs.append(torch.cat((gate, up), dim=0))
    return torch.stack(outputs).contiguous()


def export_qwen36_down(
    tensor: torch.Tensor,
    *,
    layer_id: int,
    widths: torch.Tensor,
    padded_width: int,
    rankings: dict[str, Any],
    merge_plans: dict[int, dict[str, Any]] | None = None,
) -> torch.Tensor:
    outputs = []
    for expert_id, width_tensor in enumerate(widths):
        width = int(width_tensor.item())
        ids = selected_indices(rankings, layer_id, expert_id, width)
        selected = (
            apply_channel_merge_plan(tensor[expert_id], ids, merge_plans[expert_id])
            if merge_plans is not None and expert_id in merge_plans else tensor[expert_id].index_select(1, ids)
        )
        outputs.append(pad_columns(selected, padded_width))
    return torch.stack(outputs).contiguous()


def load_channel_merge_plan(
    path: Path | None,
    model_path: Path,
    artifact_dir: Path,
    widths_by_layer: torch.Tensor,
) -> tuple[dict[int, dict[int, dict[str, Any]]] | None, dict[str, Any] | None, Path | None]:
    if path is None:
        return None, None, None
    resolved = path.expanduser().resolve()
    payload = torch.load(resolved, map_location="cpu", weights_only=True)
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("CHANNEL merge plan must use schema version 1")
    if Path(payload["model_path"]).resolve() != model_path:
        raise ValueError("CHANNEL merge plan and requested model paths do not match")
    if Path(payload["artifact_dir"]).resolve() != artifact_dir:
        raise ValueError("CHANNEL merge plan and ranking artifact paths do not match")
    expected_hashes = {
        "rankings_sha256": file_sha256(artifact_dir / "rankings.pt"),
        "profile_sha256": file_sha256(artifact_dir / "profile.pt"),
    }
    for field, expected in expected_hashes.items():
        if payload.get(field) != expected:
            raise ValueError(f"CHANNEL merge plan {field} does not match the ranking artifact")
    capture_path = Path(payload["capture_path"]).expanduser().resolve()
    if payload.get("capture_sha256") != file_sha256(capture_path):
        raise ValueError("CHANNEL merge plan capture provenance does not match")
    if payload.get("holdout_used_for_acceptance") is not True:
        raise ValueError("CHANNEL merge plan must use an independent holdout acceptance gate")
    if payload.get("benchmark_metrics_used") is not False:
        raise ValueError("CHANNEL merge plan must not use benchmark metrics")
    layers = {
        int(layer_id): {int(expert_id): plan for expert_id, plan in experts.items()}
        for layer_id, experts in payload["layers"].items()
    }
    expected_layers, expected_experts = widths_by_layer.shape
    if sorted(layers) != list(range(expected_layers)):
        raise ValueError("CHANNEL merge plan layers do not match the profile")
    for layer_id, experts in layers.items():
        if sorted(experts) != list(range(expected_experts)):
            raise ValueError(f"CHANNEL merge plan experts do not match layer {layer_id}")
        for expert_id, plan in experts.items():
            if int(plan["retained_width"]) != int(widths_by_layer[layer_id, expert_id].item()):
                raise ValueError(f"CHANNEL merge plan width does not match layer {layer_id} expert {expert_id}")
    return layers, payload, resolved


def main() -> int:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    artifact_dir = args.artifact_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rankings, profile, widths_by_layer, width_options, padded_width = load_width_metadata(artifact_dir)
    merge_layers, merge_payload, merge_path = load_channel_merge_plan(
        args.channel_merge_plan,
        model_path,
        artifact_dir,
        widths_by_layer,
    )
    method = export_method(profile)
    if merge_payload is not None:
        method = method.removesuffix("_padded") + "_sparse_merge_padded"
    weight_map = load_weight_map(model_path)
    adapter = PurePseudoModelAdapter.from_checkpoint(model_path, weight_map)
    source_config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    changed_expert_tensors = 0
    total_size = 0
    shape_changes: dict[str, Any] = {}

    for shard_name in sorted(set(weight_map.values())):
        tensors = {}
        with safe_open(model_path / shard_name, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                tensor = handle.get_tensor(name)
                changed = tensor
                if adapter.expert_gate_template is not None and ".experts." in name:
                    layer_id = layer_id_from_name(name)
                    expert_id = expert_id_from_name(name)
                    width = int(widths_by_layer[layer_id, expert_id].item())
                    if name.endswith("gate_proj.weight") or name.endswith("up_proj.weight"):
                        changed = export_qwen3_separate_tensor(
                            tensor,
                            layer_id=layer_id,
                            expert_id=expert_id,
                            width=width,
                            padded_width=padded_width,
                            rankings=rankings,
                            down_projection=False,
                        )
                    elif name.endswith("down_proj.weight"):
                        changed = export_qwen3_separate_tensor(
                            tensor,
                            layer_id=layer_id,
                            expert_id=expert_id,
                            width=width,
                            padded_width=padded_width,
                            rankings=rankings,
                            down_projection=True,
                            merge_plan=merge_layers[layer_id][expert_id] if merge_layers is not None else None,
                        )
                    else:
                        changed = tensor
                    if changed is not tensor:
                        changed_expert_tensors += 1
                elif (
                    adapter.expert_gate_template is None
                    and ".layers." in name
                    and name == adapter.expert_gate_up_name(layer_id_from_name(name))
                ):
                    layer_id = layer_id_from_name(name)
                    changed = export_qwen36_gate_up(
                        tensor,
                        layer_id=layer_id,
                        widths=widths_by_layer[layer_id],
                        padded_width=padded_width,
                        rankings=rankings,
                    )
                    changed_expert_tensors += 1
                elif (
                    adapter.expert_gate_template is None
                    and ".layers." in name
                    and name == adapter.expert_down_name(layer_id_from_name(name))
                ):
                    layer_id = layer_id_from_name(name)
                    changed = export_qwen36_down(
                        tensor,
                        layer_id=layer_id,
                        widths=widths_by_layer[layer_id],
                        padded_width=padded_width,
                        rankings=rankings,
                        merge_plans=merge_layers[layer_id] if merge_layers is not None else None,
                    )
                    changed_expert_tensors += 1
                tensors[name] = changed.contiguous()
                total_size += changed.numel() * changed.element_size()
                if list(changed.shape) != list(tensor.shape):
                    shape_changes[name] = {"source": list(tensor.shape), "exported": list(changed.shape)}
        save_file(tensors, output_dir / shard_name, metadata={"format": "pt"})

    expected_changed = adapter.num_layers * (
        3 * adapter.num_experts if adapter.expert_gate_template is not None else 2
    )
    if changed_expert_tensors != expected_changed:
        raise RuntimeError(f"Changed {changed_expert_tensors} expert tensors, expected {expected_changed}")
    for source in model_path.iterdir():
        if source.suffix == ".safetensors" or source.name in {".git", "config.json", "model.safetensors.index.json"}:
            continue
        if source.name.startswith("."):
            continue
        target = output_dir / source.name
        shutil.copytree(source, target) if source.is_dir() else shutil.copy2(source, target)
    text_config = source_config.get("text_config", source_config)
    text_config["moe_intermediate_size"] = padded_width
    (output_dir / "config.json").write_text(json.dumps(source_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    index = json.loads((model_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    index.setdefault("metadata", {})["total_size"] = total_size
    (output_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "heterogeneous_expert_config.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "method": method,
                "source_model": str(model_path),
                "artifact_dir": str(artifact_dir),
                "source_intermediate_size": int(profile["source_intermediate_size"]),
                "padded_intermediate_size": padded_width,
                "width_options": width_options,
                "widths_by_layer": widths_by_layer.tolist(),
                "padding_value": 0.0,
                "ranking_schema_version": int(rankings.get("schema_version", -1)),
                "profile_schema_version": int(profile.get("schema_version", -1)),
                "calibration": profile.get("calibration"),
                "channel_merge": merge_payload.get("summary") if merge_payload is not None else None,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 3,
        "method": method,
        "source_model": str(model_path),
        "artifact_dir": str(artifact_dir),
        "source_intermediate_size": int(profile["source_intermediate_size"]),
        "padded_intermediate_size": padded_width,
        "width_options": width_options,
        "widths_by_layer": widths_by_layer.tolist(),
        "changed_expert_tensors": changed_expert_tensors,
        "shape_changes": shape_changes,
        "padding_is_structural_zero": True,
        "ranking_schema_version": int(rankings.get("schema_version", -1)),
        "profile_schema_version": int(profile.get("schema_version", -1)),
        "capture_path": profile.get("capture_path"),
        "capture_sha256": profile.get("capture_sha256"),
        "calibration": profile.get("calibration"),
        "model_provenance": profile.get("model_provenance"),
        "channel_merge_plan_path": str(merge_path) if merge_path is not None else None,
        "channel_merge_plan_sha256": (
            file_sha256(merge_path) if merge_path is not None else None
        ),
        "channel_merge": merge_payload.get("summary") if merge_payload is not None else None,
    }
    (output_dir / "pruning_export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
