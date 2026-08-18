from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from contextlib import ExitStack
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

from WICK.build_wick_profile import rms_norm_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a fixed-mask Qwen3 MoE checkpoint with output reconstruction.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--channel-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diagnostics-output", type=Path, required=True)
    parser.add_argument("--retained-channels", type=int, required=True)
    parser.add_argument("--ridge-relative", type=float, default=1.0e-4)
    parser.add_argument("--device", default="cpu")
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


def swiglu_responses(probes: torch.Tensor, gate_proj: torch.Tensor, up_proj: torch.Tensor) -> torch.Tensor:
    if probes.ndim != 2 or gate_proj.ndim != 2 or up_proj.shape != gate_proj.shape:
        raise ValueError("probes and aligned gate/up projections must be two-dimensional.")
    if int(probes.shape[1]) != int(gate_proj.shape[1]):
        raise ValueError("probe hidden size must match gate/up input size.")
    return F.silu(F.linear(probes.float(), gate_proj.float())) * F.linear(probes.float(), up_proj.float())


def reconstruct_down_proj(
    responses: torch.Tensor,
    down_proj: torch.Tensor,
    retained: torch.Tensor,
    *,
    ridge_relative: float,
    epsilon: float = 1.0e-12,
) -> tuple[torch.Tensor, dict[str, float]]:
    if responses.ndim != 2 or down_proj.ndim != 2 or int(responses.shape[1]) != int(down_proj.shape[1]):
        raise ValueError("responses and down projection must align on the channel dimension.")
    keep = retained.to(device=responses.device, dtype=torch.long).flatten()
    channel_count = int(responses.shape[1])
    if keep.numel() == 0 or bool((keep < 0).any()) or bool((keep >= channel_count).any()):
        raise ValueError("retained must contain valid channel IDs.")
    if int(torch.unique(keep).numel()) != int(keep.numel()):
        raise ValueError("retained must not contain duplicates.")
    if not 0.0 <= float(ridge_relative):
        raise ValueError("ridge_relative must be non-negative.")

    keep_mask = torch.zeros(channel_count, dtype=torch.bool, device=responses.device)
    keep_mask[keep] = True
    pruned = torch.arange(channel_count, device=responses.device)[~keep_mask]
    responses_keep = responses.index_select(1, keep).float()
    responses_pruned = responses.index_select(1, pruned).float()
    down = down_proj.to(device=responses.device, dtype=torch.float32)
    down_keep = down.index_select(1, keep)

    probe_count = int(responses.shape[0])
    gram = responses_keep @ responses_keep.transpose(0, 1)
    regularization = float(ridge_relative) * float(gram.diagonal().sum().item()) / float(probe_count)
    lost_output = responses_pruned @ down.index_select(1, pruned).transpose(0, 1)
    system = gram + regularization * torch.eye(probe_count, dtype=gram.dtype, device=gram.device)
    if regularization > 0.0:
        coefficients = torch.linalg.solve(system, lost_output)
    else:
        coefficients = torch.linalg.lstsq(system, lost_output).solution
    delta = responses_keep.transpose(0, 1) @ coefficients
    effective = down_keep + delta.transpose(0, 1)

    full_output = responses.float() @ down.transpose(0, 1)
    residual = lost_output - responses_keep @ delta
    output_norm = torch.linalg.vector_norm(full_output).clamp_min(float(epsilon))
    before = torch.linalg.vector_norm(lost_output) / output_norm
    after = torch.linalg.vector_norm(residual) / output_norm
    recovery = 1.0 - after / before.clamp_min(float(epsilon))
    compensation_ratio = torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(down_keep).clamp_min(float(epsilon))
    diagnostics = {
        "error_before": float(before.item()),
        "error_after": float(after.item()),
        "recovery_ratio": float(recovery.item()),
        "compensation_ratio": float(compensation_ratio.item()),
        "regularization": regularization,
    }
    return effective.to(dtype=down_proj.dtype, device=down_proj.device), diagnostics


def summarize_diagnostics(records: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    summary = {}
    for key in records[0]:
        values = torch.tensor([record[key] for record in records], dtype=torch.float64)
        summary[key] = {
            "mean": float(values.mean().item()),
            "p10": float(torch.quantile(values, 0.10).item()),
            "median": float(values.median().item()),
            "p90": float(torch.quantile(values, 0.90).item()),
            "min": float(values.min().item()),
            "max": float(values.max().item()),
        }
    return summary


def _first_tensor(handles: dict[str, object], weight_map: dict[str, str], names: list[str]) -> torch.Tensor:
    for name in names:
        shard = weight_map.get(name)
        if shard is not None:
            return handles[shard].get_tensor(name)
    raise KeyError(f"Missing checkpoint tensor; tried: {names}")


def main() -> int:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_path = args.channel_cache.expanduser().resolve()
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    retained_by_expert = {
        (int(layer_id), expert_id): row[: args.retained_channels].to(torch.long)
        for layer_id, table in cache["table"].items()
        for expert_id, row in enumerate(table["ranked_indices"])
    }
    config_path = model_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_channels = int(config["moe_intermediate_size"])
    if not 0 < int(args.retained_channels) < source_channels:
        raise ValueError("retained channels must be positive and smaller than the source width.")

    index_path = model_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    device = torch.device(args.device)
    diagnostics = []
    total_size = 0
    probes_by_layer: dict[int, torch.Tensor] = {}

    with ExitStack() as stack:
        handles = {
            shard: stack.enter_context(safe_open(model_path / shard, framework="pt", device="cpu"))
            for shard in sorted(set(weight_map.values()))
        }
        for shard_name in sorted(handles):
            tensors = {}
            handle = handles[shard_name]
            for name in handle.keys():
                tensor = handle.get_tensor(name)
                ids = expert_indices(name)
                if ids is not None:
                    retained = retained_by_expert[ids]
                    if name.endswith(("gate_proj.weight", "up_proj.weight")):
                        tensor = tensor.index_select(0, retained)
                    elif name.endswith("down_proj.weight"):
                        layer_id, expert_id = ids
                        if layer_id not in probes_by_layer:
                            layer_prefix = f"model.layers.{layer_id}"
                            router = _first_tensor(handles, weight_map, [f"{layer_prefix}.mlp.gate.weight"])
                            norm = _first_tensor(
                                handles,
                                weight_map,
                                [
                                    f"{layer_prefix}.post_attention_layernorm.weight",
                                    f"{layer_prefix}.pre_feedforward_layernorm.weight",
                                    f"{layer_prefix}.input_layernorm.weight",
                                ],
                            )
                            probes_by_layer[layer_id] = rms_norm_rows(
                                router.to(device=device), norm.to(device=device), float(config["rms_norm_eps"])
                            )
                        expert_prefix = f"model.layers.{layer_id}.mlp.experts.{expert_id}"
                        gate = _first_tensor(handles, weight_map, [f"{expert_prefix}.gate_proj.weight"]).to(device=device)
                        up = _first_tensor(handles, weight_map, [f"{expert_prefix}.up_proj.weight"]).to(device=device)
                        responses = swiglu_responses(probes_by_layer[layer_id], gate, up)
                        tensor, record = reconstruct_down_proj(
                            responses,
                            tensor.to(device=device),
                            retained.to(device=device),
                            ridge_relative=float(args.ridge_relative),
                        )
                        record.update({"layer_id": layer_id, "expert_id": expert_id})
                        diagnostics.append(record)
                        tensor = tensor.cpu()
                tensors[name] = tensor.contiguous()
                total_size += tensor.numel() * tensor.element_size()
            save_file(tensors, output_dir / shard_name, metadata={"format": "pt"})
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

    config["moe_intermediate_size"] = int(args.retained_channels)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    index.setdefault("metadata", {})["total_size"] = total_size
    (output_dir / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    diagnostic_payload = {
        "probe_source": "all_rmsnorm_router_rows",
        "probe_count": int(config["num_experts"]),
        "ridge_relative": float(args.ridge_relative),
        "expert_count": len(diagnostics),
        "summary": summarize_diagnostics(diagnostics),
        "per_expert": diagnostics,
    }
    diagnostics_path = args.diagnostics_output.expanduser().resolve()
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(json.dumps(diagnostic_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {
        "source_model": str(model_path),
        "source_config_sha256": file_sha256(config_path),
        "source_weight_index_sha256": file_sha256(index_path),
        "channel_cache": str(cache_path),
        "channel_cache_sha256": file_sha256(cache_path),
        "retained_channels": int(args.retained_channels),
        "source_moe_intermediate_size": source_channels,
        "exported_weight_bytes": total_size,
        "reconstruction": {
            "probe_source": "all_rmsnorm_router_rows",
            "probe_count": int(config["num_experts"]),
            "ridge_relative": float(args.ridge_relative),
            "diagnostics": str(diagnostics_path),
            "diagnostics_sha256": file_sha256(diagnostics_path),
            "modifies": ["down_proj.weight"],
            "fixed_channel_mask": True,
        },
    }
    (output_dir / "pruning_export_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())