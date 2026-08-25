"""Apply a Router LayerProp plan and export a physically narrowed checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .adapters import Gemma4MoeAdapter, Qwen35MoeAdapter, Qwen3MoeAdapter, adapter_for_model
from .core import slice_packed_experts


def _load_torch(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _set_parameter(module: torch.nn.Module, name: str, value: torch.Tensor) -> None:
    old = getattr(module, name)
    parameter = nn.Parameter(value.to(device=old.device, dtype=old.dtype), requires_grad=old.requires_grad)
    setattr(module, name, parameter)


def _update_linear_shape(module: torch.nn.Module, *, in_features: int | None = None, out_features: int | None = None) -> None:
    if in_features is not None and hasattr(module, "in_features"):
        module.in_features = int(in_features)
    if out_features is not None and hasattr(module, "out_features"):
        module.out_features = int(out_features)


def _plan_keep(plans: dict[Any, Any], experts: int, width: int) -> torch.Tensor:
    rows = []
    for expert in range(experts):
        plan = plans.get(expert, plans.get(str(expert)))
        if plan is None:
            raise KeyError(f"Missing plan for expert {expert}")
        keep = torch.as_tensor(plan["retained_channels"], dtype=torch.long)
        if keep.numel() != width:
            raise ValueError(f"Expert {expert} retained width does not match plan")
        rows.append(keep)
    return torch.stack(rows)


def _compensated_down(plans: dict[Any, Any], down: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    output = []
    for expert in range(down.shape[0]):
        plan = plans.get(expert, plans.get(str(expert)))
        sliced = down[expert].index_select(1, keep[expert]).float()
        if bool(plan.get("compensation_accepted", False)):
            compensated = torch.as_tensor(plan["compensated_down"], dtype=torch.float32)
            if compensated.shape != sliced.shape:
                raise ValueError(f"Compensated down shape mismatch for expert {expert}")
            sliced = compensated
        output.append(sliced)
    return torch.stack(output)


def _apply_packed(layer: torch.nn.Module, plans: dict[Any, Any], adapter: Any) -> None:
    gate_up, down = adapter.expert_weights(layer)
    first_plan = plans.get(0, plans.get("0"))
    if first_plan is None:
        raise KeyError("Missing plan for expert 0")
    width = int(torch.as_tensor(first_plan["retained_channels"]).numel())
    keep = _plan_keep(plans, gate_up.shape[0], width).to(device=gate_up.device)
    new_gate_up, _ = slice_packed_experts(gate_up, down, keep)
    new_down = _compensated_down(plans, down, keep).to(device=down.device, dtype=down.dtype)
    experts = getattr(layer, "experts", None)
    if experts is None:
        experts = getattr(getattr(layer, "mlp", None), "experts")
    _set_parameter(experts, "gate_up_proj", new_gate_up)
    _set_parameter(experts, "down_proj", new_down)
    if hasattr(experts, "intermediate_size"):
        experts.intermediate_size = width
    if hasattr(experts, "intermediate_dim"):
        experts.intermediate_dim = width


def _apply_separate(layer: torch.nn.Module, plans: dict[Any, Any], adapter: Qwen3MoeAdapter) -> None:
    mlp = adapter._mlp(layer)
    experts = mlp.experts
    first_plan = plans[0] if 0 in plans else plans["0"]
    width = len(first_plan["retained_channels"])
    for expert_id, expert in enumerate(experts):
        plan = plans.get(expert_id, plans.get(str(expert_id)))
        keep = torch.as_tensor(plan["retained_channels"], dtype=torch.long, device=expert.gate_proj.weight.device)
        gate = expert.gate_proj.weight.index_select(0, keep)
        up = expert.up_proj.weight.index_select(0, keep)
        down = expert.down_proj.weight.index_select(1, keep).float()
        if bool(plan.get("compensation_accepted", False)):
            down = torch.as_tensor(plan["compensated_down"], dtype=torch.float32, device=down.device)
        _set_parameter(expert.gate_proj, "weight", gate)
        _set_parameter(expert.up_proj, "weight", up)
        _set_parameter(expert.down_proj, "weight", down)
        _update_linear_shape(expert.gate_proj, out_features=width)
        _update_linear_shape(expert.up_proj, out_features=width)
        _update_linear_shape(expert.down_proj, in_features=width)


def _update_config(model: torch.nn.Module, width: int) -> None:
    config = model.config
    for target in (config, getattr(config, "text_config", None)):
        if target is not None and hasattr(target, "moe_intermediate_size"):
            target.moe_intermediate_size = int(width)


def export_checkpoint(model_path: Path, plan_path: Path, output_dir: Path) -> None:
    """Load, narrow routed experts, preserve shared branches, and save the model."""

    plan = _load_torch(plan_path)
    if plan.get("method") != "router_conditioned_multi_origin_layerprop":
        raise ValueError("Unsupported or malformed Router LayerProp plan")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    from .build_plan import load_hf_model

    model = load_hf_model(model_path, torch.device("cpu"), torch.float32)
    adapter = adapter_for_model(model)
    width = int(plan["retained_channels"])
    layer_plans = plan["layers"]
    for layer_key, plans in layer_plans.items():
        layer_id = int(layer_key)
        layer = adapter.layers()[layer_id]
        if isinstance(adapter, Qwen3MoeAdapter) and not adapter.metadata.packed_experts:
            _apply_separate(layer, plans, adapter)
        elif isinstance(adapter, Qwen3MoeAdapter):
            _apply_packed(layer, plans, adapter)
        elif isinstance(adapter, (Qwen35MoeAdapter, Gemma4MoeAdapter)):
            _apply_packed(layer, plans, adapter)
        else:
            raise ValueError(f"Unsupported adapter for export: {type(adapter).__name__}")
    _update_config(model, width)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    for source in model_path.iterdir():
        if (
            source.name in {"config.json", ".git", ".cache"}
            or source.suffix in {".safetensors", ".bin"}
            or source.name.endswith(".index.json")
        ):
            continue
        if source.is_dir():
            shutil.copytree(source, output_dir / source.name, dirs_exist_ok=True)
        else:
            shutil.copy2(source, output_dir / source.name)
    config_path = output_dir / "config.json"
    if config_path.exists():
        config_json = json.loads(config_path.read_text(encoding="utf-8"))
        for target in (config_json, config_json.get("text_config")):
            if isinstance(target, dict) and "moe_intermediate_size" in target:
                target["moe_intermediate_size"] = width
        config_path.write_text(json.dumps(config_json, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "method": "router_conditioned_multi_origin_layerprop",
        "source_model": str(model_path),
        "plan": str(plan_path),
        "model_family": adapter.metadata.family,
        "source_expert_width": adapter.metadata.intermediate_size,
        "retained_channels": width,
        "export_layout": "uniform_packed_or_separate_expert_slice_with_optional_down_fold",
        "shared_expert_pruned": False,
        "summary": plan.get("summary", {}),
    }
    (output_dir / "router_layerprop_export_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    export_checkpoint(args.model_path.expanduser().resolve(), args.plan.expanduser().resolve(), args.output_dir.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
