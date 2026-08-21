from __future__ import annotations

import argparse
import hashlib
import json
import torch
from pathlib import Path
from typing import Any

from static_moe_prunning.code.src.calibration_data import load_shared_calibration_tokens
from Wanda.model_adapter import WandaModelAdapter
from Wanda.wanda_core import WandaStatistics


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_weight_map(model_path: Path) -> dict[str, str]:
    payload = json.loads((model_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    return {str(name): str(shard) for name, shard in payload["weight_map"].items()}


def native_route_from_gate_output(
    output: Any,
    *,
    top_k: int,
    norm_topk_prob: bool,
    weight_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(top_k_index, top_k_weights)`` from a native Qwen gate output."""

    if isinstance(output, tuple) and len(output) >= 3:
        return output[2], output[1]
    logits = output[0] if isinstance(output, tuple) else output
    if not torch.is_tensor(logits) or logits.ndim != 2:
        raise ValueError("Qwen gate output must be router logits or a (logits, weights, indices) tuple.")
    weights, indices = torch.topk(torch.softmax(logits.float(), dim=-1), int(top_k), dim=-1)
    if norm_topk_prob:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return indices, weights.to(dtype=weight_dtype)


def router_top_k(block: torch.nn.Module) -> int:
    top_k = getattr(block, "top_k", None)
    if top_k is None:
        top_k = getattr(block.gate, "top_k", None)
    if top_k is None:
        raise ValueError("Could not resolve the native router top-k.")
    return int(top_k)


def load_causal_or_conditional_model(model_path: Path, load_kwargs: dict[str, object]) -> torch.nn.Module:
    """Load CausalLM or VLM ConditionalGeneration checkpoints used by the four MoE families."""

    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
    architectures = [str(item) for item in (getattr(config, "architectures", None) or [])]
    if any("ConditionalGeneration" in item for item in architectures):
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText.from_pretrained(model_path, **load_kwargs)
    return AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)


class Gemma4WandaCapture:

    def __init__(self, model: torch.nn.Module, statistics: WandaStatistics) -> None:
        self.handles = []
        found = set()
        for module in model.modules():
            if module.__class__.__name__ != "Gemma4TextDecoderLayer":
                continue
            layer_id = int(module.layer_idx)
            if layer_id not in statistics.layer_ids or not hasattr(module, "experts"):
                continue
            self.handles.append(module.experts.register_forward_pre_hook(self._hook(layer_id, statistics)))
            found.add(layer_id)
        missing = set(statistics.layer_ids) - found
        if missing:
            raise ValueError(f"Missing Gemma4 decoder layers: {sorted(missing)}")

    @staticmethod
    def _hook(layer_id: int, statistics: WandaStatistics) -> Any:

        def hook(module: torch.nn.Module, args: tuple[Any, ...]) -> None:
            if len(args) != 3:
                raise ValueError(f"Expected Gemma4 experts to receive three inputs, got {len(args)}.")
            statistics.update(layer_id, args[0], args[1], args[2], module)

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


class QwenWandaCapture:

    def __init__(self, model: torch.nn.Module, statistics: WandaStatistics) -> None:
        self.handles = []
        self.pending: dict[int, torch.Tensor] = {}
        found = set()
        supported = {"Qwen3MoeSparseMoeBlock", "Qwen3_5MoeSparseMoeBlock"}
        for name, module in model.named_modules():
            if module.__class__.__name__ not in supported:
                continue
            parts = name.split(".")
            if "layers" not in parts:
                continue
            layer_id = int(parts[parts.index("layers") + 1])
            if layer_id not in statistics.layer_ids:
                continue
            self.handles.append(module.gate.register_forward_pre_hook(self._input_hook(layer_id)))
            self.handles.append(module.gate.register_forward_hook(self._route_hook(layer_id, module, statistics)))
            found.add(layer_id)
        missing = set(statistics.layer_ids) - found
        if missing:
            raise ValueError(f"Missing Qwen sparse MoE layers: {sorted(missing)}")

    def _input_hook(self, layer_id: int) -> Any:

        def hook(_module: torch.nn.Module, args: tuple[Any, ...]) -> None:
            self.pending[layer_id] = args[0].detach()

        return hook

    def _route_hook(self, layer_id: int, block: torch.nn.Module, statistics: WandaStatistics) -> Any:

        def hook(_module: torch.nn.Module, _args: tuple[Any, ...], output: Any) -> None:
            if layer_id not in self.pending:
                raise RuntimeError(f"Qwen router output arrived before input at layer {layer_id}.")
            inputs = self.pending.pop(layer_id)
            indices, weights = native_route_from_gate_output(
                output,
                top_k=router_top_k(block),
                norm_topk_prob=bool(getattr(block, "norm_topk_prob", getattr(block.gate, "norm_topk_prob", False))),
                weight_dtype=inputs.dtype,
            )
            statistics.update(layer_id, inputs, indices, weights, block.experts)

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        if self.pending:
            raise RuntimeError(f"Unmatched Qwen router inputs: {sorted(self.pending)}")


class DeepSeekWandaCapture:
    """Capture native DeepSeek-V2 routing without calling ``gate`` as a Linear."""

    def __init__(self, model: torch.nn.Module, statistics: WandaStatistics) -> None:
        self.handles = []
        found = set()
        supported = {"DeepseekV2MoE", "DeepseekV2Moe"}
        for name, module in model.named_modules():
            if module.__class__.__name__ not in supported:
                continue
            parts = name.split(".")
            if "layers" not in parts:
                continue
            layer_id = int(parts[parts.index("layers") + 1])
            if layer_id not in statistics.layer_ids:
                continue
            self.handles.append(module.register_forward_pre_hook(self._hook(layer_id, module, statistics)))
            found.add(layer_id)
        missing = set(statistics.layer_ids) - found
        if missing:
            raise ValueError(f"Missing DeepSeek MoE layers: {sorted(missing)}")

    @staticmethod
    def _route(module: torch.nn.Module, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if hasattr(module, "route_tokens_to_experts"):
            router_logits = torch.nn.functional.linear(
                hidden_states.type(torch.float32),
                module.gate.weight.type(torch.float32),
            )
            return module.route_tokens_to_experts(router_logits)
        routed = module.gate(hidden_states)
        if isinstance(routed, tuple) and len(routed) >= 2:
            return routed[0], routed[1]
        raise ValueError("DeepSeek MoE gate must return top-k indices and weights.")

    @classmethod
    def _hook(cls, layer_id: int, module: torch.nn.Module, statistics: WandaStatistics) -> Any:

        def hook(_module: torch.nn.Module, args: tuple[Any, ...]) -> None:
            hidden_states = args[0]
            indices, weights = cls._route(module, hidden_states)
            statistics.update(layer_id, hidden_states, indices, weights, module.experts)

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect train-only routed statistics for structured MoE Wanda.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--calibration-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--calibration-sequences", type=int)
    parser.add_argument("--route-weighting", choices=("none", "mass", "square"), default="mass")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if output_path.exists() and not args.force:
        raise FileExistsError(f"Wanda statistics already exist: {output_path}")
    weight_map = load_weight_map(model_path)
    adapter = WandaModelAdapter.from_checkpoint(model_path, weight_map)
    architecture = adapter.architecture
    tokens, calibration = load_shared_calibration_tokens(
        args.calibration_cache,
        required_sequence_length=int(args.sequence_length),
        model_path=model_path,
        device="cpu",
    )
    available_sequences = int(tokens.shape[1]) // int(args.sequence_length)
    sequence_count = available_sequences if args.calibration_sequences is None else int(args.calibration_sequences)
    if not 0 < sequence_count <= available_sequences:
        raise ValueError(f"calibration-sequences must be in [1, {available_sequences}].")
    load_kwargs: dict[str, object] = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if args.device_map != "none":
        load_kwargs["device_map"] = args.device_map
    model = load_causal_or_conditional_model(model_path, load_kwargs)
    if args.device_map == "none":
        model = model.to(torch.device(args.device))
    model.eval()
    statistics = WandaStatistics(
        architecture.moe_layer_ids(),
        architecture.num_experts,
        architecture.hidden_size,
        architecture.intermediate_size,
        args.route_weighting,
    )
    if architecture.model_family == "gemma4":
        capture = Gemma4WandaCapture(model, statistics)
    elif architecture.model_family == "deepseek_v2":
        capture = DeepSeekWandaCapture(model, statistics)
    else:
        capture = QwenWandaCapture(model, statistics)
    input_device = model.get_input_embeddings().weight.device
    try:
        with torch.inference_mode():
            for sequence_id in range(sequence_count):
                begin = sequence_id * int(args.sequence_length)
                input_ids = tokens[:, begin:begin + int(args.sequence_length)].to(input_device)
                model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False)
                completed = sequence_id + 1
                if completed == 1 or completed % 8 == 0 or completed == sequence_count:
                    print(f"wanda_calibration_progress={completed}/{sequence_count}", flush=True)
    finally:
        capture.close()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "purpose": "structured_moe_wanda_statistics",
        "model_path": str(model_path),
        "model_family": architecture.model_family,
        "architecture": adapter.metadata(),
        "model_provenance": {
            "config_sha256": file_sha256(model_path / "config.json"),
            "weight_index_sha256": file_sha256(model_path / "model.safetensors.index.json"),
        },
        "calibration": {
            **calibration,
            "path": str(args.calibration_cache.expanduser().resolve()),
            "sequence_length": int(args.sequence_length),
            "calibration_sequences": sequence_count,
            "calibration_tokens": sequence_count * int(args.sequence_length),
        },
        **statistics.payload(),
    }
    temporary = output_path.with_suffix(f"{output_path.suffix}.tmp")
    torch.save(payload, temporary)
    temporary.replace(output_path)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())