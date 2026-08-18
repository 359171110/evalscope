from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from NAPS_v2.build_naps_v2_artifacts import file_sha256, load_weight_map
from NAPS_v2.build_label_free_calibration import normalize_text
from NAPS_v2.model_adapter import PurePseudoModelAdapter


@dataclass(frozen=True)
class CaptureConfig:
    max_tokens_per_expert: int = 128
    max_length: int = 512
    input_storage_dtype: torch.dtype = torch.bfloat16

    def __post_init__(self) -> None:
        if self.max_tokens_per_expert <= 0 or self.max_length <= 0:
            raise ValueError("Capture limits must be positive")
        if self.input_storage_dtype not in {torch.bfloat16, torch.float32}:
            raise ValueError("Capture input storage dtype must be bfloat16 or float32")


def weights_only_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): weights_only_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [weights_only_safe(item) for item in value]
    return value


def validate_weights_only_payload(payload: dict[str, Any]) -> None:
    buffer = BytesIO()
    torch.save(payload, buffer)
    buffer.seek(0)
    torch.load(buffer, map_location="cpu", weights_only=True)


class RoutedTokenAccumulator:
    def __init__(
        self,
        layer_ids: list[int],
        num_experts: int,
        hidden_size: int,
        limit: int,
        input_storage_dtype: torch.dtype = torch.bfloat16,
        intermediate_size: int | None = None,
    ):
        self.layer_ids = tuple(layer_ids)
        self.num_experts = int(num_experts)
        self.hidden_size = int(hidden_size)
        self.limit = int(limit)
        self.input_storage_dtype = input_storage_dtype
        self.intermediate_size = None if intermediate_size is None else int(intermediate_size)
        if self.intermediate_size is not None and self.intermediate_size <= 0:
            raise ValueError("Response-statistic intermediate size must be positive")
        self.layer_positions = {layer_id: position for position, layer_id in enumerate(self.layer_ids)}
        self.inputs: dict[tuple[int, int], list[torch.Tensor]] = defaultdict(list)
        self.weights: dict[tuple[int, int], list[torch.Tensor]] = defaultdict(list)
        self.total_route_counts: dict[tuple[int, int], int] = defaultdict(int)
        self.total_route_mass: dict[tuple[int, int], float] = defaultdict(float)
        self.route_weighted_response_energy: torch.Tensor | None = None

    def add(
        self,
        layer_id: int,
        expert_inputs: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> None:
        inputs = expert_inputs.detach().reshape(-1, expert_inputs.shape[-1])
        indices = top_k_index.detach().reshape(-1, top_k_index.shape[-1]).long().cpu()
        weights = top_k_weights.detach().reshape(-1, top_k_weights.shape[-1]).float().cpu()
        if inputs.shape[0] != indices.shape[0] or indices.shape != weights.shape:
            raise ValueError("Expert inputs, top-k indices, and top-k weights are not row-aligned")
        if inputs.shape[1] != self.hidden_size:
            raise ValueError(f"Expected hidden size {self.hidden_size}, got {inputs.shape[1]}")
        stored_inputs: torch.Tensor | None = None
        for expert_id in torch.unique(indices).tolist():
            row_ids, slot_ids = torch.where(indices == int(expert_id))
            key = (int(layer_id), int(expert_id))
            self.total_route_counts[key] += int(row_ids.numel())
            self.total_route_mass[key] += float(weights[row_ids, slot_ids].sum().item())
            current_count = sum(chunk.shape[0] for chunk in self.inputs[key])
            remaining = self.limit - current_count
            if remaining <= 0:
                continue
            row_ids = row_ids[:remaining]
            slot_ids = slot_ids[:remaining]
            if stored_inputs is None:
                stored_inputs = inputs.to(device="cpu", dtype=self.input_storage_dtype)
            self.inputs[key].append(stored_inputs.index_select(0, row_ids))
            self.weights[key].append(weights[row_ids, slot_ids])

    def add_response_energy(
        self,
        layer_id: int,
        expert_id: int,
        responses: torch.Tensor,
        route_weights: torch.Tensor,
    ) -> None:
        if self.intermediate_size is None:
            return
        if responses.ndim != 2 or responses.shape[1] != self.intermediate_size:
            raise ValueError("Expert responses do not match the configured intermediate size")
        if route_weights.ndim != 1 or route_weights.shape[0] != responses.shape[0]:
            raise ValueError("Response rows and route weights are not aligned")
        if layer_id not in self.layer_positions or not 0 <= expert_id < self.num_experts:
            raise ValueError("Response statistic references an unknown layer or expert")
        if self.route_weighted_response_energy is None:
            self.route_weighted_response_energy = torch.zeros(
                (len(self.layer_ids), self.num_experts, self.intermediate_size),
                dtype=torch.float32,
                device=responses.device,
            )
        elif self.route_weighted_response_energy.device != responses.device:
            raise ValueError("Response statistics must remain on one device")
        energy = (responses.float() * route_weights.float().unsqueeze(1)).square().sum(0)
        self.route_weighted_response_energy[self.layer_positions[layer_id], expert_id].add_(energy)

    def payload(self) -> dict[str, Any]:
        experts: dict[int, dict[int, dict[str, torch.Tensor | int | float]]] = {}
        for layer_id in self.layer_ids:
            experts[layer_id] = {}
            for expert_id in range(self.num_experts):
                key = (layer_id, expert_id)
                input_chunks = self.inputs.get(key, [])
                weight_chunks = self.weights.get(key, [])
                inputs = (
                    torch.cat(input_chunks, dim=0)[: self.limit]
                    if input_chunks
                    else torch.empty((0, self.hidden_size), dtype=self.input_storage_dtype)
                )
                weights = (
                    torch.cat(weight_chunks, dim=0)[: self.limit]
                    if weight_chunks
                    else torch.empty((0,), dtype=torch.float32)
                )
                if inputs.shape[0] != weights.shape[0]:
                    raise RuntimeError("Stored routed input and weight counts differ")
                experts[layer_id][expert_id] = {
                    "inputs": inputs,
                    "route_weights": weights,
                    "captured_token_count": int(inputs.shape[0]),
                    "captured_route_mass": float(weights.sum().item()),
                    "total_route_count": int(self.total_route_counts.get(key, 0)),
                    "total_route_mass": float(self.total_route_mass.get(key, 0.0)),
                }
        payload = {
            "input_storage_dtype": str(self.input_storage_dtype).removeprefix("torch."),
            "layers": experts,
        }
        if self.route_weighted_response_energy is not None:
            payload["route_weighted_response_energy"] = self.route_weighted_response_energy.cpu()
            payload["response_statistic_scope"] = "all_routed_tokens"
        return payload


def expert_channel_response(
    expert_owner: torch.nn.Module,
    expert_inputs: torch.Tensor,
    expert_id: int,
) -> torch.Tensor:
    if hasattr(expert_owner, "gate_up_proj"):
        gate_up = expert_owner.gate_up_proj[expert_id]
        gate, up = F.linear(expert_inputs, gate_up).chunk(2, dim=-1)
        return expert_owner.act_fn(gate) * up
    if hasattr(expert_owner, "experts"):
        return expert_channel_response(expert_owner.experts, expert_inputs, expert_id)
    expert = expert_owner[expert_id]
    gate = expert.gate_proj(expert_inputs)
    up = expert.up_proj(expert_inputs)
    activation = getattr(expert, "act_fn", getattr(expert_owner, "act_fn", None))
    if activation is None:
        raise ValueError("Could not locate the routed expert activation")
    return activation(gate) * up


def accumulate_response_statistics(
    expert_owner: torch.nn.Module,
    accumulator: RoutedTokenAccumulator,
    layer_id: int,
    expert_inputs: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> None:
    if accumulator.intermediate_size is None:
        return
    inputs = expert_inputs.detach().reshape(-1, expert_inputs.shape[-1])
    indices = top_k_index.detach().reshape(-1, top_k_index.shape[-1])
    weights = top_k_weights.detach().reshape(-1, top_k_weights.shape[-1])
    with torch.no_grad():
        for expert_id in torch.unique(indices).tolist():
            row_ids, slot_ids = torch.where(indices == int(expert_id))
            responses = expert_channel_response(
                expert_owner,
                inputs.index_select(0, row_ids),
                int(expert_id),
            )
            accumulator.add_response_energy(
                layer_id,
                int(expert_id),
                responses,
                weights[row_ids, slot_ids],
            )


class Gemma4Capture:
    def __init__(self, layers: dict[int, torch.nn.Module], accumulator: RoutedTokenAccumulator):
        self.accumulator = accumulator
        self.handles = []
        for layer_id, layer in layers.items():
            self.handles.append(
                layer.experts.register_forward_pre_hook(self._capture_expert_call(layer_id))
            )

    def _capture_expert_call(self, layer_id: int):
        def hook(module: torch.nn.Module, args: tuple[Any, ...]) -> None:
            if len(args) != 3:
                raise ValueError(f"Expected Gemma4 experts to receive three inputs, got {len(args)}")
            expert_inputs, top_k_index, top_k_weights = args
            self.accumulator.add(layer_id, expert_inputs, top_k_index, top_k_weights)
            accumulate_response_statistics(
                module,
                self.accumulator,
                layer_id,
                expert_inputs,
                top_k_index,
                top_k_weights,
            )

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


class QwenCapture:
    def __init__(self, blocks: dict[int, torch.nn.Module], accumulator: RoutedTokenAccumulator):
        self.accumulator = accumulator
        self.pending: dict[int, torch.Tensor] = {}
        self.handles = []
        for layer_id, block in blocks.items():
            self.handles.append(block.gate.register_forward_pre_hook(self._capture_input(layer_id)))
            self.handles.append(block.gate.register_forward_hook(self._capture_route(layer_id, block)))

    def _capture_input(self, layer_id: int):
        def hook(_module: torch.nn.Module, args: tuple[Any, ...]) -> None:
            self.pending[layer_id] = args[0].detach()

        return hook

    def _capture_route(self, layer_id: int, block: torch.nn.Module):
        def hook(_module: torch.nn.Module, _args: tuple[Any, ...], output: tuple[torch.Tensor, ...]) -> None:
            if layer_id not in self.pending:
                raise RuntimeError(f"Qwen router output arrived before input at layer {layer_id}")
            inputs = self.pending.pop(layer_id)
            self.accumulator.add(layer_id, inputs, output[2], output[1])
            accumulate_response_statistics(
                block,
                self.accumulator,
                layer_id,
                inputs,
                output[2],
                output[1],
            )

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        if self.pending:
            raise RuntimeError(f"Unmatched Qwen router inputs: {sorted(self.pending)}")


def find_gemma_layers(model: torch.nn.Module, layer_ids: list[int]) -> dict[int, torch.nn.Module]:
    wanted = set(layer_ids)
    found = {
        int(module.layer_idx): module
        for module in model.modules()
        if module.__class__.__name__ == "Gemma4TextDecoderLayer"
        and int(module.layer_idx) in wanted
    }
    missing = wanted - set(found)
    if missing:
        raise ValueError(f"Missing Gemma4 decoder layers: {sorted(missing)}")
    return found


def find_qwen_blocks(model: torch.nn.Module, layer_ids: list[int]) -> dict[int, torch.nn.Module]:
    wanted = set(layer_ids)
    found = {}
    for name, module in model.named_modules():
        if module.__class__.__name__ not in {"Qwen3MoeSparseMoeBlock", "Qwen3_5MoeSparseMoeBlock"}:
            continue
        parts = name.split(".")
        if "layers" not in parts:
            continue
        layer_id = int(parts[parts.index("layers") + 1])
        if layer_id in wanted:
            found[layer_id] = module
    missing = wanted - set(found)
    if missing:
        raise ValueError(f"Missing Qwen sparse blocks: {sorted(missing)}")
    return found


def load_prompts(path: Path) -> list[str]:
    prompts = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        payload = json.loads(raw_line)
        text = normalize_text(payload.get("text", ""))
        if not text:
            raise ValueError(f"Calibration row has no text: {raw_line[:120]}")
        prompts.append(text)
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def calibration_provenance(calibration_dir: Path) -> dict[str, Any]:
    manifest_path = calibration_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    outputs = {}
    for split in ("fit", "holdout"):
        split_path = calibration_dir / f"{split}.jsonl"
        expected_hash = str(manifest["outputs"][split]["sha256"])
        actual_hash = file_sha256(split_path)
        if actual_hash != expected_hash:
            raise ValueError(f"Calibration {split} hash does not match its manifest")
        outputs[split] = {
            "path": str(split_path),
            "sha256": actual_hash,
            "count": int(manifest["counts"][split]),
        }
    if int(manifest.get("fit_holdout_source_overlap", -1)) != 0:
        raise ValueError("Calibration manifest reports fit/holdout source overlap")
    if int(manifest.get("fit_holdout_prompt_overlap", -1)) != 0:
        raise ValueError("Calibration manifest reports fit/holdout prompt overlap")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "protocol_name": str(manifest["protocol_name"]),
        "fit_holdout_source_overlap": 0,
        "fit_holdout_prompt_overlap": 0,
        "outputs": outputs,
    }


def capture_split(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: list[str],
    accumulator: RoutedTokenAccumulator,
    capture: Gemma4Capture | QwenCapture,
    config: CaptureConfig,
) -> None:
    input_device = model.get_input_embeddings().weight.device
    try:
        with torch.inference_mode():
            for prompt in prompts:
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=config.max_length,
                ).to(input_device)
                model(**inputs, use_cache=False)
    finally:
        capture.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture real routed expert inputs for CHANNEL calibration.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+")
    parser.add_argument("--max-tokens-per-expert", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--input-storage-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    calibration_dir = args.calibration_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if output_path.exists() and not args.force:
        raise FileExistsError(f"Routed-token capture already exists: {output_path}")
    weight_map = load_weight_map(model_path)
    adapter = PurePseudoModelAdapter.from_checkpoint(model_path, weight_map)
    architecture = adapter.channel_architecture
    layer_ids = list(range(adapter.num_layers)) if args.layers is None else sorted(set(args.layers))
    if any(layer_id < 0 or layer_id >= adapter.num_layers for layer_id in layer_ids):
        raise ValueError("Requested layer is outside the model layer range")
    storage_dtype = torch.bfloat16 if args.input_storage_dtype == "bfloat16" else torch.float32
    config = CaptureConfig(args.max_tokens_per_expert, args.max_length, storage_dtype)
    calibration = calibration_provenance(calibration_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(torch.device(args.device))
    model.eval()
    split_payloads = {}
    try:
        for split in ("fit", "holdout"):
            prompts = load_prompts(calibration_dir / f"{split}.jsonl")
            expected_prompt_count = int(calibration["outputs"][split]["count"])
            if len(prompts) != expected_prompt_count:
                raise ValueError(
                    f"Calibration {split} has {len(prompts)} prompts, expected {expected_prompt_count}"
                )
            accumulator = RoutedTokenAccumulator(
                layer_ids,
                architecture.num_experts,
                architecture.hidden_size,
                config.max_tokens_per_expert,
                input_storage_dtype=config.input_storage_dtype,
                intermediate_size=architecture.source_intermediate_size,
            )
            if adapter.model_family == "gemma4":
                capture = Gemma4Capture(find_gemma_layers(model, layer_ids), accumulator)
            else:
                capture = QwenCapture(find_qwen_blocks(model, layer_ids), accumulator)
            capture_split(model, tokenizer, prompts, accumulator, capture, config)
            split_payloads[split] = {
                "prompt_count": len(prompts),
                **accumulator.payload(),
            }
            print(f"Captured {split}: {len(prompts)} prompts", flush=True)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "model_path": str(model_path),
        "model_family": adapter.model_family,
        "architecture": weights_only_safe(asdict(architecture)),
        "model_provenance": {
            "config_sha256": file_sha256(model_path / "config.json"),
            "weight_index_sha256": file_sha256(model_path / "model.safetensors.index.json"),
        },
        "calibration": calibration,
        "layers": layer_ids,
        "max_tokens_per_expert": config.max_tokens_per_expert,
        "max_length": config.max_length,
        "input_storage_dtype": str(config.input_storage_dtype).removeprefix("torch."),
        "splits": split_payloads,
    }
    validate_weights_only_payload({key: value for key, value in payload.items() if key != "splits"})
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(output_path)
    print(output_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())