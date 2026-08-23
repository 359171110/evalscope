#!/usr/bin/env python3
"""Build LayerProp rankings from native decoder residuals.

This is data-free: inputs are a deterministic vocab lattice, not WikiText or
benchmark text. Packed expert modules (Gemma4, Qwen3, Qwen3.6) are hooked on
``forward(hidden, top_k_index, top_k_weights)``. DeepSeek-V2 uses separate
ModuleList experts; the routed gate is replayed in a pre-hook. Shared experts
are never scored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from AIMER_Mix.mix_core import file_sha256
from AIMER_Mix.model_adapter import AIMERMixModelAdapter
from AIMER_MIX_PLUS.build_pseudo_source_artifacts import build_source_payload, load_weight_map, scores_to_table
from AIMER_MIX_PLUS.layerprop_core import (
    accumulate_routed_channel_scores,
    finalize_layerprop_scores,
    layerprop_payload_metadata,
    pack_separate_expert_weights,
    synthetic_input_ids,
)

PACKED_EXPERT_CLASS_NAMES = {
    "Gemma4TextExperts",
    "Qwen3MoeExperts",
    "Qwen3_5MoeExperts",
}
SEPARATE_MOE_CLASS_NAMES = {
    "DeepseekV2MoE",
}


def layer_id_from_module_name(name: str) -> int | None:
    parts = name.split(".")
    if "layers" not in parts:
        return None
    index = parts.index("layers") + 1
    if index >= len(parts) or not parts[index].isdigit():
        return None
    return int(parts[index])


def load_hf_model(model_path: Path, device: torch.device) -> torch.nn.Module:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
    architectures = [str(item) for item in (getattr(config, "architectures", None) or [])]
    load_kwargs: dict[str, object] = {
        "dtype": torch.bfloat16,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "attn_implementation": "sdpa",
    }
    loader = AutoModelForImageTextToText if any(
        "ConditionalGeneration" in item for item in architectures
    ) else AutoModelForCausalLM
    model = loader.from_pretrained(model_path, **load_kwargs)
    return model.to(device).eval()


class PackedExpertLayerPropCapture:
    """Hook packed expert modules after native routing and FFN-space RMSNorm."""

    def __init__(
        self,
        model: torch.nn.Module,
        layer_ids: tuple[int, ...],
        num_experts: int,
        channels: int,
        activation: str,
        score_mode: str,
        device: torch.device,
    ) -> None:
        self.layer_ids = layer_ids
        self.activation = activation
        self.score_mode = score_mode
        self.scores = {
            layer_id: torch.zeros(num_experts, channels, dtype=torch.float32, device=device) for layer_id in layer_ids
        }
        self.mass = {layer_id: torch.zeros(num_experts, dtype=torch.float32, device=device) for layer_id in layer_ids}
        self.hit_counts = {
            layer_id: torch.zeros(num_experts, dtype=torch.float32, device=device) for layer_id in layer_ids
        }
        self.handles: list[Any] = []
        found: set[int] = set()
        for name, module in model.named_modules():
            if module.__class__.__name__ not in PACKED_EXPERT_CLASS_NAMES:
                continue
            layer_id = layer_id_from_module_name(name)
            if layer_id is None or layer_id not in self.scores:
                continue
            self.handles.append(module.register_forward_pre_hook(self._hook(layer_id)))
            found.add(layer_id)
        missing = set(layer_ids) - found
        if missing:
            raise ValueError(f"Missing packed expert modules for LayerProp: {sorted(missing)}")

    def _hook(self, layer_id: int) -> Any:

        def hook(module: torch.nn.Module, args: tuple[Any, ...]) -> None:
            if len(args) != 3:
                raise ValueError(f"Expected packed experts to receive three inputs, got {len(args)}.")
            hidden_ffn, top_k_index, top_k_weights = args
            accumulate_routed_channel_scores(
                hidden_ffn,
                top_k_index,
                top_k_weights,
                module.gate_up_proj,
                module.down_proj,
                activation=self.activation,
                score_mode=self.score_mode,
                scores=self.scores[layer_id],
                mass=self.mass[layer_id],
                hit_counts=self.hit_counts[layer_id],
            )

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


Gemma4LayerPropCapture = PackedExpertLayerPropCapture


class SeparateMoeLayerPropCapture:
    """Hook DeepSeek-style MoE blocks that keep per-expert Linear modules."""

    def __init__(
        self,
        model: torch.nn.Module,
        layer_ids: tuple[int, ...],
        num_experts: int,
        channels: int,
        activation: str,
        score_mode: str,
        device: torch.device,
    ) -> None:
        self.layer_ids = layer_ids
        self.activation = activation
        self.score_mode = score_mode
        self.scores = {
            layer_id: torch.zeros(num_experts, channels, dtype=torch.float32, device=device) for layer_id in layer_ids
        }
        self.mass = {layer_id: torch.zeros(num_experts, dtype=torch.float32, device=device) for layer_id in layer_ids}
        self.hit_counts = {
            layer_id: torch.zeros(num_experts, dtype=torch.float32, device=device) for layer_id in layer_ids
        }
        self.handles: list[Any] = []
        found: set[int] = set()
        for name, module in model.named_modules():
            if module.__class__.__name__ not in SEPARATE_MOE_CLASS_NAMES:
                continue
            layer_id = layer_id_from_module_name(name)
            if layer_id is None or layer_id not in self.scores:
                continue
            self.handles.append(module.register_forward_pre_hook(self._hook(layer_id)))
            found.add(layer_id)
        missing = set(layer_ids) - found
        if missing:
            raise ValueError(f"Missing separate MoE modules for LayerProp: {sorted(missing)}")

    def _hook(self, layer_id: int) -> Any:

        def hook(module: torch.nn.Module, args: tuple[Any, ...]) -> None:
            hidden_states = args[0]
            topk_idx, topk_weight, *_rest = module.gate(hidden_states)
            hidden_ffn = hidden_states.reshape(-1, hidden_states.shape[-1])
            tokens = int(hidden_ffn.shape[0])
            top_k = int(topk_idx.shape[-1])
            gate_up_proj, down_proj = pack_separate_expert_weights(module.experts)
            accumulate_routed_channel_scores(
                hidden_ffn,
                topk_idx.reshape(tokens, top_k),
                topk_weight.reshape(tokens, top_k),
                gate_up_proj,
                down_proj,
                activation=self.activation,
                score_mode=self.score_mode,
                scores=self.scores[layer_id],
                mass=self.mass[layer_id],
                hit_counts=self.hit_counts[layer_id],
            )

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LayerProp rankings from native decoder residuals.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--num-sequences", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--score-mode", choices=("activation", "output"), default="output")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    output_path = args.output_cache.expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(output_path)
    weight_map = load_weight_map(model_path)
    adapter = AIMERMixModelAdapter.from_checkpoint(model_path, weight_map)
    architecture = adapter.architecture
    device = torch.device(args.device)
    text_config = adapter.text_config
    input_ids = synthetic_input_ids(
        vocab_size=int(text_config["vocab_size"]),
        num_sequences=int(args.num_sequences),
        sequence_length=int(args.sequence_length),
        bos_token_id=int(text_config.get("bos_token_id", 2)),
        pad_token_id=int(text_config.get("pad_token_id") or 0),
    )
    print(
        f"layerprop_start family={architecture.model_family} device={device} sequences={args.num_sequences} "
        f"seq_len={args.sequence_length} score_mode={args.score_mode}",
        flush=True,
    )
    model = load_hf_model(model_path, device)
    capture_cls = SeparateMoeLayerPropCapture if architecture.model_family == "deepseek_v2" else PackedExpertLayerPropCapture
    capture = capture_cls(
        model,
        architecture.moe_layer_ids(),
        architecture.num_experts,
        architecture.intermediate_size,
        architecture.activation,
        args.score_mode,
        device,
    )
    embed_device = model.get_input_embeddings().weight.device
    try:
        with torch.inference_mode():
            for sequence_id, row in enumerate(input_ids):
                batch = row.unsqueeze(0).to(embed_device)
                model(input_ids=batch, attention_mask=torch.ones_like(batch), use_cache=False)
                completed = sequence_id + 1
                if completed == 1 or completed % 8 == 0 or completed == int(args.num_sequences):
                    print(f"layerprop_progress={completed}/{args.num_sequences}", flush=True)
    finally:
        capture.close()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    tables: dict[int, dict[str, torch.Tensor | int]] = {}
    coverage_rows = []
    stability_rows = []
    hit_rows = []
    layer_metadata = []
    meta = layerprop_payload_metadata(
        num_sequences=int(args.num_sequences),
        sequence_length=int(args.sequence_length),
        score_mode=str(args.score_mode),
        family=architecture.model_family,
    )
    for layer_id in architecture.moe_layer_ids():
        scores, coverage, stability = finalize_layerprop_scores(
            capture.scores[layer_id].cpu(),
            capture.mass[layer_id].cpu(),
            capture.hit_counts[layer_id].cpu(),
        )
        tables[layer_id] = scores_to_table(scores, architecture.channel_alignment)
        coverage_rows.append(coverage)
        stability_rows.append(stability)
        hit_rows.append(capture.hit_counts[layer_id].cpu().float())
        layer_metadata.append({
            "layer_id": layer_id,
            "experts_hit": int((coverage > 0).sum().item()),
            "mean_mass": float(capture.mass[layer_id].cpu().mean().item()),
            **meta,
        })
        print(
            f"scored_source=layerprop layer={layer_id} experts_hit={int((coverage > 0).sum().item())}",
            flush=True,
        )

    payload = build_source_payload(
        model_path=model_path,
        adapter=adapter,
        method="layerprop",
        tables=tables,
        coverage=torch.stack(coverage_rows),
        stability=torch.stack(stability_rows),
        layer_metadata=layer_metadata,
        score_mode=args.score_mode,
        top_q=0,
        hit_counts=torch.stack(hit_rows),
    )
    payload["pseudo_source"].update(meta)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    summary = {
        "schema_version": 1,
        "method": payload["method"],
        "model_path": str(model_path),
        "model_family": architecture.model_family,
        "layer_ids": list(architecture.moe_layer_ids()),
        "coverage_mean": float(payload["pseudo_source"]["coverage"].mean().item()),
        "stability_mean": float(payload["pseudo_source"]["stability"].mean().item()),
        "cache_sha256": file_sha256(output_path),
        "pseudo_source": {
            key: value
            for key, value in payload["pseudo_source"].items()
            if key not in {"coverage", "stability", "hit_counts"}
        },
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
