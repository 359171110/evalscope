from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from transformers import AutoModelForCausalLM, AutoTokenizer

from NAPS_v2.analyze_gemma4_real_token_damage import find_decoder_layers, load_prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure how Gemma4 branch normalization amplifies mask error.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--width", type=int, default=352)
    parser.add_argument("--max-sequences", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epsilon", type=float, default=1.0e-12)
    return parser.parse_args()


def distortion(full: torch.Tensor, pruned: torch.Tensor, epsilon: float) -> dict[str, float]:
    full = full.float().reshape(-1, full.shape[-1])
    pruned = pruned.float().reshape_as(full)
    relative_l2 = (full - pruned).square().sum() / full.square().sum().clamp_min(epsilon)
    cosine = functional.cosine_similarity(full, pruned, dim=-1, eps=epsilon)
    return {
        "relative_l2": float(relative_l2.item()),
        "cosine_distance": float((1.0 - cosine).mean().item()),
    }


class BranchAmplificationCapture:
    def __init__(
        self,
        decoder_layers: dict[int, torch.nn.Module],
        rankings: dict[str, Any],
        width: int,
        epsilon: float,
    ):
        self.rankings = rankings
        self.width = width
        self.epsilon = epsilon
        self.pending_dense: dict[int, torch.Tensor] = {}
        self.pending_residual: dict[int, torch.Tensor] = {}
        self.rows: list[dict[str, Any]] = []
        self.handles = []
        for layer_id, layer in decoder_layers.items():
            self.handles.append(layer.mlp.register_forward_hook(self._capture_dense(layer_id)))
            self.handles.append(
                layer.pre_feedforward_layernorm_2.register_forward_pre_hook(self._capture_residual(layer_id))
            )
            self.handles.append(layer.experts.register_forward_hook(self._capture_sparse(layer_id, layer)))

    def _capture_dense(self, layer_id: int):
        def hook(_module: torch.nn.Module, _args: tuple[Any, ...], output: torch.Tensor) -> None:
            self.pending_dense[layer_id] = output.detach()

        return hook

    def _capture_residual(self, layer_id: int):
        def hook(_module: torch.nn.Module, args: tuple[Any, ...]) -> None:
            self.pending_residual[layer_id] = args[0].detach()

        return hook

    def _capture_sparse(self, layer_id: int, layer: torch.nn.Module):
        def hook(module: torch.nn.Module, args: tuple[Any, ...], output: torch.Tensor) -> None:
            expert_inputs, top_k_index, top_k_weights = args
            dense_raw = self.pending_dense.pop(layer_id)
            residual = self.pending_residual.pop(layer_id).reshape_as(dense_raw)
            layer_rankings = self.rankings["table"][layer_id]
            width_options = layer_rankings["width_options"].to(torch.long)
            width_positions = torch.where(width_options == self.width)[0]
            if width_positions.numel() != 1:
                raise ValueError(f"Width {self.width} is not uniquely present in layer {layer_id}")
            retained_by_expert = layer_rankings["ranked_indices_by_width"][
                :, int(width_positions.item()), : self.width
            ].to(expert_inputs.device)
            pruned_sparse = torch.zeros_like(output)
            expert_mask = functional.one_hot(top_k_index, num_classes=module.num_experts).permute(2, 1, 0)
            for expert_tensor in torch.where(expert_mask.sum(dim=(-1, -2)) > 0)[0]:
                expert_id = int(expert_tensor.item())
                top_k_pos, token_idx = torch.where(expert_mask[expert_id])
                retained = retained_by_expert[expert_id]
                gate_up = module.gate_up_proj[expert_id]
                intermediate_size = gate_up.shape[0] // 2
                gate_rows = retained
                up_rows = retained + intermediate_size
                selected_rows = torch.cat((gate_rows, up_rows))
                current_state = expert_inputs.index_select(0, token_idx)
                gate, up = functional.linear(current_state, gate_up.index_select(0, selected_rows)).chunk(2, dim=-1)
                response = module.act_fn(gate) * up
                down = module.down_proj[expert_id].index_select(1, retained)
                expert_output = functional.linear(response, down)
                expert_output = expert_output * top_k_weights[token_idx, top_k_pos, None]
                pruned_sparse.index_add_(0, token_idx, expert_output.to(pruned_sparse.dtype))

            full_sparse = output.reshape_as(dense_raw)
            pruned_sparse = pruned_sparse.reshape_as(dense_raw)
            dense_norm = layer.post_feedforward_layernorm_1(dense_raw)
            full_sparse_norm = layer.post_feedforward_layernorm_2(full_sparse)
            pruned_sparse_norm = layer.post_feedforward_layernorm_2(pruned_sparse)
            full_combined = dense_norm + full_sparse_norm
            pruned_combined = dense_norm + pruned_sparse_norm
            full_final = layer.post_feedforward_layernorm(full_combined)
            pruned_final = layer.post_feedforward_layernorm(pruned_combined)
            self.rows.append({
                "layer_id": layer_id,
                "token_count": int(expert_inputs.shape[0]),
                "raw_sparse": distortion(full_sparse, pruned_sparse, self.epsilon),
                "normalized_sparse": distortion(full_sparse_norm, pruned_sparse_norm, self.epsilon),
                "combined_branches": distortion(full_combined, pruned_combined, self.epsilon),
                "final_feedforward": distortion(full_final, pruned_final, self.epsilon),
                "residual_integrated": distortion(
                    residual + full_final,
                    residual + pruned_final,
                    self.epsilon,
                ),
            })

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        if self.pending_dense or self.pending_residual:
            raise RuntimeError("Unmatched Gemma4 branch captures remain")


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stages = ("raw_sparse", "normalized_sparse", "combined_branches", "final_feedforward", "residual_integrated")
    result: dict[str, Any] = {"forward_count": len(rows)}
    for stage in stages:
        result[stage] = {
            metric: float(torch.tensor([row[stage][metric] for row in rows], dtype=torch.double).mean().item())
            for metric in ("relative_l2", "cosine_distance")
        }
    result["relative_l2_multiplier_vs_raw_sparse"] = {
        stage: result[stage]["relative_l2"] / max(result["raw_sparse"]["relative_l2"], 1.0e-12)
        for stage in stages[1:]
    }
    return result


def main() -> int:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    prompts = load_prompts(args.input_file.expanduser().resolve(), args.max_sequences)
    rankings = torch.load(args.artifact_dir / "rankings.pt", map_location="cpu", weights_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(torch.device(args.device))
    model.eval()
    capture = BranchAmplificationCapture(
        find_decoder_layers(model, args.layers),
        rankings,
        args.width,
        args.epsilon,
    )
    input_device = model.get_input_embeddings().weight.device
    try:
        with torch.inference_mode():
            for sequence_id, prompt in enumerate(prompts):
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length).to(input_device)
                model(**inputs, use_cache=False)
                print(f"Measured sequence {sequence_id + 1}/{len(prompts)}", flush=True)
    finally:
        capture.close()

    summary = {
        "model_path": str(model_path),
        "artifact_dir": str(args.artifact_dir.expanduser().resolve()),
        "input_file": str(args.input_file.expanduser().resolve()),
        "layers": args.layers,
        "sequence_count": len(prompts),
        "width": args.width,
        "overall": aggregate(capture.rows),
        "by_layer": {
            str(layer_id): aggregate([row for row in capture.rows if row["layer_id"] == layer_id])
            for layer_id in args.layers
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())