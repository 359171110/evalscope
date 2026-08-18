from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from NAPS_v2.analyze_aimer_proxy_damage import reconstruction_loss, spearman, summarize_expert, top_overlap
from NAPS_v2.build_naps_v2_artifacts import iter_expert_weights, load_weight_map
from NAPS_v2.model_adapter import PurePseudoModelAdapter
from NAPS_v2.naps_v2_core import NapsV2Config, stable_concat_score, swiglu_response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Qwen AIMER masks with real routed expert inputs.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--max-sequences", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-tokens-per-expert", type=int, default=256)
    parser.add_argument("--min-tokens-per-expert", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epsilon", type=float, default=1.0e-12)
    return parser.parse_args()


def load_prompts(path: Path, limit: int) -> list[str]:
    prompts = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(raw_line)
        prompts.append(str(payload.get("question", payload.get("text", payload.get("prompt", "")))))
        if len(prompts) >= limit:
            break
    if not prompts or any(not prompt for prompt in prompts):
        raise ValueError(f"No valid prompts found in {path}")
    return prompts


def find_sparse_blocks(model: torch.nn.Module, layer_ids: list[int]) -> dict[int, torch.nn.Module]:
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
        raise ValueError(f"Could not find Qwen sparse blocks: {sorted(missing)}")
    return found


class RoutedInputCapture:
    def __init__(self, sparse_blocks: dict[int, torch.nn.Module], max_tokens_per_expert: int):
        self.max_tokens_per_expert = max_tokens_per_expert
        self.inputs: dict[tuple[int, int], list[torch.Tensor]] = defaultdict(list)
        self.pending_inputs: dict[int, torch.Tensor] = {}
        self.handles = []
        for layer_id, block in sparse_blocks.items():
            self.handles.append(block.gate.register_forward_pre_hook(self._capture_input(layer_id)))
            self.handles.append(block.gate.register_forward_hook(self._capture_route(layer_id)))

    def _capture_input(self, layer_id: int):
        def hook(_module: torch.nn.Module, args: tuple[Any, ...]) -> None:
            self.pending_inputs[layer_id] = args[0].detach().reshape(-1, args[0].shape[-1])

        return hook

    def _capture_route(self, layer_id: int):
        def hook(_module: torch.nn.Module, _args: tuple[Any, ...], output: tuple[torch.Tensor, ...]) -> None:
            expert_inputs = self.pending_inputs.pop(layer_id)
            top_k_index = output[2].detach()
            for expert_id in torch.unique(top_k_index).tolist():
                key = (layer_id, int(expert_id))
                remaining = self.max_tokens_per_expert - sum(rows.shape[0] for rows in self.inputs[key])
                if remaining <= 0:
                    continue
                token_rows = torch.where((top_k_index == expert_id).any(dim=1))[0][:remaining]
                self.inputs[key].append(expert_inputs.index_select(0, token_rows).float().cpu())

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        if self.pending_inputs:
            raise RuntimeError(f"Unmatched router inputs for layers {sorted(self.pending_inputs)}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    artifact_dir = args.artifact_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(args.input_file.expanduser().resolve(), args.max_sequences)
    weight_map = load_weight_map(model_path)
    adapter = PurePseudoModelAdapter.from_checkpoint(model_path, weight_map)
    if adapter.model_family not in {"qwen3", "qwen3.6"}:
        raise ValueError("This diagnostic only supports Qwen3 and Qwen3.6")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.to(torch.device(args.device))
    model.eval()
    capture = RoutedInputCapture(find_sparse_blocks(model, args.layers), args.max_tokens_per_expert)
    input_device = model.get_input_embeddings().weight.device
    try:
        with torch.inference_mode():
            for sequence_id, prompt in enumerate(prompts):
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length).to(input_device)
                model(**inputs, use_cache=False)
                print(f"Captured sequence {sequence_id + 1}/{len(prompts)}", flush=True)
    finally:
        capture.close()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rankings = torch.load(artifact_dir / "rankings.pt", map_location="cpu", weights_only=True)
    config = NapsV2Config()
    activation = str(adapter.text_config.get("hidden_act", "silu"))
    rows = []
    for layer_id in args.layers:
        layer_rankings = rankings["table"][layer_id]
        width_options = layer_rankings["width_options"].to(torch.long)
        width_position = int(torch.where(width_options == args.width)[0].item())
        for expert_id, gate, up, down in iter_expert_weights(
            model_path, weight_map, adapter, layer_id, torch.device("cpu")
        ):
            captured = capture.inputs.get((layer_id, expert_id), [])
            token_count = sum(item.shape[0] for item in captured)
            if token_count < args.min_tokens_per_expert:
                continue
            expert_inputs = torch.cat(captured)[:args.max_tokens_per_expert]
            responses = swiglu_response(expert_inputs, gate, up, activation=activation)
            full_output = responses.float() @ down.float().transpose(0, 1)
            denominator = full_output.square().sum().clamp_min(args.epsilon)
            activation_energy = responses.float().square().sum(0)
            down_energy = down.float().square().sum(0)
            damage = activation_energy * down_energy / denominator
            aimer_scores = stable_concat_score(gate, up, down, config)
            finite_scores = aimer_scores.masked_fill(
                ~torch.isfinite(aimer_scores), aimer_scores[torch.isfinite(aimer_scores)].min()
            )
            baseline_retained = torch.topk(finite_scores, args.width).indices
            actual_retained = layer_rankings["ranked_indices_by_width"][
                expert_id, width_position, :args.width
            ].to(torch.long)
            activation_retained = torch.topk(activation_energy, args.width).indices
            down_retained = torch.topk(down_energy, args.width).indices
            oracle_retained = torch.topk(damage, args.width).indices
            rows.append({
                "layer_id": layer_id,
                "expert_id": expert_id,
                "real_token_count": token_count,
                **summarize_expert(finite_scores, damage, baseline_retained, actual_retained, args.epsilon),
                "baseline_reconstruction_loss": reconstruction_loss(responses, down, baseline_retained, args.epsilon),
                "actual_reconstruction_loss": reconstruction_loss(responses, down, actual_retained, args.epsilon),
                "spearman_activation_damage": spearman(activation_energy, damage, args.epsilon),
                "activation_top_width_overlap": top_overlap(activation_energy, damage, args.width),
                "activation_reconstruction_loss": reconstruction_loss(
                    responses, down, activation_retained, args.epsilon
                ),
                "spearman_down_damage": spearman(down_energy, damage, args.epsilon),
                "down_top_width_overlap": top_overlap(down_energy, damage, args.width),
                "down_reconstruction_loss": reconstruction_loss(responses, down, down_retained, args.epsilon),
                "energy_oracle_reconstruction_loss": reconstruction_loss(
                    responses, down, oracle_retained, args.epsilon
                ),
            })
    if not rows:
        raise RuntimeError("No experts met min-tokens-per-expert")

    def mean(key: str) -> float:
        return float(torch.tensor([row[key] for row in rows], dtype=torch.double).mean().item())

    summary = {
        "model_path": str(model_path),
        "model_family": adapter.model_family,
        "artifact_dir": str(artifact_dir),
        "input_file": str(args.input_file.expanduser().resolve()),
        "layers": args.layers,
        "sequence_count": len(prompts),
        "expert_count": len(rows),
        "width": args.width,
        "spearman_mean": mean("spearman_aimer_damage"),
        "top_width_overlap_mean": mean("top_width_overlap"),
        "baseline_reconstruction_loss_mean": mean("baseline_reconstruction_loss"),
        "actual_reconstruction_loss_mean": mean("actual_reconstruction_loss"),
        "activation_spearman_mean": mean("spearman_activation_damage"),
        "activation_top_width_overlap_mean": mean("activation_top_width_overlap"),
        "activation_reconstruction_loss_mean": mean("activation_reconstruction_loss"),
        "down_spearman_mean": mean("spearman_down_damage"),
        "down_top_width_overlap_mean": mean("down_top_width_overlap"),
        "down_reconstruction_loss_mean": mean("down_reconstruction_loss"),
        "energy_oracle_reconstruction_loss_mean": mean("energy_oracle_reconstruction_loss"),
    }
    write_csv(output_dir / "expert_real_token_damage.csv", rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())