from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare dense and pruned decoder-layer hidden states.")
    parser.add_argument("--dense-model", type=Path, required=True)
    parser.add_argument("--pruned-model", type=Path, required=True)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-sequences", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_prompts(path: Path, limit: int) -> list[str]:
    prompts = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(raw_line)
        prompt = str(payload.get("question", payload.get("text", payload.get("prompt", ""))))
        if not prompt:
            raise ValueError(f"Prompt row has no text: {raw_line[:120]}")
        prompts.append(prompt)
        if len(prompts) >= limit:
            break
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def decoder_layers(model: torch.nn.Module) -> dict[int, torch.nn.Module]:
    layers = {}
    for module_path, module in model.named_modules():
        module_name = module.__class__.__name__
        if not module_name.endswith("DecoderLayer"):
            continue
        if hasattr(module, "layer_idx"):
            layer_id = int(module.layer_idx)
        else:
            parts = module_path.split(".")
            if "layers" not in parts:
                continue
            layer_id = int(parts[parts.index("layers") + 1])
        layers[layer_id] = module
    if not layers:
        raise ValueError("No decoder layers with layer_idx were found")
    return layers


class LayerCapture:
    def __init__(self, layers: dict[int, torch.nn.Module]):
        self.outputs: dict[int, list[torch.Tensor]] = {layer_id: [] for layer_id in layers}
        self.handles = [module.register_forward_hook(self._hook(layer_id)) for layer_id, module in layers.items()]

    def _hook(self, layer_id: int):
        def hook(_module: torch.nn.Module, _args: tuple[Any, ...], output: Any) -> None:
            hidden_states = output[0] if isinstance(output, tuple) else output
            self.outputs[layer_id].append(hidden_states.detach().float().cpu())

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


@dataclass
class SequenceTrace:
    input_ids: torch.Tensor
    prompt_length: int
    generated_length: int
    predicted_ids: torch.Tensor


@dataclass
class ModelTrace:
    hidden_states: dict[int, list[torch.Tensor]]
    sequences: list[SequenceTrace]


def load_model(model_path: Path, device: torch.device) -> torch.nn.Module:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    return model


def build_reference_sequences(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: list[str],
    max_length: int,
    max_new_tokens: int,
) -> list[SequenceTrace]:
    input_device = model.get_input_embeddings().weight.device
    sequences = []
    with torch.inference_mode():
        for sequence_id, prompt in enumerate(prompts):
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(input_device)
            prompt_length = int(inputs.input_ids.shape[1])
            if max_new_tokens > 0:
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            else:
                generated_ids = inputs.input_ids
            sequences.append(
                SequenceTrace(
                    input_ids=generated_ids.detach().cpu(),
                    prompt_length=prompt_length,
                    generated_length=int(generated_ids.shape[1] - prompt_length),
                    predicted_ids=torch.empty(0, dtype=torch.long),
                )
            )
            print(f"dense reference: {sequence_id + 1}/{len(prompts)}", flush=True)
    return sequences


def collect(model_path: Path, references: list[SequenceTrace], device: torch.device) -> ModelTrace:
    model = load_model(model_path, device)
    capture = LayerCapture(decoder_layers(model))
    input_device = model.get_input_embeddings().weight.device
    sequences = []
    try:
        with torch.inference_mode():
            for sequence_id, reference in enumerate(references):
                input_ids = reference.input_ids.to(input_device)
                output = model(input_ids=input_ids, use_cache=False)
                predictor_start = reference.prompt_length - 1
                predictor_end = predictor_start + reference.generated_length
                predicted_ids = output.logits[:, predictor_start:predictor_end].argmax(dim=-1).detach().cpu()
                sequences.append(
                    SequenceTrace(
                        input_ids=reference.input_ids,
                        prompt_length=reference.prompt_length,
                        generated_length=reference.generated_length,
                        predicted_ids=predicted_ids,
                    )
                )
                print(f"{model_path.name}: {sequence_id + 1}/{len(references)}", flush=True)
    finally:
        capture.close()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return ModelTrace(hidden_states=capture.outputs, sequences=sequences)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def vector_drift_metrics(dense: torch.Tensor, pruned: torch.Tensor) -> dict[str, float]:
    if dense.shape != pruned.shape:
        raise ValueError(f"Hidden-state shape mismatch: {dense.shape} != {pruned.shape}")
    dense = dense.float()
    pruned = pruned.float()
    return {
        "relative_l2": float(((pruned - dense).norm() / dense.norm().clamp_min(1.0e-12)).item()),
        "cosine_drift": float((1.0 - F.cosine_similarity(dense, pruned, dim=-1).mean()).item()),
        "rms_ratio": float(
            (pruned.square().mean().sqrt() / dense.square().mean().sqrt().clamp_min(1.0e-12)).item()
        ),
    }


def token_divergence_metrics(dense_ids: torch.Tensor, pruned_ids: torch.Tensor) -> dict[str, float | int | None]:
    if dense_ids.shape != pruned_ids.shape:
        raise ValueError(f"Predicted-token shape mismatch: {dense_ids.shape} != {pruned_ids.shape}")
    disagreement = dense_ids.ne(pruned_ids).reshape(-1)
    positions = torch.where(disagreement)[0]
    return {
        "prediction_match_rate": float((~disagreement).float().mean().item()) if disagreement.numel() else 1.0,
        "first_token_divergence": int(positions[0].item() + 1) if positions.numel() else None,
    }


def is_before_divergent_token(generated_position: int, first_token_divergence: int | None) -> bool:
    return first_token_divergence is None or generated_position <= first_token_divergence


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    from transformers import AutoTokenizer

    args = parse_args()
    dense_model = args.dense_model.expanduser().resolve()
    pruned_model = args.pruned_model.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(args.input_file.expanduser().resolve(), args.max_sequences)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(dense_model, trust_remote_code=True)
    dense_reference_model = load_model(dense_model, device)
    try:
        references = build_reference_sequences(
            dense_reference_model,
            tokenizer,
            prompts,
            args.max_length,
            args.max_new_tokens,
        )
    finally:
        del dense_reference_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    dense = collect(dense_model, references, device)
    pruned = collect(pruned_model, references, device)
    if dense.hidden_states.keys() != pruned.hidden_states.keys():
        raise ValueError("Dense and pruned checkpoints expose different decoder layers")

    divergence_metrics = [
        token_divergence_metrics(dense_sequence.predicted_ids, pruned_sequence.predicted_ids)
        for dense_sequence, pruned_sequence in zip(dense.sequences, pruned.sequences)
    ]

    rows = []
    position_rows = []
    position_summary_rows = []
    for layer_id in sorted(dense.hidden_states):
        dense_layer = dense.hidden_states[layer_id]
        pruned_layer = pruned.hidden_states[layer_id]
        if len(dense_layer) != len(pruned_layer):
            raise ValueError(f"Layer {layer_id} captured different sequence counts")
        relative_errors = []
        cosine_drifts = []
        rms_ratios = []
        position_metrics: dict[int, dict[str, list[float]]] = {}
        for sequence_id, (dense_hidden, pruned_hidden) in enumerate(zip(dense_layer, pruned_layer)):
            if dense_hidden.shape != pruned_hidden.shape:
                raise ValueError(f"Layer {layer_id} hidden shape mismatch")
            dense_flat = dense_hidden.reshape(-1, dense_hidden.shape[-1])
            pruned_flat = pruned_hidden.reshape(-1, pruned_hidden.shape[-1])
            sequence_metrics = vector_drift_metrics(dense_flat, pruned_flat)
            relative_errors.append(sequence_metrics["relative_l2"])
            cosine_drifts.append(sequence_metrics["cosine_drift"])
            rms_ratios.append(sequence_metrics["rms_ratio"])
            reference = references[sequence_id]
            predictor_start = reference.prompt_length - 1
            for generated_position in range(reference.generated_length):
                token_position = predictor_start + generated_position
                dense_vector = dense_hidden[0, token_position].float()
                pruned_vector = pruned_hidden[0, token_position].float()
                vector_metrics = vector_drift_metrics(dense_vector, pruned_vector)
                one_based_position = generated_position + 1
                first_divergence = divergence_metrics[sequence_id]["first_token_divergence"]
                reference_token_id = int(reference.input_ids[0, reference.prompt_length + generated_position].item())
                dense_token_id = int(dense.sequences[sequence_id].predicted_ids[0, generated_position].item())
                pruned_token_id = int(pruned.sequences[sequence_id].predicted_ids[0, generated_position].item())
                position_rows.append({
                    "sequence_id": sequence_id,
                    "layer_id": layer_id,
                    "generated_position": one_based_position,
                    "token_position": token_position,
                    "reference_token_id": reference_token_id,
                    "dense_next_token_id": dense_token_id,
                    "pruned_next_token_id": pruned_token_id,
                    "token_diverged": dense_token_id != pruned_token_id,
                    "first_token_divergence": first_divergence,
                    "before_divergent_token_is_fed": is_before_divergent_token(one_based_position, first_divergence),
                    **vector_metrics,
                    "cosine_similarity": 1.0 - vector_metrics["cosine_drift"],
                })
                metrics = position_metrics.setdefault(
                    one_based_position,
                    {"relative_l2": [], "cosine_drift": [], "rms_ratio": []},
                )
                for name in metrics:
                    metrics[name].append(vector_metrics[name])
        rows.append({
            "layer_id": layer_id,
            "relative_l2": mean(relative_errors),
            "cosine_drift": mean(cosine_drifts),
            "rms_ratio": mean(rms_ratios),
        })
        for generated_position, metrics in sorted(position_metrics.items()):
            position_summary_rows.append({
                "layer_id": layer_id,
                "generated_position": generated_position,
                "sequence_count": len(metrics["cosine_drift"]),
                "relative_l2": mean(metrics["relative_l2"]),
                "cosine_drift": mean(metrics["cosine_drift"]),
                "cosine_similarity": 1.0 - mean(metrics["cosine_drift"]),
                "rms_ratio": mean(metrics["rms_ratio"]),
            })

    write_csv(output_dir / "layer_drift.csv", rows)
    write_csv(output_dir / "layer_position_drift.csv", position_rows)
    write_csv(output_dir / "layer_position_summary.csv", position_summary_rows)
    divergence_rows = []
    for sequence_id, (reference, dense_sequence, pruned_sequence) in enumerate(
        zip(references, dense.sequences, pruned.sequences)
    ):
        target_ids = reference.input_ids[:, reference.prompt_length :]
        dense_matches = dense_sequence.predicted_ids.eq(target_ids)
        pruned_matches = pruned_sequence.predicted_ids.eq(target_ids)
        divergence = divergence_metrics[sequence_id]
        divergence_rows.append({
            "sequence_id": sequence_id,
            "prompt_tokens": reference.prompt_length,
            "generated_tokens": reference.generated_length,
            "dense_reference_match_rate": float(dense_matches.float().mean().item()) if target_ids.numel() else 1.0,
            "pruned_reference_match_rate": float(pruned_matches.float().mean().item()) if target_ids.numel() else 1.0,
            "dense_pruned_prediction_match_rate": divergence["prediction_match_rate"],
            "first_token_divergence": divergence["first_token_divergence"],
        })
    write_csv(output_dir / "token_divergence.csv", divergence_rows)
    summary = {
        "dense_model": str(dense_model),
        "pruned_model": str(pruned_model),
        "input_file": str(args.input_file.expanduser().resolve()),
        "sequence_count": len(prompts),
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "layers": len(rows),
        "final_relative_l2": rows[-1]["relative_l2"],
        "final_cosine_drift": rows[-1]["cosine_drift"],
        "final_rms_ratio": rows[-1]["rms_ratio"],
        "max_relative_l2": max(row["relative_l2"] for row in rows),
        "max_cosine_drift": max(row["cosine_drift"] for row in rows),
        "mean_pruned_reference_match_rate": mean(
            [row["pruned_reference_match_rate"] for row in divergence_rows]
        ),
        "sequences_with_token_divergence": sum(row["first_token_divergence"] is not None for row in divergence_rows),
        "first_token_divergence_min": min(
            (
                row["first_token_divergence"]
                for row in divergence_rows
                if row["first_token_divergence"] is not None
            ),
            default=None,
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())