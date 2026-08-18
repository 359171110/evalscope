from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


DEFAULT_MODEL = "/data01/datasets/gemma-4-26B-A4B-it"


def load_tensor(root: Path, key: str) -> torch.Tensor:
    for shard in sorted(root.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            if key in handle.keys():
                return handle.get_tensor(key).float()
    raise KeyError(f"Tensor not found: {key}")


def route_probes(
    router_weight: torch.Tensor,
    router_scale: torch.Tensor,
    per_expert_scale: torch.Tensor,
    eps: float,
    top_k: int,
) -> dict[str, torch.Tensor]:
    router_weight = router_weight.float()
    probes = router_weight * torch.rsqrt(router_weight.square().mean(dim=-1, keepdim=True) + eps)

    # The construction RMSNorm is Gemma4TextRouter.norm applied exactly once.
    router_input = probes * router_scale.float() * (router_weight.shape[1] ** -0.5)
    logits = router_input @ router_weight.transpose(0, 1)
    probabilities = torch.softmax(logits, dim=-1)
    top_probabilities, top_ids = torch.topk(probabilities, k=top_k, dim=-1)
    top_probabilities = top_probabilities / top_probabilities.sum(dim=-1, keepdim=True)
    routing_weights = top_probabilities * per_expert_scale[top_ids].float()

    order = torch.argsort(logits, dim=-1, descending=True, stable=True)
    ranks = torch.argsort(order, dim=-1, stable=True) + 1
    self_rank = ranks.diagonal().clone()
    self_logit = logits.diagonal()
    kth_logit = logits.gather(1, top_ids[:, -1:,]).squeeze(1)
    next_ids = order[:, top_k]
    next_logit = logits.gather(1, next_ids[:, None]).squeeze(1)
    native_mask = torch.zeros_like(logits, dtype=torch.bool)
    native_mask.scatter_(1, top_ids, True)

    return {
        "probes": probes,
        "logits": logits,
        "probabilities": probabilities,
        "top_ids": top_ids,
        "top_logits": logits.gather(1, top_ids),
        "top_probabilities": top_probabilities,
        "routing_weights": routing_weights,
        "native_mask": native_mask,
        "self_rank": self_rank,
        "self_logit": self_logit,
        "self_vs_topk_boundary_margin": self_logit - kth_logit,
        "topk_boundary_margin": kth_logit - next_logit,
    }


def summarize_layer(layer_index: int, routed: dict[str, torch.Tensor], num_experts: int, top_k: int) -> dict:
    top_ids = routed["top_ids"]
    counts = torch.bincount(top_ids.reshape(-1), minlength=num_experts)
    self_natural = routed["native_mask"].diagonal()
    native_indices = [torch.where(routed["native_mask"][:, expert])[0].tolist() for expert in range(num_experts)]
    coverage_indices = [sorted(set(indices + [expert])) for expert, indices in enumerate(native_indices)]
    self_margin = routed["self_vs_topk_boundary_margin"]
    return {
        "layer_index": layer_index,
        "num_experts": num_experts,
        "router_shape": [num_experts, int(routed["probes"].shape[1])],
        "router_has_bias": False,
        "native_top_k": top_k,
        "routing_activation": "softmax over all experts",
        "selection_rule": "topk over full softmax probabilities; selected probabilities renormalized; per_expert_scale multiplied after renormalization",
        "self_route_rate": float(self_natural.float().mean()),
        "self_route_count": int(self_natural.sum()),
        "zero_native_experts": int((counts == 0).sum()),
        "native_probe_count_min": int(counts.min()),
        "native_probe_count_median": float(counts.float().median()),
        "native_probe_count_max": int(counts.max()),
        "native_probe_count_histogram": {str(int(k)): int(v) for k, v in zip(*torch.unique(counts, return_counts=True))},
        "self_rank_median": float(routed["self_rank"].float().median()),
        "self_rank_max": int(routed["self_rank"].max()),
        "self_topk_margin_median": float(self_margin.median()),
        "self_topk_margin_min": float(self_margin.min()),
        "self_topk_margin_max": float(self_margin.max()),
        "topk_boundary_margin_median": float(routed["topk_boundary_margin"].median()),
        "routing_weight_min": float(routed["routing_weights"].min()),
        "routing_weight_median": float(routed["routing_weights"].median()),
        "routing_weight_max": float(routed["routing_weights"].max()),
        "native_probe_indices_by_expert": native_indices,
        "coverage_probe_indices_by_expert": coverage_indices,
        "native_probe_count_by_expert": counts.tolist(),
        "self_naturally_routed_by_expert": self_natural.tolist(),
        "self_anchor_added_by_expert": [not bool(self_natural[expert]) for expert in range(num_experts)],
    }


def analyze(model_dir: Path, output_dir: Path) -> None:
    config = json.loads((model_dir / "config.json").read_text())
    text_config = config["text_config"]
    num_layers = int(text_config["num_hidden_layers"])
    num_experts = int(text_config["num_experts"])
    top_k = int(text_config["top_k_experts"])
    hidden_size = int(text_config["hidden_size"])
    eps = float(text_config["rms_norm_eps"])
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    payload = {}
    probe_audit = {}

    for layer_index in range(num_layers):
        prefix = f"model.language_model.layers.{layer_index}.router."
        weight = load_tensor(model_dir, prefix + "proj.weight")
        scale = load_tensor(model_dir, prefix + "scale")
        per_expert_scale = load_tensor(model_dir, prefix + "per_expert_scale")
        if tuple(weight.shape) != (num_experts, hidden_size):
            raise ValueError(f"Layer {layer_index} router shape {tuple(weight.shape)} != {(num_experts, hidden_size)}")
        routed = route_probes(weight, scale, per_expert_scale, eps, top_k)
        summaries.append(summarize_layer(layer_index, routed, num_experts, top_k))
        probe_audit[str(layer_index)] = [
            {
                "probe_index": probe_index,
                "top_expert_ids": routed["top_ids"][probe_index].tolist(),
                "top_logits": routed["top_logits"][probe_index].tolist(),
                "top_probabilities": routed["top_probabilities"][probe_index].tolist(),
                "routing_weights": routed["routing_weights"][probe_index].tolist(),
                "self_rank": int(routed["self_rank"][probe_index]),
                "self_vs_topk_boundary_margin": float(routed["self_vs_topk_boundary_margin"][probe_index]),
                "topk_boundary_margin": float(routed["topk_boundary_margin"][probe_index]),
            }
            for probe_index in range(num_experts)
        ]
        payload[str(layer_index)] = {
            "probe": routed["probes"],
            "top_ids": routed["top_ids"],
            "top_logits": routed["top_logits"],
            "top_probabilities": routed["top_probabilities"],
            "routing_weights": routed["routing_weights"],
            "logits": routed["logits"],
            "probabilities": routed["probabilities"],
            "self_rank": routed["self_rank"],
            "self_vs_topk_boundary_margin": routed["self_vs_topk_boundary_margin"],
            "topk_boundary_margin": routed["topk_boundary_margin"],
        }

    summary = {
        "model_dir": str(model_dir),
        "dtype_for_analysis": "float32",
        "probe_definition": "q_i = RMSNorm(router.proj.weight[i, :]) exactly once; router norm is represented by this construction norm",
        "router_input_position": "Gemma4TextDecoderLayer residual, immediately before Gemma4TextRouter",
        "rms_norm_eps": eps,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "hidden_size": hidden_size,
        "native_top_k": top_k,
        "layers": summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "per_probe.json").write_text(json.dumps(probe_audit, indent=2) + "\n")
    torch.save(payload, output_dir / "per_probe.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Gemma4 native routing of NAPS-v2 router probes.")
    parser.add_argument("--model-dir", type=Path, default=Path(DEFAULT_MODEL))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.model_dir, args.output_dir)


if __name__ == "__main__":
    main()
