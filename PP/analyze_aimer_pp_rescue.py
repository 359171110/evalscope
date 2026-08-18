from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from PP.build_protected_rankings import cache_orders
from PP.build_pure_pseudo_profile import (
    _load_first_tensor,
    _load_model_config,
    _load_tensor,
    _load_weight_map,
)
from WICK.build_wick_profile import file_sha256, rms_norm_rows, router_gram_neighbors


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = Path("/data01/datasets/Qwen3-30B-A3B-Instruct-2507")
DEFAULT_AIMER_CACHE = (
    REPO_ROOT / "WICK/experiments/profiles/qwen3_wick_aimer_fixed_diagnostics_20260806/aimer_fixed_rankings.pt"
)
DEFAULT_PP_CACHE = (
    REPO_ROOT
    / "PP/experiments/profiles/down_proj_norm_ablation_20260807/PurePseudo-K8-Q4-NoDownNorm/rankings.pt"
)
DEFAULT_PROFILE_ROOT = REPO_ROOT / "PP/experiments/profiles/functional_pp_frozen_v1_20260807"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "PP/experiments/analysis/aimer_pp_rescue_20260807"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze channels rescued by PP from the frozen AIMER ranking.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--aimer-cache", type=Path, default=DEFAULT_AIMER_CACHE)
    parser.add_argument("--pp-cache", type=Path, default=DEFAULT_PP_CACHE)
    parser.add_argument("--profile-root", type=Path, default=DEFAULT_PROFILE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--router-neighbors", type=int, default=8)
    parser.add_argument("--top-q", type=int, default=4)
    parser.add_argument("--scatter-samples", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def pp_no_down_norm_score(
    probes: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    top_q: int,
) -> torch.Tensor:
    """Return PP-Frozen-v1 scores without a down-projection norm multiplier."""

    if probes.ndim != 2 or gate_weight.ndim != 2 or up_weight.shape != gate_weight.shape:
        raise ValueError("probes, gate_weight, and up_weight have incompatible shapes.")
    if int(probes.shape[1]) != int(gate_weight.shape[1]):
        raise ValueError("probe hidden size must match gate/up input size.")
    selected = int(top_q)
    if not 1 <= selected <= int(probes.shape[0]):
        raise ValueError("top_q must be in [1, number of probes].")
    gate_hidden = F.linear(probes.float(), gate_weight.float())
    up_hidden = F.linear(probes.float(), up_weight.float())
    response = (F.silu(gate_hidden) * up_hidden).abs()
    return torch.topk(response, k=selected, dim=0, largest=True, sorted=False).values.mean(dim=0)


def collect_pp_scores(
    *,
    model_path: Path,
    device: torch.device,
    router_neighbors: int,
    top_q: int,
) -> torch.Tensor:
    config = _load_model_config(model_path)
    weight_map = _load_weight_map(model_path)
    num_layers = int(config["num_hidden_layers"])
    num_experts = int(config["num_experts"])
    scores_by_layer = []
    for layer_id in range(num_layers):
        layer_prefix = f"model.layers.{layer_id}"
        router = _load_tensor(model_path, weight_map, f"{layer_prefix}.mlp.gate.weight").to(device=device)
        norm_weight = _load_first_tensor(
            model_path,
            weight_map,
            [
                f"{layer_prefix}.post_attention_layernorm.weight",
                f"{layer_prefix}.pre_feedforward_layernorm.weight",
                f"{layer_prefix}.input_layernorm.weight",
            ],
        ).to(device=device)
        neighbor_ids = router_gram_neighbors(router, router_neighbors)
        normalized_router = rms_norm_rows(router, norm_weight, float(config["rms_norm_eps"]))
        expert_scores = []
        for expert_id in range(num_experts):
            expert_prefix = f"{layer_prefix}.mlp.experts.{expert_id}"
            gate = _load_tensor(model_path, weight_map, f"{expert_prefix}.gate_proj.weight").to(device=device)
            up = _load_tensor(model_path, weight_map, f"{expert_prefix}.up_proj.weight").to(device=device)
            probes = normalized_router.index_select(0, neighbor_ids[expert_id])
            scores = pp_no_down_norm_score(probes, gate, up, min(top_q, int(probes.shape[0])))
            expert_scores.append(scores.cpu())
            del gate, up
        scores_by_layer.append(torch.stack(expert_scores))
        print(f"Scored PP-Frozen-v1 layer {layer_id + 1}/{num_layers}", flush=True)
    return torch.stack(scores_by_layer)


def inverse_ranks(orders: torch.Tensor) -> torch.Tensor:
    channel_count = int(orders.shape[-1])
    ranks = torch.empty_like(orders)
    ranks.scatter_(2, orders, torch.arange(1, channel_count + 1).expand_as(orders))
    return ranks


def summarize(values: torch.Tensor) -> dict[str, float]:
    values = values.to(torch.float64)
    return {
        "mean": float(values.mean().item()),
        "p10": float(torch.quantile(values, 0.10).item()),
        "median": float(values.median().item()),
        "p90": float(torch.quantile(values, 0.90).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }


def extract_budget_records(
    *,
    budget: str,
    retained_channels: int,
    aimer_orders: torch.Tensor,
    combined_orders: torch.Tensor,
    aimer_ranks: torch.Tensor,
    pp_ranks: torch.Tensor,
    pp_scores: torch.Tensor,
) -> tuple[list[dict[str, int | float | str]], dict[str, object]]:
    records: list[dict[str, int | float | str]] = []
    rescue_count_by_expert = []
    for layer_id in range(int(aimer_orders.shape[0])):
        for expert_id in range(int(aimer_orders.shape[1])):
            aimer_set = set(aimer_orders[layer_id, expert_id, :retained_channels].tolist())
            combined_set = set(combined_orders[layer_id, expert_id, :retained_channels].tolist())
            rescue = sorted(combined_set - aimer_set)
            displaced = sorted(aimer_set - combined_set)
            if len(rescue) != len(displaced):
                raise ValueError("fixed-width rescue and displaced sets must have equal sizes.")
            rescue_count_by_expert.append(len(rescue))
            expert_scores = pp_scores[layer_id, expert_id].to(torch.float64)
            score_median = float(expert_scores.median().item())
            score_scale = max(score_median, torch.finfo(torch.float64).tiny)
            for population, channels in (("rescue", rescue), ("displaced", displaced)):
                for channel_id in channels:
                    score = float(expert_scores[channel_id].item())
                    records.append(
                        {
                            "budget": budget,
                            "population": population,
                            "layer_id": layer_id,
                            "expert_id": expert_id,
                            "channel_id": channel_id,
                            "aimer_rank": int(aimer_ranks[layer_id, expert_id, channel_id].item()),
                            "pp_rank": int(pp_ranks[layer_id, expert_id, channel_id].item()),
                            "pp_score": score,
                            "pp_score_over_expert_median": score / score_scale,
                        }
                    )

    rescue_records = [record for record in records if record["population"] == "rescue"]
    displaced_records = [record for record in records if record["population"] == "displaced"]
    rescue_aimer_ranks = torch.tensor([record["aimer_rank"] for record in rescue_records])
    displaced_aimer_ranks = torch.tensor([record["aimer_rank"] for record in displaced_records])
    rescue_depth = rescue_aimer_ranks - retained_channels
    rescue_pp_scores = torch.tensor([record["pp_score_over_expert_median"] for record in rescue_records])
    displaced_pp_scores = torch.tensor([record["pp_score_over_expert_median"] for record in displaced_records])
    pp_high_threshold = 2.0
    summary = {
        "retained_channels": retained_channels,
        "expert_count": int(aimer_orders.shape[0] * aimer_orders.shape[1]),
        "rescue_channels": len(rescue_records),
        "displaced_channels": len(displaced_records),
        "rescue_per_expert": summarize(torch.tensor(rescue_count_by_expert)),
        "rescue_aimer_rank": summarize(rescue_aimer_ranks),
        "displaced_aimer_rank": summarize(displaced_aimer_ranks),
        "rescue_depth_below_cutoff": summarize(rescue_depth),
        "rescue_fraction_below_aimer_cutoff": float((rescue_aimer_ranks > retained_channels).float().mean().item()),
        "displaced_fraction_within_64_of_aimer_cutoff": float(
            (displaced_aimer_ranks > retained_channels - 64).float().mean().item()
        ),
        "pp_high_definition": "pp_score_over_expert_median >= 2.0",
        "rescue_fraction_aimer_low_pp_high": float(
            ((rescue_aimer_ranks > retained_channels) & (rescue_pp_scores >= pp_high_threshold)).float().mean().item()
        ),
        "displaced_fraction_pp_high": float((displaced_pp_scores >= pp_high_threshold).float().mean().item()),
        "rescue_fraction_at_least_64_below_cutoff": float((rescue_depth >= 64).float().mean().item()),
        "rescue_fraction_at_least_128_below_cutoff": float((rescue_depth >= 128).float().mean().item()),
        "rescue_pp_score": summarize(torch.tensor([record["pp_score"] for record in rescue_records])),
        "displaced_pp_score": summarize(torch.tensor([record["pp_score"] for record in displaced_records])),
        "rescue_pp_score_over_expert_median": summarize(rescue_pp_scores),
        "displaced_pp_score_over_expert_median": summarize(displaced_pp_scores),
        "median_pp_score_ratio_rescue_vs_displaced": float(
            rescue_pp_scores.median().item() / displaced_pp_scores.median().item()
        ),
    }
    return records, summary


def write_records(path: Path, records: list[dict[str, int | float | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def sampled(records: list[dict[str, int | float | str]], population: str, count: int, seed: int) -> list[dict]:
    selected = [record for record in records if record["population"] == population]
    if len(selected) <= count:
        return selected
    generator = np.random.default_rng(seed)
    indices = generator.choice(len(selected), size=count, replace=False)
    return [selected[int(index)] for index in indices]


def draw_figure(
    output_path: Path,
    records_by_budget: dict[str, list[dict[str, int | float | str]]],
    summaries: dict[str, dict[str, object]],
    scatter_samples: int,
    seed: int,
) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold"})
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9.2), constrained_layout=True)
    background = "#f7f5ef"
    figure.patch.set_facecolor(background)
    rescue_color = "#176b68"
    displaced_color = "#a13d2d"
    for column, budget in enumerate(("B6", "B9")):
        records = records_by_budget[budget]
        summary = summaries[budget]
        retained = int(summary["retained_channels"])
        rescue = [record for record in records if record["population"] == "rescue"]

        histogram_axis = axes[0, column]
        histogram_axis.set_facecolor(background)
        histogram_axis.hist(
            [
                [record["aimer_rank"] for record in rescue],
                [record["aimer_rank"] for record in records if record["population"] == "displaced"],
            ],
            bins=np.arange(1, 770, 16),
            label=["PP rescued (R)", "Displaced (D)"],
            color=[rescue_color, displaced_color],
            alpha=0.72,
            histtype="stepfilled",
            stacked=False,
        )
        histogram_axis.axvline(retained, color=displaced_color, linewidth=2, linestyle="--")
        median_rank = float(summary["rescue_aimer_rank"]["median"])
        histogram_axis.axvline(median_rank, color="#172121", linewidth=1.6)
        histogram_axis.set_title(f"{budget}: where PP-rescued channels sit in AIMER")
        histogram_axis.set_xlabel("AIMER rank (1 = highest importance)")
        histogram_axis.set_ylabel("Channel count")
        histogram_axis.legend(frameon=False, loc="upper left")
        histogram_axis.text(
            0.98,
            0.94,
            f"cutoff = {retained}\nmedian = {median_rank:.0f}\nN = {len(rescue):,}",
            transform=histogram_axis.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            color="#172121",
        )
        histogram_axis.grid(axis="y", color="#d8d6cf", linewidth=0.8)

        scatter_axis = axes[1, column]
        scatter_axis.set_facecolor(background)
        displaced_sample = sampled(records, "displaced", scatter_samples, seed + column)
        rescue_sample = sampled(records, "rescue", scatter_samples, seed + 10 + column)
        scatter_axis.scatter(
            [record["aimer_rank"] for record in displaced_sample],
            [record["pp_score_over_expert_median"] for record in displaced_sample],
            s=8,
            alpha=0.12,
            color=displaced_color,
            edgecolors="none",
            label="Displaced",
        )
        scatter_axis.scatter(
            [record["aimer_rank"] for record in rescue_sample],
            [record["pp_score_over_expert_median"] for record in rescue_sample],
            s=8,
            alpha=0.18,
            color=rescue_color,
            edgecolors="none",
            label="PP rescued",
        )
        scatter_axis.axvline(retained, color="#52605f", linewidth=1.5, linestyle="--")
        scatter_axis.axhline(2.0, color="#52605f", linewidth=1.2, linestyle=":")
        scatter_axis.set_yscale("log")
        scatter_axis.set_xlim(1, 768)
        scatter_axis.set_title(f"{budget}: AIMER-low / PP-high population")
        scatter_axis.set_xlabel("AIMER rank")
        scatter_axis.set_ylabel(r"PP score / expert median PP score")
        scatter_axis.grid(color="#d8d6cf", linewidth=0.7, alpha=0.8)
        scatter_axis.legend(frameon=False, loc="upper left")
        score_ratio = float(summary["median_pp_score_ratio_rescue_vs_displaced"])
        scatter_axis.text(
            0.98,
            0.94,
            f"PP-high: >= 2x expert median\nmedian R / D = {score_ratio:.2f}x",
            transform=scatter_axis.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            color="#172121",
        )

    figure.suptitle(
        "What PP rescues from AIMER",
        fontsize=20,
        fontweight="bold",
        color="#172121",
    )
    figure.text(
        0.5,
        0.995,
        "Qwen3-30B-A3B-Instruct-2507 | PP-Frozen-v1 (positive, K=8, Q=4, NoDownNorm)",
        ha="center",
        va="top",
        fontsize=11,
        color="#52605f",
    )
    figure.savefig(output_path, dpi=180, facecolor=background)
    figure.savefig(output_path.with_suffix(".svg"), facecolor=background)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    aimer_cache = args.aimer_cache.expanduser().resolve()
    pp_cache = args.pp_cache.expanduser().resolve()
    profile_root = args.profile_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    aimer_orders = cache_orders(torch.load(aimer_cache, map_location="cpu", weights_only=True))
    pp_orders = cache_orders(torch.load(pp_cache, map_location="cpu", weights_only=True))
    pp_score_path = output_dir / "pp_frozen_v1_scores.pt"
    if pp_score_path.exists():
        pp_scores = torch.load(pp_score_path, map_location="cpu", weights_only=True)
    else:
        pp_scores = collect_pp_scores(
            model_path=model_path,
            device=torch.device(args.device),
            router_neighbors=int(args.router_neighbors),
            top_q=int(args.top_q),
        )
        torch.save(pp_scores, pp_score_path)
    if pp_scores.shape != aimer_orders.shape or pp_orders.shape != aimer_orders.shape:
        raise ValueError("AIMER, PP ranking, and PP score tensors must align.")
    recomputed_pp_orders = torch.argsort(pp_scores, dim=2, descending=True, stable=True)
    mismatch_mask = recomputed_pp_orders != pp_orders
    mismatch_count = int(mismatch_mask.sum().item())
    if mismatch_count:
        recomputed_scores = pp_scores.gather(2, recomputed_pp_orders)
        frozen_scores = pp_scores.gather(2, pp_orders)
        mismatch_score_delta = (recomputed_scores[mismatch_mask] - frozen_scores[mismatch_mask]).abs()
        top77_set_match = all(
            set(recomputed_pp_orders[layer_id, expert_id, :77].tolist())
            == set(pp_orders[layer_id, expert_id, :77].tolist())
            for layer_id in range(int(pp_orders.shape[0]))
            for expert_id in range(int(pp_orders.shape[1]))
        )
        if not top77_set_match:
            raise ValueError("recomputed PP scores change at least one frozen PP Top-77 protection set.")
    else:
        mismatch_score_delta = torch.zeros(1)

    aimer_ranks = inverse_ranks(aimer_orders)
    pp_ranks = inverse_ranks(pp_orders)
    records_by_budget = {}
    summaries = {}
    all_records = []
    for budget, retained in (("B6", 384), ("B9", 576)):
        combined_path = profile_root / f"AIMER-PPFv1-G10-{budget}of12/rankings.pt"
        combined_orders = cache_orders(torch.load(combined_path, map_location="cpu", weights_only=True))
        records, summary = extract_budget_records(
            budget=budget,
            retained_channels=retained,
            aimer_orders=aimer_orders,
            combined_orders=combined_orders,
            aimer_ranks=aimer_ranks,
            pp_ranks=pp_ranks,
            pp_scores=pp_scores,
        )
        records_by_budget[budget] = records
        summaries[budget] = summary
        all_records.extend(records)

    write_records(output_dir / "rescue_displaced_channels.csv", all_records)
    payload = {
        "definition": {
            "aimer_only": "S_A",
            "aimer_plus_pp": "S_AP",
            "rescue": "S_AP minus S_A",
            "displaced": "S_A minus S_AP",
        },
        "protocol": {
            "model_path": str(model_path),
            "aimer_cache": str(aimer_cache),
            "aimer_cache_sha256": file_sha256(aimer_cache),
            "pp_cache": str(pp_cache),
            "pp_cache_sha256": file_sha256(pp_cache),
            "pp_score": "mean top-4 absolute SwiGLU response over positive router self plus K=8 neighbors",
            "down_proj_norm": False,
            "recomputed_pp_ranking_mismatch_positions": mismatch_count,
            "recomputed_pp_ranking_mismatch_max_abs_score_delta": float(mismatch_score_delta.max().item()),
            "recomputed_pp_top77_sets_match": True,
            "recomputed_pp_ranking_mismatch_reason": "tail numerical ties/order differences; frozen permutation remains the set/rank ground truth",
        },
        "budgets": summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    draw_figure(
        output_dir / "aimer_pp_rescue_population.png",
        records_by_budget,
        summaries,
        int(args.scatter_samples),
        int(args.seed),
    )
    print(json.dumps(summaries, indent=2))
    print(output_dir / "aimer_pp_rescue_population.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())