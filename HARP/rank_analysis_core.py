"""Ranking-structure analysis for HARP Layer/Expert/Channel-SP caches."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

EPS = 1.0e-12
MAD_SCALE = 1.4826


def file_sha256(path: Path) -> str:
    """Return SHA256 of a file without importing CSP at module import time."""

    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_scores(values: torch.Tensor) -> torch.Tensor:
    """Replace non-finite scores with a large negative sentinel for ranking."""

    scores = values.to(dtype=torch.float64)
    return torch.where(torch.isfinite(scores), scores, torch.full_like(scores, -1.0e30))


def participation_mass(scores: torch.Tensor) -> torch.Tensor:
    """Nonnegative participation mass from Channel-SP via expm1."""

    clipped = finite_scores(scores).clamp(min=-80.0, max=80.0)
    return torch.expm1(clipped).clamp(min=0.0)


def ordinary_z(values: torch.Tensor) -> torch.Tensor:
    """Z-score over the last dimension, unbiased=False."""

    mean = values.mean(dim=-1, keepdim=True)
    std = values.std(dim=-1, unbiased=False, keepdim=True).clamp_min(EPS)
    return (values - mean) / std


def robust_z(values: torch.Tensor) -> torch.Tensor:
    """Median / MAD z-score over the last dimension."""

    median = values.median(dim=-1, keepdim=True).values
    abs_dev = (values - median).abs()
    mad = abs_dev.median(dim=-1, keepdim=True).values
    scale = (MAD_SCALE * mad).clamp_min(EPS)
    return (values - median) / scale


def ranks_descending(values: torch.Tensor) -> torch.Tensor:
    """1-based ranks, highest score first, stable on ties."""

    order = torch.argsort(values, dim=-1, descending=True, stable=True)
    ranks = torch.empty_like(order)
    n = int(values.shape[-1])
    iota = torch.arange(1, n + 1, device=values.device)
    ranks.scatter_(-1, order, iota.expand_as(order))
    return ranks


def percentile_from_rank(rank: torch.Tensor, count: int) -> torch.Tensor:
    """Upper-tail percentile: rank 1 -> 1.0."""

    if count <= 1:
        return torch.ones_like(rank, dtype=torch.float64)
    return (int(count) - rank.to(dtype=torch.float64) + 1.0) / float(count)


def pearson(left: torch.Tensor, right: torch.Tensor) -> float:
    """Pearson correlation of two 1-d tensors."""

    a = left.reshape(-1).to(dtype=torch.float64)
    b = right.reshape(-1).to(dtype=torch.float64)
    mask = torch.isfinite(a) & torch.isfinite(b)
    a = a[mask]
    b = b[mask]
    if int(a.numel()) < 3:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.square().sum().sqrt() * b.square().sum().sqrt()).clamp_min(EPS)
    return float((a * b).sum() / denom)


def spearman(left: torch.Tensor, right: torch.Tensor) -> float:
    """Spearman correlation via Pearson of descending ranks."""

    a = left.reshape(-1).to(dtype=torch.float64)
    b = right.reshape(-1).to(dtype=torch.float64)
    return pearson(ranks_descending(a).to(dtype=torch.float64), ranks_descending(b).to(dtype=torch.float64))


def kendall_tau(left: torch.Tensor, right: torch.Tensor) -> float:
    """Kendall tau-b on two 1-d vectors (O(n^2), for short sequences)."""

    a = left.reshape(-1).to(dtype=torch.float64)
    b = right.reshape(-1).to(dtype=torch.float64)
    n = int(a.numel())
    if n < 3:
        return float("nan")
    concordant = 0
    discordant = 0
    for i in range(n - 1):
        da = a[i + 1:] - a[i]
        db = b[i + 1:] - b[i]
        prod = da * db
        concordant += int((prod > 0).sum().item())
        discordant += int((prod < 0).sum().item())
    denom = concordant + discordant
    if denom == 0:
        return float("nan")
    return float((concordant - discordant) / denom)


def cohens_d_prefix(sorted_desc: torch.Tensor, keep_frac: float) -> float:
    """Cohen's d between a descending prefix and the remainder."""

    values = sorted_desc.reshape(-1).to(dtype=torch.float64)
    n = int(values.numel())
    k = max(1, min(n - 1, int(round(keep_frac * n))))
    keep = values[:k]
    drop = values[k:]
    std = values.std(unbiased=False).clamp_min(EPS)
    return float((keep.mean() - drop.mean()) / std)


def jaccard(left: set[int], right: set[int]) -> float:
    """Jaccard overlap of two id sets."""

    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 1.0
    return float(len(left & right) / len(union))


def adjacent_gaps(sorted_desc: torch.Tensor) -> torch.Tensor:
    """Positive gaps between consecutive ranked scores."""

    values = sorted_desc.to(dtype=torch.float64).reshape(-1)
    return values[:-1] - values[1:]


def log_gaps(sorted_desc: torch.Tensor) -> torch.Tensor:
    """Log gaps for nonnegative ranked scores."""

    values = sorted_desc.to(dtype=torch.float64).reshape(-1).clamp_min(0.0)
    return torch.log(values[:-1] + EPS) - torch.log(values[1:] + EPS)


def one_change_point(sorted_desc: torch.Tensor, *, min_frac: float = 0.05) -> dict[str, Any]:
    """Single mean-shift change point on a descending sequence.

    Returns 1-based rank of the last head item, or None when n is too small.
    """

    values = sorted_desc.to(dtype=torch.float64).reshape(-1)
    n = int(values.numel())
    span = max(2, int(math.ceil(min_frac * n)))
    if n < 2 * span:
        return {"rank": None, "stat": 0.0, "n": n}
    best_k = span
    best_stat = -1.0
    for k in range(span, n - span + 1):
        left = values[:k]
        right = values[k:]
        pooled = torch.sqrt(0.5 * (left.var(unbiased=False) + right.var(unbiased=False))).clamp_min(EPS)
        stat = float((left.mean() - right.mean()).abs() / pooled)
        if stat > best_stat:
            best_stat = stat
            best_k = k
    gaps = adjacent_gaps(values)
    gap = float(gaps[best_k - 1].item()) if best_k - 1 < int(gaps.numel()) else 0.0
    gap_q = float(torch.quantile(gaps, 0.90).item()) if int(gaps.numel()) else 0.0
    endpoint_only = best_k <= 2
    natural = (gap >= gap_q) and (not endpoint_only) and best_stat > 1.0
    return {
        "rank": int(best_k),
        "stat": float(best_stat),
        "gap": gap,
        "gap_q90": gap_q,
        "endpoint_only": bool(endpoint_only),
        "natural_boundary": bool(natural),
        "n": n,
    }


def perturb_scores(values: torch.Tensor, relative: float, generator: torch.Generator) -> torch.Tensor:
    """Add relative Gaussian noise to scores."""

    noise = torch.randn(values.shape, dtype=torch.float64, generator=generator)
    scale = values.to(dtype=torch.float64).abs().clamp_min(EPS) * float(relative)
    return values.to(dtype=torch.float64) + noise * scale


def change_point_stability(
    values: torch.Tensor,
    *,
    relatives: tuple[float, ...] = (0.001, 0.005, 0.01),
    repeats: int = 16,
    seed: int = 0,
) -> dict[str, Any]:
    """Displacement of the one-change rank under relative score noise."""

    base = one_change_point(torch.sort(values.reshape(-1), descending=True).values)
    base_rank = base["rank"]
    if base_rank is None:
        return {"base": base, "mean_abs_shift": float("nan"), "within_two": float("nan")}
    shifts: list[int] = []
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    for relative in relatives:
        for _ in range(int(repeats)):
            noisy = perturb_scores(values.reshape(-1), relative, generator)
            ranked = torch.sort(noisy, descending=True).values
            found = one_change_point(ranked)["rank"]
            if found is None:
                continue
            shifts.append(abs(int(found) - int(base_rank)))
    if not shifts:
        return {"base": base, "mean_abs_shift": float("nan"), "within_two": float("nan")}
    tensor = torch.tensor(shifts, dtype=torch.float64)
    return {
        "base": base,
        "mean_abs_shift": float(tensor.mean().item()),
        "median_abs_shift": float(tensor.median().item()),
        "within_two": float((tensor <= 2).to(dtype=torch.float64).mean().item()),
        "repeats": len(shifts),
    }


def prefix_jaccard_stability(
    values: torch.Tensor,
    keep_frac: float,
    *,
    relative: float = 0.005,
    repeats: int = 16,
    seed: int = 0,
) -> float:
    """Mean Jaccard of a top-k id set under score perturbation."""

    scores = values.reshape(-1).to(dtype=torch.float64)
    n = int(scores.numel())
    k = max(1, int(round(keep_frac * n)))
    base = set(torch.argsort(scores, descending=True, stable=True)[:k].tolist())
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    overlaps: list[float] = []
    for _ in range(int(repeats)):
        noisy = perturb_scores(scores, relative, generator)
        other = set(torch.argsort(noisy, descending=True, stable=True)[:k].tolist())
        overlaps.append(jaccard(base, other))
    return float(sum(overlaps) / max(1, len(overlaps)))


def load_ranking_cache(path: Path) -> dict[str, Any]:
    """Load a HARP ranking cache from disk."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("purpose") != "harp_channel_ranking":
        raise ValueError(f"Unexpected ranking purpose: {payload.get('purpose')}")
    return payload


def build_layer_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Build one Layer-SP rank row per MoE layer."""

    table = payload["table"]
    layer_ids = sorted(int(key) for key in table)
    scores = torch.tensor([float(value) for value in payload["layer_scores"]], dtype=torch.float64)
    order = torch.argsort(scores, descending=True, stable=True)
    ranks = ranks_descending(scores)
    z_all = ordinary_z(scores.unsqueeze(0)).reshape(-1)
    rz_all = robust_z(scores.unsqueeze(0)).reshape(-1)
    mean = float(scores.mean().item())
    std = float(scores.std(unbiased=False).item())
    rows: list[dict[str, Any]] = []
    for index, layer_id in enumerate(layer_ids):
        expert = table[layer_id]["expert_structural_scores"].to(dtype=torch.float64)
        e_mean = float(expert.mean().item())
        e_std = float(expert.std(unbiased=False).item())
        rows.append({
            "model": str(payload.get("model_family") or payload.get("model_path")),
            "layer_id": int(layer_id),
            "rank": int(ranks[index].item()),
            "percentile": float(percentile_from_rank(ranks[index], int(scores.numel())).item()),
            "score": float(scores[index].item()),
            "robust_z": float(rz_all[index].item()),
            "ordinary_z": float(z_all[index].item()),
            "is_2sigma_candidate": bool(float(scores[index].item()) > mean + 2.0 * std),
            "is_robust_outlier_candidate": bool(float(rz_all[index].item()) > 2.0),
            "mean_expert_score": e_mean,
            "expert_score_std": e_std,
            "expert_score_cv": e_std / abs(e_mean) if abs(e_mean) > EPS else float("nan"),
        })
    rows.sort(key=lambda row: row["rank"])
    _ = order
    return rows


def build_expert_rows(payload: dict[str, Any], layer_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build Expert-SP rank rows with parent layer ranks and channel diagnostics."""

    layer_rank = {int(row["layer_id"]): int(row["rank"]) for row in layer_rows}
    table = payload["table"]
    rows: list[dict[str, Any]] = []
    for layer_id in sorted(int(key) for key in table):
        expert = table[layer_id]["expert_structural_scores"].to(dtype=torch.float64)
        channel = finite_scores(table[layer_id]["channel_scores"])
        n_experts = int(expert.numel())
        ranks = ranks_descending(expert)
        z_vals = ordinary_z(expert.unsqueeze(0)).reshape(-1)
        rz_vals = robust_z(expert.unsqueeze(0)).reshape(-1)
        ch_z = ordinary_z(channel)
        ch_mean = channel.mean(dim=1)
        ch_std = channel.std(dim=1, unbiased=False)
        top1 = ch_z.max(dim=1).values
        top5 = torch.topk(ch_z, k=min(5, int(channel.shape[1])), dim=1).values.mean(dim=1)
        mass = participation_mass(channel)
        mass_sum = mass.sum(dim=1, keepdim=True).clamp_min(EPS)
        ranked_mass, _ = torch.sort(mass, dim=1, descending=True)
        top10_n = max(1, int(round(0.10 * int(channel.shape[1]))))
        top10_mass = ranked_mass[:, :top10_n].sum(dim=1) / mass_sum.reshape(-1)
        for expert_id in range(n_experts):
            mean_e = float(expert.mean().item())
            rows.append({
                "layer_id": int(layer_id),
                "layer_rank": int(layer_rank[int(layer_id)]),
                "expert_id": int(expert_id),
                "rank_in_layer": int(ranks[expert_id].item()),
                "percentile_in_layer": float(percentile_from_rank(ranks[expert_id], n_experts).item()),
                "score": float(expert[expert_id].item()),
                "ordinary_z_in_layer": float(z_vals[expert_id].item()),
                "robust_z_in_layer": float(rz_vals[expert_id].item()),
                "is_2sigma_candidate": bool(float(z_vals[expert_id].item()) > 2.0),
                "is_robust_outlier_candidate": bool(float(rz_vals[expert_id].item()) > 2.0),
                "channel_mean": float(ch_mean[expert_id].item()),
                "channel_std": float(ch_std[expert_id].item()),
                "channel_cv": float((ch_std[expert_id] / ch_mean[expert_id].abs().clamp_min(EPS)).item()),
                "channel_top1_z": float(top1[expert_id].item()),
                "channel_top5_mean_z": float(top5[expert_id].item()),
                "channel_top10_mass": float(top10_mass[expert_id].item()),
            })
        _ = mean_e
    return rows


def build_channel_payload(payload: dict[str, Any], expert_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build compact per-layer Channel-SP rank tensors."""

    expert_rank = {(int(row["layer_id"]), int(row["expert_id"])): int(row["rank_in_layer"]) for row in expert_rows}
    layer_rank = {int(row["layer_id"]): int(row["layer_rank"]) for row in expert_rows}
    table = payload["table"]
    packed: dict[str, Any] = {}
    for layer_id in sorted(int(key) for key in table):
        channel = finite_scores(table[layer_id]["channel_scores"])
        ranked_idx = table[layer_id]["ranked_indices"].long()
        ranks = ranks_descending(channel)
        z_vals = ordinary_z(channel)
        rz_vals = robust_z(channel)
        mass = participation_mass(channel)
        norm = mass / mass.sum(dim=1, keepdim=True).clamp_min(EPS)
        n_experts, n_channels = int(channel.shape[0]), int(channel.shape[1])
        expert_ranks = torch.tensor(
            [expert_rank[(int(layer_id), expert_id)] for expert_id in range(n_experts)],
            dtype=torch.long,
        )
        packed[str(layer_id)] = {
            "layer_rank": int(layer_rank[int(layer_id)]),
            "scores": channel.to(dtype=torch.float32),
            "ranked_indices": ranked_idx,
            "rank_in_expert": ranks.to(dtype=torch.int16),
            "z_in_expert": z_vals.to(dtype=torch.float32),
            "robust_z_in_expert": rz_vals.to(dtype=torch.float32),
            "is_2sigma_candidate": (z_vals > 2.0),
            "is_3sigma_candidate": (z_vals > 3.0),
            "participation_mass": mass.to(dtype=torch.float32),
            "normalized_participation_mass": norm.to(dtype=torch.float32),
            "expert_rank_in_layer": expert_ranks,
            "n_experts": n_experts,
            "n_channels": n_channels,
        }
    return packed


def curve_table(sorted_desc: torch.Tensor) -> list[dict[str, float]]:
    """Rank / score / gap / cumulative-mass rows for a descending sequence."""

    values = sorted_desc.to(dtype=torch.float64).reshape(-1)
    n = int(values.numel())
    gaps = torch.cat([adjacent_gaps(values), torch.zeros(1, dtype=torch.float64)])
    logs = torch.cat([log_gaps(values), torch.zeros(1, dtype=torch.float64)])
    mass = participation_mass(values)
    cum = torch.cumsum(mass, dim=0) / mass.sum().clamp_min(EPS)
    rows = []
    for index in range(n):
        rows.append({
            "rank": float(index + 1),
            "score": float(values[index].item()),
            "gap": float(gaps[index].item()),
            "log_gap": float(logs[index].item()),
            "cum_mass": float(cum[index].item()),
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    """Write a list of dict rows as CSV."""

    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    names = fieldnames or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def special_layer_sets(layer_rows: list[dict[str, Any]], change: dict[str, Any]) -> dict[str, Any]:
    """Fixed-prefix, z-score, and change-point layer candidate sets."""

    n = len(layer_rows)
    by_rank = {int(row["rank"]): int(row["layer_id"]) for row in layer_rows}

    def prefix(frac: float) -> list[int]:
        k = max(1, int(math.ceil(frac * n)))
        return [by_rank[rank] for rank in range(1, k + 1)]

    z2 = [int(row["layer_id"]) for row in layer_rows if row["is_2sigma_candidate"]]
    robust = [int(row["layer_id"]) for row in layer_rows if row["is_robust_outlier_candidate"]]
    head_rank = change["base"]["rank"] if change.get("base") else None
    head = prefix(head_rank / n) if head_rank else []
    sets = {
        "top5": prefix(0.05),
        "top10": prefix(0.10),
        "top25": prefix(0.25),
        "z2": z2,
        "robust_z2": robust,
        "change_head": head,
    }
    sets["strong"] = sorted(set(sets["top10"]) & (set(z2) | set(robust) | set(head)))
    return sets


def special_expert_summary(expert_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-layer expert special-set summary."""

    from collections import defaultdict

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in expert_rows:
        grouped[int(row["layer_id"])].append(row)
    summary = []
    for layer_id, rows in sorted(grouped.items()):
        n = len(rows)
        ranked = sorted(rows, key=lambda item: int(item["rank_in_layer"]))
        scores = torch.tensor([float(item["score"]) for item in ranked], dtype=torch.float64)
        change = one_change_point(scores)
        k5 = max(1, int(math.ceil(0.05 * n)))
        k10 = max(1, int(math.ceil(0.10 * n)))
        k25 = max(1, int(math.ceil(0.25 * n)))
        z2 = [int(item["expert_id"]) for item in rows if item["is_2sigma_candidate"]]
        robust = [int(item["expert_id"]) for item in rows if item["is_robust_outlier_candidate"]]
        head_k = int(change["rank"] or k10)
        concentrated = [
            int(item["expert_id"])
            for item in rows
            if float(item["channel_top10_mass"]) > 0.20 and int(item["rank_in_layer"]) <= k25
        ]
        summary.append({
            "layer_id": int(layer_id),
            "n_experts": n,
            "change_rank": change["rank"],
            "natural_boundary": bool(change["natural_boundary"]),
            "change_stat": change["stat"],
            "n_top5": k5,
            "n_top10": k10,
            "n_top25": k25,
            "n_z2": len(z2),
            "n_robust_z2": len(robust),
            "n_concentrated_head": len(concentrated),
            "head_k": head_k,
            "mean_top10_mass": float(sum(float(item["channel_top10_mass"]) for item in rows) / n),
            "cohens_d10": cohens_d_prefix(scores, 0.10),
            "cohens_d25": cohens_d_prefix(scores, 0.25),
            "cohens_d50": cohens_d_prefix(scores, 0.50),
        })
    return summary


def special_channel_summary(channel_payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-expert channel special-set summary."""

    rows: list[dict[str, Any]] = []
    for layer_key, packed in channel_payload.items():
        layer_id = int(layer_key)
        scores = packed["scores"].to(dtype=torch.float64)
        z_vals = packed["z_in_expert"]
        rz_vals = packed["robust_z_in_expert"]
        norm = packed["normalized_participation_mass"]
        n_experts = int(packed["n_experts"])
        n_channels = int(packed["n_channels"])
        for expert_id in range(n_experts):
            ranked = torch.sort(scores[expert_id], descending=True).values
            change = one_change_point(ranked)
            k1 = max(1, int(math.ceil(0.01 * n_channels)))
            mass_head = float(torch.sort(norm[expert_id], descending=True).values[:k1].sum().item())
            rows.append({
                "layer_id": layer_id,
                "expert_id": expert_id,
                "expert_rank_in_layer": int(packed["expert_rank_in_layer"][expert_id].item()),
                "n_channels": n_channels,
                "change_rank": change["rank"],
                "natural_boundary": bool(change["natural_boundary"]),
                "n_z2": int((z_vals[expert_id] > 2.0).sum().item()),
                "n_z3": int((z_vals[expert_id] > 3.0).sum().item()),
                "n_robust_z2": int((rz_vals[expert_id] > 2.0).sum().item()),
                "top1_mass": mass_head,
                "top10_mass": float(torch.sort(norm[expert_id], descending=True).values[:max(1, int(round(0.10 * n_channels)))].sum().item()),
                "cohens_d10": cohens_d_prefix(ranked, 0.10),
                "cohens_d25": cohens_d_prefix(ranked, 0.25),
                "cohens_d50": cohens_d_prefix(ranked, 0.50),
            })
    return rows


def hierarchy_links(
    layer_rows: list[dict[str, Any]],
    expert_rows: list[dict[str, Any]],
    channel_summary: list[dict[str, Any]],
) -> dict[str, Any]:
    """Channel->expert and expert->layer correlations plus attribution labels."""

    from collections import defaultdict

    layer_score = {int(row["layer_id"]): float(row["score"]) for row in layer_rows}
    layer_z = {int(row["layer_id"]): float(row["ordinary_z"]) for row in layer_rows}
    expert_by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in expert_rows:
        expert_by_layer[int(row["layer_id"])].append(row)
    ch_by_expert = {(int(row["layer_id"]), int(row["expert_id"])): row for row in channel_summary}

    expert_score = torch.tensor([float(row["score"]) for row in expert_rows], dtype=torch.float64)
    top10_mass = torch.tensor([float(row["channel_top10_mass"]) for row in expert_rows], dtype=torch.float64)
    top1_z = torch.tensor([float(row["channel_top1_z"]) for row in expert_rows], dtype=torch.float64)
    ch_cv = torch.tensor([float(row["channel_cv"]) for row in expert_rows], dtype=torch.float64)
    layer_of_expert = torch.tensor([float(layer_score[int(row["layer_id"])]) for row in expert_rows], dtype=torch.float64)
    n_z2_ch = torch.tensor(
        [float(ch_by_expert[(int(row["layer_id"]), int(row["expert_id"]))]["n_z2"]) for row in expert_rows],
        dtype=torch.float64,
    )
    head_len = torch.tensor(
        [float(ch_by_expert[(int(row["layer_id"]), int(row["expert_id"]))]["change_rank"] or 0) for row in expert_rows],
        dtype=torch.float64,
    )

    def residual(target: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        xc = control - control.mean()
        yc = target - target.mean()
        beta = (xc * yc).sum() / xc.square().sum().clamp_min(EPS)
        return yc - beta * xc

    c1 = {
        "pearson_expert_vs_top10_mass": pearson(expert_score, top10_mass),
        "spearman_expert_vs_top10_mass": spearman(expert_score, top10_mass),
        "pearson_expert_vs_channel_top1_z": pearson(expert_score, top1_z),
        "spearman_expert_vs_channel_top1_z": spearman(expert_score, top1_z),
        "pearson_expert_vs_channel_cv": pearson(expert_score, ch_cv),
        "pearson_expert_vs_n_z2_channels": pearson(expert_score, n_z2_ch),
        "pearson_expert_vs_channel_head_len": pearson(expert_score, head_len),
        "partial_pearson_expert_vs_top10_mass_ctrl_layer": pearson(residual(expert_score, layer_of_expert), residual(top10_mass, layer_of_expert)),
        "partial_spearman_expert_vs_top10_mass_ctrl_layer": spearman(residual(expert_score, layer_of_expert), residual(top10_mass, layer_of_expert)),
    }

    layer_ids = sorted(layer_score)
    mean_e = []
    median_e = []
    top1_e = []
    frac_z2 = []
    mean_top10 = []
    loo_z = []
    labels = []
    for layer_id in layer_ids:
        rows = expert_by_layer[layer_id]
        scores = torch.tensor([float(item["score"]) for item in rows], dtype=torch.float64)
        mean_e.append(float(scores.mean().item()))
        median_e.append(float(scores.median().item()))
        top1_e.append(float(scores.max().item()))
        frac_z2.append(float(sum(1 for item in rows if item["is_2sigma_candidate"]) / max(1, len(rows))))
        mean_top10.append(float(sum(float(item["channel_top10_mass"]) for item in rows) / max(1, len(rows))))
        others = torch.tensor(
            [float(layer_score[other]) for other in layer_ids if other != layer_id],
            dtype=torch.float64,
        )
        rest_mean = float((scores.sum() - scores.max()) / max(1, int(scores.numel()) - 1))
        loo = float((rest_mean - float(others.mean().item())) / float(others.std(unbiased=False).clamp_min(EPS).item()))
        full_z = layer_z[layer_id]
        loo_z.append(loo)
        if abs(full_z) > 2.0 and abs(loo) < 0.5 * abs(full_z) and abs(loo) < 2.0:
            label = "outlier-driven"
        elif frac_z2[-1] >= 0.25 or (abs(full_z) <= 2.0 and abs(loo - full_z) < 0.5):
            label = "distributed-shift"
        else:
            label = "mixed"
        if abs(full_z) <= 1.0 and frac_z2[-1] < 0.10:
            label = "distributed-shift"
        labels.append({"layer_id": int(layer_id), "full_z": full_z, "loo_z_without_top_expert": loo, "frac_expert_z2": frac_z2[-1], "driver": label})

    mean_t = torch.tensor(mean_e, dtype=torch.float64)
    layer_t = torch.tensor([layer_score[lid] for lid in layer_ids], dtype=torch.float64)
    c2 = {
        "pearson_layer_vs_expert_mean": pearson(layer_t, mean_t),
        "spearman_layer_vs_expert_mean": spearman(layer_t, mean_t),
        "pearson_layer_vs_expert_median": pearson(layer_t, torch.tensor(median_e, dtype=torch.float64)),
        "pearson_layer_vs_expert_top1": pearson(layer_t, torch.tensor(top1_e, dtype=torch.float64)),
        "pearson_layer_vs_frac_expert_z2": pearson(layer_t, torch.tensor(frac_z2, dtype=torch.float64)),
        "pearson_layer_vs_mean_channel_top10_mass": pearson(layer_t, torch.tensor(mean_top10, dtype=torch.float64)),
        "attribution": labels,
    }
    counts = {"outlier-driven": 0, "distributed-shift": 0, "mixed": 0}
    for item in labels:
        counts[str(item["driver"])] += 1
    c2["driver_counts"] = counts
    return {"channel_to_expert": c1, "expert_to_layer": c2}


def null_controls(
    layer_rows: list[dict[str, Any]],
    expert_rows: list[dict[str, Any]],
    *,
    repeats: int = 32,
    seed: int = 0,
) -> dict[str, Any]:
    """Perturbation, within-parent permutation, and aggregation controls."""

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    layer_scores = torch.tensor([float(row["score"]) for row in sorted(layer_rows, key=lambda item: int(item["layer_id"]))], dtype=torch.float64)
    layer_ids = [int(row["layer_id"]) for row in sorted(layer_rows, key=lambda item: int(item["layer_id"]))]
    observed_tau = 1.0
    perturb = {}
    for frac, name in ((0.05, "top5"), (0.10, "top10"), (0.25, "top25")):
        perturb[name] = prefix_jaccard_stability(layer_scores, frac, relative=0.005, repeats=repeats, seed=seed)
    kendalls = []
    for relative in (0.001, 0.005, 0.01):
        noisy = perturb_scores(layer_scores, relative, generator)
        kendalls.append(kendall_tau(layer_scores, noisy))
    from collections import defaultdict
    by_layer: dict[int, list[float]] = defaultdict(list)
    for row in expert_rows:
        by_layer[int(row["layer_id"])].append(float(row["score"]))
    mean_e = torch.tensor([float(sum(by_layer[lid]) / len(by_layer[lid])) for lid in layer_ids], dtype=torch.float64)
    observed = pearson(layer_scores, mean_e)
    null_corr = []
    n_layers = int(layer_scores.numel())
    for _ in range(int(repeats)):
        perm = torch.randperm(n_layers, generator=generator)
        null_corr.append(pearson(layer_scores, mean_e[perm]))
    expert_scores = torch.tensor([float(row["score"]) for row in expert_rows], dtype=torch.float64)
    top10 = torch.tensor([float(row["channel_top10_mass"]) for row in expert_rows], dtype=torch.float64)
    observed_c1 = spearman(expert_scores, top10)
    null_c1 = []
    for _ in range(int(repeats)):
        perm = torch.randperm(int(top10.numel()), generator=generator)
        null_c1.append(spearman(expert_scores, top10[perm]))
    return {
        "layer_prefix_jaccard_0.5pct": perturb,
        "layer_kendall_tau_perturb": {
            "0.1pct": kendalls[0],
            "0.5pct": kendalls[1],
            "1pct": kendalls[2],
            "mean": float(sum(kendalls) / len(kendalls)),
        },
        "expert_mean_to_layer": {
            "observed_pearson": observed,
            "null_mean": float(sum(null_corr) / len(null_corr)),
            "null_std": float(torch.tensor(null_corr).std(unbiased=False).item()),
            "excess": float(observed - sum(null_corr) / len(null_corr)),
        },
        "channel_top10_to_expert": {
            "observed_spearman": observed_c1,
            "null_mean": float(sum(null_c1) / len(null_c1)),
            "null_std": float(torch.tensor(null_c1).std(unbiased=False).item()),
            "excess": float(observed_c1 - sum(null_c1) / len(null_c1)),
        },
        "observed_tau_placeholder": observed_tau,
        "repeats": int(repeats),
    }


def plot_layer_curve(curve: list[dict[str, float]], path: Path, title: str) -> None:
    """Save a Layer-SP ranked score / gap figure."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ranks = [row["rank"] for row in curve]
    scores = [row["score"] for row in curve]
    gaps = [row["gap"] for row in curve]
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(ranks, scores, color="#1f4e79")
    axes[0].set_ylabel("Layer-SP")
    axes[0].set_title(title)
    axes[1].plot(ranks[:-1], gaps[:-1], color="#8b1e1e")
    axes[1].set_ylabel("Adjacent gap")
    axes[1].set_xlabel("Rank (1 = highest Layer-SP)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def decide_allocation(layer_seg: dict[str, Any], nulls: dict[str, Any], links: dict[str, Any]) -> str:
    """Map evidence onto a HARP budget-tier decision."""

    natural = bool(layer_seg.get("natural_boundary"))
    stable = float(layer_seg.get("within_two") or 0.0) >= 0.75
    excess = float(nulls["expert_mean_to_layer"]["excess"])
    if natural and stable and excess > 0.2:
        return "use segment"
    if excess > 0.2:
        return "use quantile only"
    return "no reliable segment"


def conclusion_block(
    model: str,
    layer_seg: dict[str, Any],
    expert_summary: list[dict[str, Any]],
    channel_summary: list[dict[str, Any]],
    links: dict[str, Any],
    nulls: dict[str, Any],
) -> str:
    """Format the section-11 conclusion for one model."""

    natural_e = sum(1 for row in expert_summary if row["natural_boundary"])
    natural_c = sum(1 for row in channel_summary if row["natural_boundary"])
    driver_counts = links["expert_to_layer"]["driver_counts"]
    majority = max(driver_counts, key=lambda key: driver_counts[key])
    decision = decide_allocation(layer_seg, nulls, links)
    return "\n".join([
        f"Model: {model}",
        (
            f"Layer segments: head_rank={layer_seg.get('rank')}; "
            f"natural={layer_seg.get('natural_boundary')}; "
            f"stability_within_two={layer_seg.get('within_two')}; "
            f"driver={majority}"
        ),
        f"Expert segments: {natural_e}/{len(expert_summary)} layers have a natural head boundary",
        f"Channel segments: {natural_c}/{len(channel_summary)} experts have a natural head boundary",
        (
            f"Channel -> Expert: spearman_top10_mass="
            f"{links['channel_to_expert']['spearman_expert_vs_top10_mass']:.3f}; "
            f"null_excess={nulls['channel_top10_to_expert']['excess']:.3f}"
        ),
        (
            f"Expert -> Layer: pearson_mean="
            f"{links['expert_to_layer']['pearson_layer_vs_expert_mean']:.3f}; "
            f"null_excess={nulls['expert_mean_to_layer']['excess']:.3f}"
        ),
        f"Decision: {decision}",
    ])


def analyze_model(cache_path: Path, output_dir: Path, model_key: str) -> dict[str, Any]:
    """Run the full ranking-structure pipeline for one HARP cache."""

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = load_ranking_cache(cache_path)
    layer_rows = build_layer_rows(payload)
    expert_rows = build_expert_rows(payload, layer_rows)
    channel_payload = build_channel_payload(payload, expert_rows)
    layer_sorted = torch.tensor([float(row["score"]) for row in sorted(layer_rows, key=lambda item: int(item["rank"]))], dtype=torch.float64)
    layer_curve = curve_table(layer_sorted)
    layer_stab = change_point_stability(
        torch.tensor([float(row["score"]) for row in sorted(layer_rows, key=lambda item: int(item["layer_id"]))], dtype=torch.float64),
        seed=0,
    )
    layer_seg = {**layer_stab["base"], "within_two": layer_stab.get("within_two"), "mean_abs_shift": layer_stab.get("mean_abs_shift")}
    expert_summary = special_expert_summary(expert_rows)
    channel_summary = special_channel_summary(channel_payload)
    special_layers = special_layer_sets(layer_rows, layer_stab)
    links = hierarchy_links(layer_rows, expert_rows, channel_summary)
    nulls = null_controls(layer_rows, expert_rows, repeats=32, seed=0)

    write_csv(output_dir / "layer_rank_curve.csv", layer_curve)
    write_csv(output_dir / "layer_gap_curve.csv", layer_curve, ["rank", "gap", "log_gap"])
    write_csv(output_dir / "expert_segment_summary.csv", expert_summary)
    write_csv(output_dir / "channel_segment_summary.csv", channel_summary)
    natural_c = [row for row in channel_summary if row["natural_boundary"]]
    change_ranks = [int(row["change_rank"] or 0) for row in channel_summary]
    (output_dir / "channel_segments.json").write_text(
        json.dumps(
            {
                "n_experts": len(channel_summary),
                "n_natural_head": len(natural_c),
                "frac_natural_head": len(natural_c) / max(1, len(channel_summary)),
                "mean_change_rank": float(sum(change_ranks) / max(1, len(change_ranks))),
                "median_change_rank": float(sorted(change_ranks)[len(change_ranks) // 2] if change_ranks else 0),
                "mean_cohens_d10": float(sum(float(row["cohens_d10"]) for row in channel_summary) / max(1, len(channel_summary))),
                "mean_top10_mass": float(sum(float(row["top10_mass"]) for row in channel_summary) / max(1, len(channel_summary))),
                "note": "Per-expert channel CSVs omitted; canonical ranks are in channel_rank.pt.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    plot_layer_curve(layer_curve, output_dir / "layer_rank_curve.png", f"{model_key} Layer-SP ranked")

    expert_curve_dir = output_dir / "expert_rank_curves"
    expert_curve_dir.mkdir(parents=True, exist_ok=True)
    from collections import defaultdict
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in expert_rows:
        grouped[int(row["layer_id"])].append(row)
    for layer_id, rows in grouped.items():
        ranked = sorted(rows, key=lambda item: int(item["rank_in_layer"]))
        scores = torch.tensor([float(item["score"]) for item in ranked], dtype=torch.float64)
        write_csv(expert_curve_dir / f"{layer_id}.csv", curve_table(scores))

    channel_curve_dir = output_dir / "channel_rank_curves"
    channel_curve_dir.mkdir(parents=True, exist_ok=True)
    pooled: list[torch.Tensor] = []
    for layer_key, packed in channel_payload.items():
        scores = packed["scores"].to(dtype=torch.float64)
        ranked = torch.sort(scores, dim=1, descending=True).values
        mean_curve = ranked.mean(dim=0)
        write_csv(channel_curve_dir / f"layer_{layer_key}_mean.csv", curve_table(mean_curve))
        pooled.append(mean_curve)
    write_csv(channel_curve_dir / "pooled_mean.csv", curve_table(torch.stack(pooled).mean(dim=0)))

    torch.save({"rows": layer_rows}, output_dir / "layer_rank.pt")
    (output_dir / "layer_rank.json").write_text(json.dumps(layer_rows, indent=2), encoding="utf-8")
    torch.save({"rows": expert_rows}, output_dir / "expert_rank.pt")
    torch.save(channel_payload, output_dir / "channel_rank.pt")
    manifest = {
        "model": model_key,
        "cache_path": str(cache_path),
        "cache_sha256": file_sha256(cache_path),
        "model_path": str(payload.get("model_path")),
        "model_family": payload.get("model_family"),
        "model_provenance": payload.get("model_provenance"),
        "n_layers": len(layer_rows),
        "n_experts": len(expert_rows),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "layer_segments.json").write_text(json.dumps(layer_seg, indent=2), encoding="utf-8")
    (output_dir / "expert_segments.json").write_text(json.dumps(expert_summary, indent=2), encoding="utf-8")
    (output_dir / "segment_analysis.json").write_text(
        json.dumps({"layer": layer_seg, "special_layers": special_layers, "expert_layers_with_natural_head": sum(1 for row in expert_summary if row["natural_boundary"])}, indent=2),
        encoding="utf-8",
    )
    (output_dir / "hierarchy_links.json").write_text(json.dumps(links, indent=2), encoding="utf-8")
    (output_dir / "null_controls.json").write_text(json.dumps(nulls, indent=2), encoding="utf-8")
    conclusion = conclusion_block(model_key, layer_seg, expert_summary, channel_summary, links, nulls)
    (output_dir / "conclusion.txt").write_text(conclusion + "\n", encoding="utf-8")
    return {"manifest": manifest, "conclusion": conclusion, "layer_seg": layer_seg, "links": links, "nulls": nulls, "special_layers": special_layers}
