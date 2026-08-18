from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

import torch


DEFAULT_SOURCE_ORDER = ("wikitext", "gsm8k", "mbpp", "math")
DEFAULT_SPLIT_QUOTAS = {
    "fit": {"wikitext": 48, "gsm8k": 12, "mbpp": 24, "math": 12},
    "validation": {"wikitext": 24, "gsm8k": 6, "mbpp": 12, "math": 6},
    "audit": {"wikitext": 24, "gsm8k": 6, "mbpp": 12, "math": 6},
}
E1_SPLIT_QUOTAS = {
    "fit": {"wikitext": 128, "gsm8k": 32, "mbpp": 64, "math": 32},
    "validation": {"wikitext": 64, "gsm8k": 16, "mbpp": 32, "math": 16},
    "audit": {"wikitext": 64, "gsm8k": 16, "mbpp": 32, "math": 16},
}


def index_tensor_sha256(indices: torch.Tensor) -> str:
    """Hash a canonical int64 sequence-index tensor."""

    canonical = indices.detach().to(dtype=torch.int64, device="cpu").contiguous()
    return hashlib.sha256(canonical.numpy().tobytes(order="C")).hexdigest()


def build_stratified_split_indices(
    sequence_order: Sequence[str],
    *,
    seed: int = 42,
    quotas: Mapping[str, Mapping[str, int]] | None = None,
    source_order: Sequence[str] = DEFAULT_SOURCE_ORDER,
) -> dict[str, torch.Tensor]:
    """Build deterministic, disjoint sequence-index splits by source."""

    order = tuple(str(source) for source in source_order)
    if len(set(order)) != len(order):
        raise ValueError("source_order must not contain duplicate sources.")
    sequence_sources = [str(source) for source in sequence_order]
    unknown = sorted(set(sequence_sources) - set(order))
    if unknown:
        raise ValueError(f"sequence_order contains unknown sources: {unknown}")
    split_quotas = DEFAULT_SPLIT_QUOTAS if quotas is None else quotas
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    shuffled_by_source: dict[str, torch.Tensor] = {}
    for source in order:
        indices = torch.tensor(
            [index for index, value in enumerate(sequence_sources) if value == source],
            dtype=torch.int64,
        )
        shuffled_by_source[source] = indices[torch.randperm(indices.numel(), generator=generator)]

    offsets = {source: 0 for source in order}
    splits: dict[str, torch.Tensor] = {}
    used: list[torch.Tensor] = []
    for split_name, split_quota in split_quotas.items():
        pieces = []
        for source in order:
            count = int(split_quota.get(source, 0))
            if count < 0:
                raise ValueError(f"quota for {split_name}/{source} must be non-negative.")
            begin = offsets[source]
            end = begin + count
            available = int(shuffled_by_source[source].numel())
            if end > available:
                raise ValueError(
                    f"quota for {split_name}/{source} needs {count} items, "
                    f"but only {available - begin} remain."
                )
            pieces.append(shuffled_by_source[source][begin:end])
            offsets[source] = end
        indices = torch.cat(pieces).sort().values if pieces else torch.empty(0, dtype=torch.int64)
        if indices.numel() != int(sum(int(value) for value in split_quota.values())):
            raise ValueError(f"split {split_name} has an inconsistent quota.")
        splits[str(split_name)] = indices
        used.append(indices)

    if used:
        all_used = torch.cat(used)
        if int(torch.unique(all_used).numel()) != int(all_used.numel()):
            raise ValueError("split quotas produced overlapping sequence indices.")
    return splits


def select_representative_experts(
    route_counts: Mapping[int, torch.Tensor],
    *,
    layers: Sequence[int] = (0, 15, 31, 47),
    quantiles: Sequence[tuple[str, float]] = (("low", 0.10), ("medium", 0.50), ("high", 0.90)),
    per_stratum: int = 2,
) -> list[dict[str, int | str]]:
    """Select physical experts nearest to fixed route-count quantiles."""

    if int(per_stratum) <= 0:
        raise ValueError("per_stratum must be positive.")
    selected: list[dict[str, int | str]] = []
    for layer_idx in layers:
        if int(layer_idx) not in route_counts:
            raise KeyError(f"route_counts has no layer {layer_idx}.")
        counts = torch.as_tensor(route_counts[int(layer_idx)], dtype=torch.int64, device="cpu").flatten()
        if counts.numel() == 0 or bool((counts < 0).any()):
            raise ValueError(f"route_counts for layer {layer_idx} must be non-empty and non-negative.")
        used_ids: set[int] = set()
        for stratum, quantile in quantiles:
            if not 0.0 <= float(quantile) <= 1.0:
                raise ValueError(f"quantile for {stratum} must be in [0, 1].")
            target = torch.quantile(counts.to(torch.float64), float(quantile), interpolation="linear").item()
            candidates = sorted(
                range(int(counts.numel())),
                key=lambda expert_idx: (abs(int(counts[expert_idx]) - target), expert_idx),
            )
            picked = [expert_idx for expert_idx in candidates if expert_idx not in used_ids][: int(per_stratum)]
            if len(picked) != int(per_stratum):
                raise ValueError(f"not enough distinct experts for layer {layer_idx}/{stratum}.")
            used_ids.update(picked)
            for expert_idx in picked:
                selected.append(
                    {
                        "layer": int(layer_idx),
                        "expert": int(expert_idx),
                        "stratum": str(stratum),
                        "route_count": int(counts[expert_idx]),
                    }
                )
    return selected