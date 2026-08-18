from __future__ import annotations

import torch

from ramp_protocol import (
    E1_SPLIT_QUOTAS,
    build_stratified_split_indices,
    index_tensor_sha256,
    select_representative_experts,
)


def test_stratified_splits_are_deterministic_and_disjoint() -> None:
    sequence_order = ["wikitext", "gsm8k", "mbpp", "math"] * 3
    quotas = {
        "fit": {"wikitext": 1, "gsm8k": 1, "mbpp": 1, "math": 1},
        "validation": {"wikitext": 1, "gsm8k": 1, "mbpp": 1, "math": 1},
    }

    first = build_stratified_split_indices(sequence_order, quotas=quotas)
    second = build_stratified_split_indices(sequence_order, quotas=quotas)

    assert all(torch.equal(first[name], second[name]) for name in first)
    assert torch.unique(torch.cat(list(first.values()))).numel() == 8
    assert len(index_tensor_sha256(first["fit"])) == 64


def test_representative_experts_use_physical_ids_and_deterministic_ties() -> None:
    route_counts = {0: torch.tensor([0, 10, 20, 30, 40, 50])}

    selected = select_representative_experts(
        route_counts,
        layers=(0,),
        quantiles=(("low", 0.0), ("high", 1.0)),
        per_stratum=2,
    )

    assert [(item["expert"], item["route_count"]) for item in selected] == [
        (0, 0),
        (1, 10),
        (5, 50),
        (4, 40),
    ]


def test_e1_split_quotas_cover_all_512_sequences() -> None:
    assert sum(sum(values.values()) for values in E1_SPLIT_QUOTAS.values()) == 512