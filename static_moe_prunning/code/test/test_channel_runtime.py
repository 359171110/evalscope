from __future__ import annotations

import torch

from src.channel_runtime import (
    _build_layer_channel_table_from_raw_scores,
    _channel_path_score,
)


def test_channel_calibration_builds_normalized_prefix_table() -> None:
    gamma = torch.ones(3)
    gate = torch.tensor([[3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
    up = gate.clone()
    down = torch.eye(3)

    raw = _channel_path_score(gamma, gate, up, down, eps=1.0e-8)
    table = _build_layer_channel_table_from_raw_scores(
        torch.stack((raw, raw.flip(0))),
        block_size=2,
    )

    assert table.ranked_indices.shape == (2, 3)
    assert table.block_relative_scores.shape == (2, 2)
    assert table.block_coverage_scores.shape == (2, 2)
    assert table.block_sizes.tolist() == [2, 1]
    assert torch.allclose(table.block_coverage_scores.sum(dim=1), torch.ones(2))
    assert torch.all(
        table.block_relative_scores[:, :-1]
        >= table.block_relative_scores[:, 1:]
    )