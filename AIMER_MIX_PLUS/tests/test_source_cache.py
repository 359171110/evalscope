from __future__ import annotations

from pathlib import Path

import pytest
import torch

from AIMER_MIX_PLUS.source_cache import load_pseudo_source


def _write_cache(path: Path, model_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "purpose": "test_pseudo_ranking",
        "model_path": str(model_path),
        "calibration_sequences": 0,
        "test_metrics_used": False,
        "table": {
            1: {"ranked_indices": torch.tensor([[3, 2, 1, 0], [0, 1, 2, 3]])},
            2: {"ranked_indices": torch.tensor([[0, 2, 1, 3], [3, 1, 2, 0]])},
        },
        "naps": {
            "coverage": [
                {"layer_id": 1, "experts_covered": 1},
                {"layer_id": 2, "experts_covered": 2},
            ]
        },
    }
    torch.save(payload, path)


def test_load_ranking_only_source_and_layer_coverage(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    cache = tmp_path / "source.pt"
    _write_cache(cache, model_path)
    source = load_pseudo_source(
        name="layerprop",
        cache_path=cache,
        layer_ids=(1, 2),
        num_experts=2,
        channels=4,
        model_path=model_path,
        coverage_floor=0.4,
    )
    assert source.order.shape == (2, 2, 4)
    assert torch.allclose(source.coverage[0], torch.full((2,), 0.7))
    assert torch.allclose(source.coverage[1], torch.ones(2))


def test_reject_calibration_backed_source(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    cache = tmp_path / "source.pt"
    _write_cache(cache, model_path)
    payload = torch.load(cache, map_location="cpu", weights_only=True)
    payload["calibration_sequences"] = 1
    torch.save(payload, cache)
    with pytest.raises(ValueError, match="data-free"):
        load_pseudo_source(
            name="pp",
            cache_path=cache,
            layer_ids=(1, 2),
            num_experts=2,
            channels=4,
            model_path=model_path,
        )


def test_explicit_per_expert_confidence_overrides_layer_summary(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    cache = tmp_path / "source.pt"
    _write_cache(cache, model_path)
    payload = torch.load(cache, map_location="cpu", weights_only=True)
    payload["pseudo_source"] = {
        "coverage": torch.tensor([[0.2, 0.4], [0.6, 0.8]]),
        "stability": torch.tensor([[0.9, 0.7], [0.5, 0.3]]),
    }
    torch.save(payload, cache)
    source = load_pseudo_source(
        name="prp",
        cache_path=cache,
        layer_ids=(1, 2),
        num_experts=2,
        channels=4,
        model_path=model_path,
    )
    assert torch.allclose(source.coverage, torch.tensor([[0.2, 0.4], [0.6, 0.8]]))
    assert torch.allclose(source.stability, torch.tensor([[0.9, 0.7], [0.5, 0.3]]))
