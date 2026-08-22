from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open

from AIMER_Mix.build_aimer_mix_artifacts import main as build_base
from AIMER_Mix.tests.helpers import write_checkpoint
from AIMER_MIX_PLUS.build_aimer_mix_plus_artifacts import main as build_plus
from AIMER_MIX_PLUS.export_aimer_mix_plus_checkpoint import main as export_plus


def _build_base(tmp_path: Path, monkeypatch, family: str) -> tuple[Path, Path, int]:
    model_path = tmp_path / f"{family}-model"
    write_checkpoint(model_path, family)
    retained = {"qwen3": 64, "gemma4": 32, "qwen3.6": 64, "deepseek": 32}[family]
    artifact = tmp_path / f"{family}-artifact"
    base_cache = artifact / "aimer_mix.pt"
    base_profile = artifact / "base_profile.pt"
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_aimer_mix_artifacts",
            "--model-path",
            str(model_path),
            "--output-channel-cache",
            str(base_cache),
            "--output-profile",
            str(base_profile),
            "--retained-channels",
            str(retained),
        ],
    )
    assert build_base() == 0
    return model_path, base_cache, retained


def _write_source(base_cache: Path, source_path: Path, source_name: str) -> None:
    base = torch.load(base_cache, map_location="cpu", weights_only=True)
    tables = {}
    for layer_id, row in base["table"].items():
        reversed_order = row["ranked_indices"].flip(1)
        tables[layer_id] = {"ranked_indices": reversed_order}
    payload = {
        "schema_version": 1,
        "purpose": "aimer_mix_plus_pseudo_source",
        "model_path": base["model_path"],
        "calibration_sequences": 0,
        "test_metrics_used": False,
        "table": tables,
        "pseudo_source": {
            "name": source_name,
            "coverage": torch.ones(len(tables), base["architecture"]["num_experts"]),
            "stability": torch.ones(len(tables), base["architecture"]["num_experts"]),
        },
    }
    torch.save(payload, source_path)


def _build_plus(tmp_path: Path, monkeypatch, family: str) -> tuple[Path, Path, Path, int]:
    model_path, base_cache, retained = _build_base(tmp_path, monkeypatch, family)
    source_path = tmp_path / f"{family}-layerprop.pt"
    _write_source(base_cache, source_path, "layerprop")
    artifact = tmp_path / f"{family}-plus"
    cache = artifact / "rankings.pt"
    profile = artifact / "profile.pt"
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_aimer_mix_plus_artifacts",
            "--model-path",
            str(model_path),
            "--aimer-mix-cache",
            str(base_cache),
            "--source",
            f"layerprop={source_path}",
            "--output-channel-cache",
            str(cache),
            "--output-profile",
            str(profile),
            "--retained-channels",
            str(retained),
            "--boundary-fraction",
            "0.5",
            "--minimum-boundary-channels",
            "1",
            "--maximum-boundary-fraction",
            "0.5",
            "--pseudo-floor",
            "0.1",
        ],
    )
    assert build_plus() == 0
    return model_path, cache, profile, retained


def test_amp_artifact_is_width_specific_and_records_sources(tmp_path: Path, monkeypatch) -> None:
    _model_path, cache_path, profile_path, retained = _build_plus(tmp_path, monkeypatch, "gemma4")
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    profile = torch.load(profile_path, map_location="cpu", weights_only=True)
    assert cache["purpose"] == "aimer_mix_plus_ranking"
    assert cache["ranking_is_width_specific"] is True
    assert cache["retained_channels"] == retained
    assert [source["name"] for source in cache["aimer_mix_plus"]["sources"]] == ["layerprop"]
    assert profile["method"] == "aimer_mix_plus"
    assert profile["retained_channels"] == retained


def test_amp_export_supports_all_model_layouts(tmp_path: Path, monkeypatch) -> None:
    for family in ("qwen3", "gemma4", "qwen3.6", "deepseek"):
        family_root = tmp_path / family
        family_root.mkdir()
        model_path, cache_path, profile_path, retained = _build_plus(family_root, monkeypatch, family)
        output = family_root / "output"
        monkeypatch.setattr(
            "sys.argv",
            [
                "export_aimer_mix_plus_checkpoint",
                "--model-path",
                str(model_path),
                "--profile",
                str(profile_path),
                "--channel-cache",
                str(cache_path),
                "--output-dir",
                str(output),
            ],
        )
        assert export_plus() == 0
        manifest = json.loads((output / "pruning_export_manifest.json").read_text(encoding="utf-8"))
        assert manifest["method"] == "aimer_mix_plus"
        assert manifest["retained_channels"] == retained
        with safe_open(output / "model.safetensors", framework="pt", device="cpu") as handle:
            assert handle.keys()