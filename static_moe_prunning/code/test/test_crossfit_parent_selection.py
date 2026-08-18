from __future__ import annotations

import argparse
import json

import pytest
import torch

from scripts import build_train_selection_token_cache
from scripts import select_crossfit_parent
from src.corpus_ppl import validate_token_cache_payload
from src.crossfit_parent_selection import (
    validate_disjoint_token_ranges,
    select_stable_candidate,
)


def test_validate_disjoint_token_ranges_rejects_calibration_overlap() -> None:
    with pytest.raises(ValueError, match="overlaps calibration"):
        validate_disjoint_token_ranges(
            calibration_range=(0, 262144),
            selection_ranges=[(250000, 282768)],
        )


def test_validate_disjoint_token_ranges_rejects_fold_overlap() -> None:
    with pytest.raises(ValueError, match="overlap each other"):
        validate_disjoint_token_ranges(
            calibration_range=(0, 262144),
            selection_ranges=[(300000, 332768), (320000, 352768)],
        )


def test_stable_selector_enables_refinement_after_majority_wins() -> None:
    result = select_stable_candidate(
        fold_ppl={
            "route_tail": [10.0, 10.1, 9.9, 10.2],
            "tail_risk": [9.8, 9.9, 9.95, 10.0],
        },
        fallback="route_tail",
    )

    assert result["selected"] == "tail_risk"
    assert result["selection_reason"] == "stable_majority_and_lower_mean"
    assert result["win_counts"] == {"route_tail": 1, "tail_risk": 3}


def test_stable_selector_falls_back_when_refinement_is_unstable() -> None:
    result = select_stable_candidate(
        fold_ppl={
            "route_tail": [10.0, 10.2, 9.8, 10.1],
            "tail_risk": [9.9, 10.3, 9.7, 10.2],
        },
        fallback="route_tail",
    )

    assert result["selected"] == "route_tail"
    assert result["selection_reason"] == "fallback_no_stable_refinement"
    assert result["win_counts"] == {"route_tail": 2, "tail_risk": 2}


def test_stable_selector_falls_back_if_majority_candidate_has_worse_mean() -> None:
    result = select_stable_candidate(
        fold_ppl={
            "route_tail": [10.0, 10.0, 10.0, 10.0],
            "tail_risk": [9.9, 9.9, 9.9, 20.0],
        },
        fallback="route_tail",
    )

    assert result["selected"] == "route_tail"
    assert result["selection_reason"] == "fallback_no_stable_refinement"


def test_stable_selector_requires_equal_nonempty_fold_counts() -> None:
    with pytest.raises(ValueError, match="same non-zero fold count"):
        select_stable_candidate(
            fold_ppl={"route_tail": [10.0, 10.1], "tail_risk": [9.9]},
            fallback="route_tail",
        )


def test_train_selection_cache_passes_frozen_corpus_validation(
    tmp_path, monkeypatch
) -> None:
    output_cache = tmp_path / "selection_fold.pt"
    args = argparse.Namespace(
        model_path="/models/test-model",
        output_cache=output_cache,
        dataset="test-dataset",
        config="raw-v1",
        split="train",
        text_field="text",
        arrow_file=[],
        sequence_length=2048,
        selection_windows=2,
        token_offset=300000,
        calibration_token_start=0,
        calibration_token_end=262144,
        row_batch_size=16,
        protocol_name="train_selection_regression_v1",
    )
    frozen_tokens = torch.arange(4096, dtype=torch.long).unsqueeze(0)
    source = {
        "dataset": "test-dataset",
        "config": "raw-v1",
        "split": "train",
    }

    monkeypatch.setattr(
        build_train_selection_token_cache, "parse_args", lambda: args
    )
    monkeypatch.setattr(
        build_train_selection_token_cache,
        "load_calibration_text_dataset",
        lambda **_kwargs: ([{"text": "unused"}], source),
    )
    monkeypatch.setattr(
        build_train_selection_token_cache.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        build_train_selection_token_cache,
        "collect_contiguous_text_tokens",
        lambda *_args, **_kwargs: (frozen_tokens, {"sha256": "a" * 64}),
    )
    monkeypatch.setattr(
        build_train_selection_token_cache,
        "build_model_cache_identity",
        lambda *_args, **_kwargs: {
            "tokenizer_sha256": "b" * 64,
            "tokenization_config_sha256": "c" * 64,
        },
    )

    assert build_train_selection_token_cache.main() == 0
    payload = torch.load(output_cache, map_location="cpu", weights_only=False)
    validated = validate_token_cache_payload(payload)

    assert payload["frozen_before_selection"] is True
    assert payload["frozen_before_evaluation"] is True
    assert len(payload["input_ids_sha256"]) == 64
    assert payload["attention_mask_semantics"] == "all_ones_no_padding"
    assert torch.equal(validated, frozen_tokens)


def test_crossfit_selector_rejects_manifest_profile_hash_mismatch(
    tmp_path, monkeypatch
) -> None:
    profile_path = tmp_path / "profile.pt"
    profile_path.write_bytes(b"frozen-profile")
    result_path = tmp_path / "fold.json"
    result_path.write_text(
        json.dumps(
            [
                {
                    "split": "train",
                    "ppl": 10.0,
                    "profile_path": str(profile_path.resolve()),
                    "profile_file_sha256": "0" * 64,
                    "profile_frozen_before_evaluation": True,
                    "test_metrics_used_for_profile": False,
                    "protocol_name": "selection_fold_v1",
                }
            ]
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "fallback": "route_tail",
                "candidates": {
                    "route_tail": {
                        "profile_path": str(profile_path.resolve()),
                        "profile_file_sha256": "0" * 64,
                        "result_paths": [str(result_path.resolve())],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        select_crossfit_parent,
        "parse_args",
        lambda: argparse.Namespace(
            manifest=manifest_path,
            output=tmp_path / "selection.json",
        ),
    )

    with pytest.raises(ValueError, match="profile SHA256"):
        select_crossfit_parent.main()
