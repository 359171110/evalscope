from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

import torch

from scripts import build_official_reap_profile
from src.calibration_data import token_tensor_sha256


class _FakeObserver:
    def __init__(self, model, hook_config) -> None:
        self.model = model
        self.hook_config = hook_config

    def record_all_blocks(self, data_batches, batch_group_size):
        assert len(data_batches) == 2
        assert batch_group_size == 1
        assert self.hook_config.renormalize_router_weights is True
        return {0: {"reap": torch.tensor([0.4, 0.1, 0.3, 0.2])}}


class _FakeHookConfig:
    def __init__(self, renormalize_router_weights, record_pruning_metrics_only) -> None:
        self.renormalize_router_weights = renormalize_router_weights
        self.record_pruning_metrics_only = record_pruning_metrics_only


class FakeQwen3:
    pass


def test_official_reap_builder_uses_shared_tokens_and_pins_commit(monkeypatch, tmp_path) -> None:
    reap_root = tmp_path / "reap"
    (reap_root / "src").mkdir(parents=True)
    model_path = tmp_path / "model"
    model_path.mkdir()
    calibration_path = tmp_path / "calibration.pt"
    channel_path = tmp_path / "channel.pt"
    observer_path = tmp_path / "observer.pt"
    profile_path = tmp_path / "profile.pt"
    tokens = torch.arange(2048, dtype=torch.long).view(1, -1)
    torch.save(
        {
            "schema_version": 1,
            "purpose": "shared_moe_pruning_calibration",
            "protocol_name": "c1_test",
            "model_identity": {},
            "dataset": "wikitext",
            "split": "train",
            "sequence_length": 1024,
            "calibration_sequences": 2,
            "calibration_tokens": 2048,
            "input_ids": tokens,
            "input_ids_sha256": token_tensor_sha256(tokens),
            "attention_mask_semantics": "all_ones_no_padding",
            "frozen_before_profile": True,
            "test_metrics_used": False,
            "source": {"dataset": "wikitext", "split": "train", "arrow_files": []},
        },
        calibration_path,
    )
    torch.save(
        {
            "split": "train",
            "sequence_length": 1024,
            "table": {
                0: {
                    "ranked_indices": torch.arange(8).view(1, -1).expand(4, -1),
                    "block_relative_scores": torch.ones(4, 2),
                    "block_coverage_scores": torch.full((4, 2), 0.5),
                    "block_sizes": torch.tensor([4, 4]),
                    "intermediate_size": 8,
                }
            },
        },
        channel_path,
    )
    args = Namespace(
        official_reap_root=reap_root,
        official_reap_commit="official-commit",
        model_path=str(model_path),
        model_family="qwen3",
        calibration_cache=calibration_path,
        channel_cache=channel_path,
        output_observer=observer_path,
        output_profile=profile_path,
        experts_to_prune_per_layer=2,
        sequence_length=1024,
        batch_group_size=1,
        device_map="cpu",
    )
    monkeypatch.setattr(build_official_reap_profile, "parse_args", lambda: args)
    monkeypatch.setattr(build_official_reap_profile, "_git_commit", lambda path: "official-commit")
    monkeypatch.setattr(build_official_reap_profile, "_git_is_clean", lambda path: True)
    monkeypatch.setattr(
        build_official_reap_profile,
        "calibration_batches_from_payload",
        lambda payload, **kwargs: [
            {
                "input_ids": tokens[:, :1024],
                "attention_mask": torch.ones(1, 1024, dtype=torch.long),
            },
            {
                "input_ids": tokens[:, 1024:],
                "attention_mask": torch.ones(1, 1024, dtype=torch.long),
            },
        ],
    )
    fake_model = FakeQwen3()
    monkeypatch.setattr(build_official_reap_profile, "load_supported_moe", lambda *args, **kwargs: (fake_model, None))
    monkeypatch.setattr(
        build_official_reap_profile,
        "iter_moe_layer_bindings",
        lambda model: [SimpleNamespace(layer_idx=0, top_k=2)],
    )
    monkeypatch.setattr(
        build_official_reap_profile,
        "_load_official_observer_classes",
        lambda repository: (_FakeObserver, {"FakeQwen3": _FakeHookConfig}),
    )
    monkeypatch.setattr(
        build_official_reap_profile,
        "source_tree_identity",
        lambda *args, **kwargs: {
            "commit": "bridge-commit",
            "runtime_tree_sha256": "d" * 64,
        },
    )

    assert build_official_reap_profile.main() == 0
    observer = torch.load(observer_path, map_location="cpu", weights_only=True)
    profile = torch.load(profile_path, map_location="cpu", weights_only=True)

    assert observer["official_reap_commit"] == "official-commit"
    assert observer["renormalize_router_weights"] is True
    assert observer["bridge_source_identity"]["runtime_tree_sha256"] == "d" * 64
    assert profile["method"] == "official_reap"
    assert profile["retained_experts_by_layer"] == [2]
    assert profile["cache_provenance"]["observer"]["sha256"]
    assert profile["bridge_source_identity"] == observer["bridge_source_identity"]