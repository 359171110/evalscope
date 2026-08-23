from __future__ import annotations

from pathlib import Path

import torch

from AIMER_MIX_PLUS.build_pseudo_source_artifacts import scores_to_table
from AIMER_MIX_PLUS.layerprop_core import (
    accumulate_routed_channel_scores,
    finalize_layerprop_scores,
    synthetic_input_ids,
)
from AIMER_MIX_PLUS.source_cache import load_pseudo_source


def test_synthetic_lattice_is_data_free_and_skips_specials() -> None:
    ids = synthetic_input_ids(
        vocab_size=16,
        num_sequences=2,
        sequence_length=4,
        bos_token_id=2,
        pad_token_id=0,
    )
    assert ids.shape == (2, 4)
    assert torch.equal(ids[:, 0], torch.tensor([2, 2]))
    assert not (ids == 0).any()
    assert ids.min() >= 1
    assert ids.max() <= 15


def test_routed_scores_use_ffn_hidden_and_down_norm() -> None:
    hidden = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    top_k_index = torch.tensor([[0], [0]])
    top_k_weights = torch.tensor([[1.0], [1.0]])
    gate_up = torch.zeros(2, 4, 2)
    gate_up[0, 0] = torch.tensor([1.0, 0.0])
    gate_up[0, 2] = torch.tensor([1.0, 0.0])
    gate_up[0, 1] = torch.tensor([0.0, 1.0])
    gate_up[0, 3] = torch.tensor([0.0, 1.0])
    down = torch.zeros(2, 2, 2)
    down[0, :, 0] = torch.tensor([1.0, 0.0])
    down[0, :, 1] = torch.tensor([10.0, 0.0])
    scores = torch.zeros(2, 2)
    mass = torch.zeros(2)
    hits = torch.zeros(2)
    accumulate_routed_channel_scores(
        hidden,
        top_k_index,
        top_k_weights,
        gate_up,
        down,
        activation="gelu_pytorch_tanh",
        score_mode="output",
        scores=scores,
        mass=mass,
        hit_counts=hits,
    )
    mean_scores, coverage, stability = finalize_layerprop_scores(scores, mass, hits)
    assert coverage[0] == 1.0
    assert coverage[1] == 0.0
    assert mean_scores[0, 1] > mean_scores[0, 0]
    assert torch.equal(mean_scores[1], torch.zeros(2))
    assert stability[0] > 0.0
    assert stability[1] == 0.0


def test_uncovered_expert_keeps_identity_ranking(tmp_path: Path) -> None:
    scores = torch.tensor([[3.0, 1.0, 2.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    mass = torch.tensor([2.0, 0.0])
    hits = torch.tensor([2.0, 0.0])
    mean_scores, coverage, _stability = finalize_layerprop_scores(scores, mass, hits)
    table = scores_to_table(mean_scores, 2)
    cache = tmp_path / "layerprop.pt"
    torch.save(
        {
            "schema_version": 1,
            "purpose": "aimer_mix_plus_pseudo_source",
            "calibration_sequences": 0,
            "test_metrics_used": False,
            "table": {0: table},
            "pseudo_source": {
                "name": "layerprop",
                "coverage": coverage.unsqueeze(0),
                "stability": torch.tensor([[1.0, 0.0]]),
            },
        },
        cache,
    )
    source = load_pseudo_source(
        name="layerprop",
        cache_path=cache,
        layer_ids=(0,),
        num_experts=2,
        channels=4,
    )
    assert source.order.shape == (1, 2, 4)
    assert source.coverage[0, 0] == 1.0
    assert source.coverage[0, 1] == 0.0
    assert torch.equal(source.order[0, 0, :2], torch.tensor([0, 2]))


def test_layer_id_from_packed_and_vl_module_names() -> None:
    from AIMER_MIX_PLUS.build_layerprop_source_artifacts import layer_id_from_module_name

    assert layer_id_from_module_name("model.layers.12.mlp.experts") == 12
    assert layer_id_from_module_name("model.language_model.layers.3.experts") == 3
    assert layer_id_from_module_name("experts") is None


def test_pack_separate_expert_weights_matches_linear_layout() -> None:
    import torch.nn as nn

    from AIMER_MIX_PLUS.layerprop_core import pack_separate_expert_weights

    class TinyMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(2, 3, bias=False)
            self.up_proj = nn.Linear(2, 3, bias=False)
            self.down_proj = nn.Linear(3, 2, bias=False)

    experts = nn.ModuleList([TinyMLP(), TinyMLP()])
    gate_up, down = pack_separate_expert_weights(experts)
    assert gate_up.shape == (2, 6, 2)
    assert down.shape == (2, 2, 3)
    assert torch.equal(gate_up[0, :3], experts[0].gate_proj.weight.detach())
    assert torch.equal(gate_up[0, 3:], experts[0].up_proj.weight.detach())
    assert torch.equal(down[1], experts[1].down_proj.weight.detach())
