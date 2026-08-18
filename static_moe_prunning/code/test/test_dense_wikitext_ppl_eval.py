from __future__ import annotations

import sys

from scripts import dense_wikitext_ppl_eval


def test_dense_eval_accepts_auto_device_map_and_reap_labels(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dense_wikitext_ppl_eval.py",
            "--model-path",
            "/model",
            "--output-dir",
            str(tmp_path),
            "--device-map",
            "auto",
            "--method",
            "reap_per_layer",
            "--mode",
            "reap_whole_expert_pruning",
            "--structural-pruning-ratio",
            "0.59375",
        ],
    )
    args = dense_wikitext_ppl_eval.parse_args()
    assert args.device_map == "auto"
    assert args.method == "reap_per_layer"
    assert args.mode == "reap_whole_expert_pruning"
    assert args.structural_pruning_ratio == 0.59375
