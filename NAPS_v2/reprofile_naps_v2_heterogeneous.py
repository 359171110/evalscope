from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from NAPS_v2.build_naps_v2_artifacts import file_sha256
from NAPS_v2.build_naps_v2_heterogeneous import assign_expert_widths_adaptive, json_ready


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an adaptive NAPS-v2 heterogeneous profile from cached AIMER rankings."
    )
    parser.add_argument("--source-artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def adaptive_widths_from_rankings(
    rankings: dict[str, Any],
    widths: tuple[int, int, int],
) -> torch.Tensor:
    table = rankings.get("table")
    if not isinstance(table, dict) or not table:
        raise ValueError("rankings must contain a non-empty layer table")
    assigned_by_layer = []
    expert_count = None
    for layer_id in range(len(table)):
        layer = table.get(layer_id, table.get(str(layer_id)))
        if layer is None or "expert_aimer_scores" not in layer:
            raise ValueError(f"missing expert AIMER scores for layer {layer_id}")
        scores = layer["expert_aimer_scores"]
        if expert_count is None:
            expert_count = int(scores.numel())
        elif int(scores.numel()) != expert_count:
            raise ValueError("all layers must contain the same number of experts")
        assigned_by_layer.append(assign_expert_widths_adaptive(scores, widths))
    return torch.stack(assigned_by_layer)


def main() -> int:
    args = parse_args()
    source_dir = args.source_artifact_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rankings = torch.load(source_dir / "rankings.pt", map_location="cpu", weights_only=True)
    profile = torch.load(source_dir / "profile.pt", map_location="cpu", weights_only=True)
    audit = torch.load(source_dir / "routing_audit.pt", map_location="cpu", weights_only=True)
    diagnostics = json.loads((source_dir / "diagnostics.json").read_text(encoding="utf-8"))

    widths = tuple(int(value) for value in profile["width_options"].tolist())
    block_size = int(profile["channel_block_size"])
    profile_widths = adaptive_widths_from_rankings(rankings, widths)
    expected_total = profile_widths.numel() * widths[1]
    if int(profile_widths.sum().item()) != expected_total:
        raise ValueError("adaptive profile does not preserve the medium-width budget")

    metadata = dict(profile.get("naps", {}))
    metadata.update(
        {
            "version": 4,
            "width_assignment": "per_layer_balanced_minimum_variance_aimer_bands",
            "source_artifact_dir": str(source_dir),
        }
    )
    rankings["naps"] = metadata
    profile.update(
        {
            "schema_version": 4,
            "mode": "expert_aimer_minimum_variance_bands_padded_homogeneous",
            "profile_widths": profile_widths // block_size,
            "allocation_scope": "per_layer_expert_aimer_minimum_variance_bands",
            "total_blocks": int((profile_widths // block_size).sum().item()),
            "naps": metadata,
        }
    )

    records_by_key = {
        (int(record["layer_id"]), int(record["expert_id"])): record for record in diagnostics.get("records", [])
    }
    tail_counts = []
    for layer_id, layer_widths in enumerate(profile_widths):
        tail_count = int((layer_widths == widths[0]).sum().item())
        if tail_count != int((layer_widths == widths[2]).sum().item()):
            raise ValueError(f"layer {layer_id} has unbalanced small and large expert counts")
        tail_counts.append(tail_count)
        for expert_id, width in enumerate(layer_widths.tolist()):
            record = records_by_key.get((layer_id, expert_id))
            if record is not None:
                record["assigned_width"] = int(width)

    rankings_path = output_dir / "rankings.pt"
    torch.save(rankings, rankings_path)
    profile["cache_provenance"] = {
        "channel_sha256": file_sha256(rankings_path),
        "source_channel_sha256": file_sha256(source_dir / "rankings.pt"),
    }
    torch.save(profile, output_dir / "profile.pt")
    torch.save(audit, output_dir / "routing_audit.pt")
    diagnostics.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "schema_version": 4,
            "source_artifact_dir": str(source_dir),
            "allocation_scope": "per_layer_expert_aimer_minimum_variance_bands",
            "profile_widths": profile_widths.tolist(),
            "tail_counts_by_layer": tail_counts,
            "ranking_cache_sha256": file_sha256(rankings_path),
        }
    )
    (output_dir / "diagnostics.json").write_text(
        json.dumps(json_ready(diagnostics), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())