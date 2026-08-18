from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from NAPS_v2.build_naps_v2_artifacts import file_sha256
from NAPS_v2.build_naps_v2_heterogeneous import json_ready


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a fixed-budget heterogeneous profile from AIMER and PP expert ranks."
    )
    parser.add_argument("--source-artifact-dir", type=Path, required=True)
    parser.add_argument("--pp-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def rank_positions(values: torch.Tensor, *, descending: bool) -> torch.Tensor:
    if values.ndim != 1:
        raise ValueError("ranked values must be one-dimensional")
    order = torch.argsort(values.float(), descending=descending, stable=True)
    sorted_values = values.float().index_select(0, order)
    positions = torch.empty(order.numel(), dtype=torch.float32, device=values.device)
    start = 0
    while start < order.numel():
        end = start + 1
        while end < order.numel() and bool(sorted_values[end] == sorted_values[start]):
            end += 1
        positions[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return positions


def assign_expert_widths_aimer_pp(
    expert_aimer_scores: torch.Tensor,
    pp_rescue_counts: torch.Tensor,
    widths: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse AIMER removability and PP exposure while preserving fixed quartiles."""

    if expert_aimer_scores.ndim != 1 or pp_rescue_counts.ndim != 1:
        raise ValueError("expert scores and PP counts must be one-dimensional")
    if expert_aimer_scores.numel() != pp_rescue_counts.numel() or expert_aimer_scores.numel() < 4:
        raise ValueError("expert scores and PP counts must have the same minimum length")
    small_width, medium_width, large_width = (int(value) for value in widths)
    if not small_width < medium_width < large_width:
        raise ValueError("widths must be strictly increasing")
    if medium_width - small_width != large_width - medium_width:
        raise ValueError("widths must be symmetric around the medium width")
    expert_count = int(expert_aimer_scores.numel())
    if expert_count % 4:
        raise ValueError("expert count must be divisible by four")

    aimer_removability_rank = rank_positions(expert_aimer_scores, descending=True)
    pp_removability_rank = rank_positions(pp_rescue_counts, descending=False)
    fused_rank = aimer_removability_rank + pp_removability_rank
    order = torch.argsort(fused_rank, stable=True)
    quarter = expert_count // 4
    assigned = torch.full((expert_count,), medium_width, dtype=torch.long)
    assigned[order[:quarter]] = small_width
    assigned[order[-quarter:]] = large_width
    return assigned, fused_rank


def load_pp_counts(path: Path, num_layers: int, num_experts: int) -> torch.Tensor:
    counts = torch.full((num_layers, num_experts), float("nan"), dtype=torch.float32)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            layer_id = int(row["layer_id"])
            expert_id = int(row["expert_id"])
            if not 0 <= layer_id < num_layers or not 0 <= expert_id < num_experts:
                raise ValueError("PP summary contains an out-of-range expert")
            counts[layer_id, expert_id] = float(row["pp_rescue_count"])
    if not torch.isfinite(counts).all():
        raise ValueError("PP summary does not cover every layer and expert")
    return counts


def main() -> int:
    args = parse_args()
    source_dir = args.source_artifact_dir.expanduser().resolve()
    pp_summary = args.pp_summary.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rankings = torch.load(source_dir / "rankings.pt", map_location="cpu", weights_only=True)
    profile = torch.load(source_dir / "profile.pt", map_location="cpu", weights_only=True)
    audit = torch.load(source_dir / "routing_audit.pt", map_location="cpu", weights_only=True)
    diagnostics = json.loads((source_dir / "diagnostics.json").read_text(encoding="utf-8"))
    num_layers = int(profile["num_layers"])
    num_experts = int(profile["num_experts"])
    widths = tuple(int(value) for value in profile["width_options"].tolist())
    block_size = int(profile["channel_block_size"])
    pp_counts = load_pp_counts(pp_summary, num_layers, num_experts)

    records_by_key = {
        (int(record["layer_id"]), int(record["expert_id"])): record
        for record in diagnostics.get("records", [])
    }
    profile_widths = []
    fused_ranks_by_layer = []
    for layer_id in range(num_layers):
        aimer_scores = torch.tensor(
            [records_by_key[(layer_id, expert_id)]["expert_aimer_score"] for expert_id in range(num_experts)],
            dtype=torch.float32,
        )
        assigned, fused_rank = assign_expert_widths_aimer_pp(
            aimer_scores,
            pp_counts[layer_id],
            widths,
        )
        if int(assigned.sum().item()) != num_experts * widths[1]:
            raise RuntimeError(f"Layer {layer_id} does not preserve the B6 budget")
        profile_widths.append(assigned)
        fused_ranks_by_layer.append(fused_rank)
        for expert_id, width in enumerate(assigned.tolist()):
            record = records_by_key[(layer_id, expert_id)]
            record["assigned_width"] = int(width)
            record["pp_rescue_count"] = float(pp_counts[layer_id, expert_id].item())
            record["aimer_pp_fused_rank"] = float(fused_rank[expert_id].item())

    profile_widths = torch.stack(profile_widths)
    metadata = dict(profile.get("naps", {}))
    metadata.update(
        {
            "version": 5,
            "expert_importance": "equal_rank_fusion_original_aimer_and_pp_rescue_exposure",
            "width_assignment": "per_layer_aimer_pp_rank_fusion_quartiles",
            "pp_signal": "pp_rescue_count",
            "source_artifact_dir": str(source_dir),
            "pp_summary": str(pp_summary),
        }
    )
    rankings["naps"] = metadata
    profile.update(
        {
            "schema_version": 5,
            "mode": "expert_aimer_pp_rank_fusion_quartiles_padded_homogeneous",
            "profile_widths": profile_widths // block_size,
            "allocation_scope": "per_layer_expert_aimer_pp_rank_fusion_quartiles",
            "total_blocks": int((profile_widths // block_size).sum().item()),
            "naps": metadata,
        }
    )

    rankings_path = output_dir / "rankings.pt"
    torch.save(rankings, rankings_path)
    profile["cache_provenance"] = {
        "channel_sha256": file_sha256(rankings_path),
        "source_channel_sha256": file_sha256(source_dir / "rankings.pt"),
        "pp_summary_sha256": file_sha256(pp_summary),
    }
    torch.save(profile, output_dir / "profile.pt")
    torch.save(audit, output_dir / "routing_audit.pt")
    diagnostics.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "schema_version": 5,
            "source_artifact_dir": str(source_dir),
            "pp_summary": str(pp_summary),
            "allocation_scope": "per_layer_expert_aimer_pp_rank_fusion_quartiles",
            "profile_widths": profile_widths.tolist(),
            "fused_ranks_by_layer": [fused.tolist() for fused in fused_ranks_by_layer],
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