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
        description="Build a fixed-budget heterogeneous profile from Gaussian functional width utility."
    )
    parser.add_argument("--source-artifact-dir", type=Path, required=True)
    parser.add_argument("--assignment-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_gaussian_assignments(
    path: Path,
    num_layers: int,
    num_experts: int,
    widths: tuple[int, int, int],
) -> torch.Tensor:
    """Load a complete per-layer Gaussian utility assignment with fixed quartile counts."""

    small_width, medium_width, large_width = widths
    if not small_width < medium_width < large_width:
        raise ValueError("widths must be strictly increasing")
    if medium_width - small_width != large_width - medium_width:
        raise ValueError("widths must be symmetric around the medium width")
    if num_experts % 4:
        raise ValueError("expert count must be divisible by four")

    assigned = torch.full((num_layers, num_experts), -1, dtype=torch.long)
    valid_widths = set(widths)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_fields = {"layer_id", "expert_id", "allocation", "assigned_width"}
        if reader.fieldnames is None or not required_fields.issubset(reader.fieldnames):
            raise ValueError("Gaussian assignment CSV is missing required columns")
        for row in reader:
            layer_id = int(row["layer_id"])
            expert_id = int(row["expert_id"])
            width = int(row["assigned_width"])
            if not 0 <= layer_id < num_layers or not 0 <= expert_id < num_experts:
                raise ValueError("Gaussian assignment contains an out-of-range expert")
            if int(assigned[layer_id, expert_id].item()) >= 0:
                raise ValueError("Gaussian assignment contains a duplicate expert")
            if width not in valid_widths:
                raise ValueError("Gaussian assignment contains an unsupported width")
            expected_allocation = {
                small_width: "small",
                medium_width: "medium",
                large_width: "large",
            }[width]
            if row["allocation"] != expected_allocation:
                raise ValueError("Gaussian allocation label does not match its assigned width")
            assigned[layer_id, expert_id] = width

    if bool((assigned < 0).any()):
        raise ValueError("Gaussian assignment does not cover every layer and expert")

    expected_counts = (num_experts // 4, num_experts // 2, num_experts // 4)
    for layer_id, layer_widths in enumerate(assigned):
        counts = tuple(int((layer_widths == width).sum().item()) for width in widths)
        if counts != expected_counts:
            raise ValueError(f"Layer {layer_id} does not preserve fixed quartile counts")
        if int(layer_widths.sum().item()) != num_experts * medium_width:
            raise ValueError(f"Layer {layer_id} does not preserve the medium-width budget")
    return assigned


def main() -> int:
    args = parse_args()
    source_dir = args.source_artifact_dir.expanduser().resolve()
    assignment_csv = args.assignment_csv.expanduser().resolve()
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
    profile_widths = load_gaussian_assignments(
        assignment_csv,
        num_layers,
        num_experts,
        widths,
    )

    metadata = dict(profile.get("naps", {}))
    metadata.update(
        {
            "version": 6,
            "expert_importance": "gaussian_functional_marginal_width_utility",
            "width_assignment": "per_layer_exact_disjoint_gaussian_functional_quartiles",
            "source_artifact_dir": str(source_dir),
            "assignment_csv": str(assignment_csv),
        }
    )
    rankings["naps"] = metadata
    profile.update(
        {
            "schema_version": 6,
            "mode": "gaussian_functional_marginal_quartiles_padded_homogeneous",
            "profile_widths": profile_widths // block_size,
            "allocation_scope": "per_layer_exact_disjoint_gaussian_functional_quartiles",
            "total_blocks": int((profile_widths // block_size).sum().item()),
            "naps": metadata,
        }
    )

    records_by_key = {
        (int(record["layer_id"]), int(record["expert_id"])): record
        for record in diagnostics.get("records", [])
    }
    for layer_id, layer_widths in enumerate(profile_widths):
        for expert_id, width in enumerate(layer_widths.tolist()):
            record = records_by_key.get((layer_id, expert_id))
            if record is not None:
                record["assigned_width"] = int(width)

    rankings_path = output_dir / "rankings.pt"
    torch.save(rankings, rankings_path)
    profile["cache_provenance"] = {
        "channel_sha256": file_sha256(rankings_path),
        "source_channel_sha256": file_sha256(source_dir / "rankings.pt"),
        "assignment_csv_sha256": file_sha256(assignment_csv),
    }
    torch.save(profile, output_dir / "profile.pt")
    torch.save(audit, output_dir / "routing_audit.pt")
    diagnostics.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "schema_version": 6,
            "source_artifact_dir": str(source_dir),
            "assignment_csv": str(assignment_csv),
            "allocation_scope": "per_layer_exact_disjoint_gaussian_functional_quartiles",
            "profile_widths": profile_widths.tolist(),
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