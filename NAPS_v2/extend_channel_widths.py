from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch

from NAPS_v2.build_channel_profile import validate_nested_rankings
from NAPS_v2.build_naps_v2_artifacts import file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add nested width slots to a CHANNEL ranking artifact.")
    parser.add_argument("--source-artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--widths", type=int, nargs="+", required=True)
    return parser.parse_args()


def layer_table(rankings: dict[str, Any], layer_id: int) -> dict[str, Any]:
    table = rankings["table"].get(layer_id, rankings["table"].get(str(layer_id)))
    if table is None:
        raise KeyError(f"Missing CHANNEL ranking table for layer {layer_id}")
    return table


def extend_widths(rankings: dict[str, Any], requested_widths: list[int]) -> dict[str, Any]:
    layer_ids, _, source_width, existing_widths, block_size = validate_nested_rankings(rankings)
    added_widths = sorted(set(map(int, requested_widths)) - set(existing_widths))
    for width in added_widths:
        if width <= 0 or width > source_width or width % block_size:
            raise ValueError(
                f"Width {width} must be positive, no larger than {source_width}, and aligned to {block_size}"
            )
    output = dict(rankings)
    output["width_options"] = tuple(sorted((*existing_widths, *added_widths)))
    output["table"] = dict(rankings["table"])
    for layer_id in layer_ids:
        source_table = layer_table(rankings, layer_id)
        table = dict(source_table)
        canonical_order = source_table["ranked_indices"].to(torch.long)
        width_options = output["width_options"]
        table["width_options"] = torch.tensor(width_options, dtype=torch.long)
        table["ranked_indices_by_width"] = canonical_order[:, None, :].expand(
            -1, len(width_options), -1
        ).clone()
        output["table"][layer_id] = table
        if str(layer_id) in output["table"] and str(layer_id) != layer_id:
            del output["table"][str(layer_id)]
    output["width_extension"] = {
        "construction": "canonical_nested_order_replication",
        "added_widths": added_widths,
        "scores_recomputed": False,
        "rankings_changed": False,
    }
    validate_nested_rankings(output)
    return output


def main() -> int:
    args = parse_args()
    source_dir = args.source_artifact_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = source_dir / "rankings.pt"
    rankings = torch.load(source_path, map_location="cpu", weights_only=True)
    extended = extend_widths(rankings, args.widths)
    extended["width_extension"]["source_rankings_path"] = str(source_path)
    extended["width_extension"]["source_rankings_sha256"] = file_sha256(source_path)
    output_path = output_dir / "rankings.pt"
    torch.save(extended, output_path)
    for name in ("config.json", "metadata.json", "diagnostics.json"):
        source_file = source_dir / name
        if source_file.exists():
            shutil.copy2(source_file, output_dir / name)
    summary = {
        **extended["width_extension"],
        "output_rankings_path": str(output_path),
        "output_rankings_sha256": file_sha256(output_path),
        "width_options": list(extended["width_options"]),
    }
    (output_dir / "width_extension.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())