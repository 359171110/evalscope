from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Derive a fixed-width Pure-Pseudo profile from a frozen profile.")
    parser.add_argument("--source-profile", type=Path, required=True)
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument("--retained-blocks", type=int, required=True)
    return parser.parse_args()


def derive_fixed_width_profile(source: dict, retained_blocks: int) -> dict:
    if source.get("method") != "pure_pseudo":
        raise ValueError("source profile must be produced by the Pure-Pseudo builder.")
    num_blocks = int(source["num_blocks"])
    retained = int(retained_blocks)
    if not 1 <= retained < num_blocks:
        raise ValueError("retained_blocks must be in [1, num_blocks).")

    profile = dict(source)
    widths = torch.full_like(source["profile_widths"], retained, dtype=torch.long)
    total_blocks = int(widths.sum().item())
    maximum_blocks = int(widths.numel() * num_blocks)
    target_blocks_by_layer = widths.sum(dim=1).tolist()
    profile.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "mode": "router_gram_pure_pseudo_channel_ranking_fixed_width_sweep",
            "allocation_scope": "per_expert_fixed",
            "target_blocks_by_layer": target_blocks_by_layer,
            "actual_blocks_by_layer": target_blocks_by_layer,
            "total_blocks": total_blocks,
            "maximum_blocks": maximum_blocks,
            "target_pruning_ratio": 1.0 - retained / num_blocks,
            "actual_structural_pruning_ratio": 1.0 - total_blocks / maximum_blocks,
            "profile_widths": widths,
            "profile_sha256": hashlib.sha256(widths.numpy().tobytes(order="C")).hexdigest(),
            "width_sweep": {
                "retained_blocks": retained,
                "pruned_blocks": num_blocks - retained,
                "source_profile_sha256": source.get("profile_sha256"),
            },
        }
    )
    return profile


def main() -> int:
    args = parse_args()
    source_path = args.source_profile.expanduser().resolve()
    output_path = args.output_profile.expanduser().resolve()
    source = torch.load(source_path, map_location="cpu", weights_only=True)
    profile = derive_fixed_width_profile(source, args.retained_blocks)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(profile, output_path)
    summary = {key: value for key, value in profile.items() if key != "profile_widths"}
    summary["width_histogram"] = {
        str(int(width)): int(count) for width, count in zip(*torch.unique(profile["profile_widths"], return_counts=True))
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())