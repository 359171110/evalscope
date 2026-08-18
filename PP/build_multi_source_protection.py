from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from PP.build_gate_hybrid_protection import build_aimer_filled_order
from PP.build_protected_rankings import build_protected_artifacts, cache_orders
from WICK.build_wick_profile import file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a fixed-budget multi-source protection ranking with AIMER fill.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--aimer-cache", type=Path, required=True)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="Ordered source specification NAME=QUOTA=RANKING_CACHE.",
    )
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument("--output-channel-cache", type=Path, required=True)
    parser.add_argument("--diagnostics-output", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--retained-blocks", type=int, required=True)
    parser.add_argument("--protection-ratio", type=float, default=0.10)
    parser.add_argument("--channel-block-size", type=int, default=64)
    return parser.parse_args()


def parse_source_spec(spec: str) -> tuple[str, int, Path]:
    parts = spec.split("=", 2)
    if len(parts) != 3 or not parts[0]:
        raise ValueError("source must use NAME=QUOTA=RANKING_CACHE format.")
    try:
        quota = int(parts[1])
    except ValueError as error:
        raise ValueError("source quota must be an integer.") from error
    if quota < 1:
        raise ValueError("source quota must be positive.")
    return parts[0], quota, Path(parts[2]).expanduser().resolve()


def select_quota_protection(
    source_orders: list[tuple[str, int, torch.Tensor]],
    *,
    aimer_order: torch.Tensor | None = None,
    aimer_retained_channels: int | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Select exact ordered quotas while skipping channels chosen by earlier sources."""

    if not source_orders:
        raise ValueError("at least one protection source is required.")
    channel_count = int(source_orders[0][2].numel())
    if aimer_order is not None:
        if aimer_order.ndim != 1 or int(aimer_order.numel()) != channel_count:
            raise ValueError("AIMER order must be aligned with source orders.")
        if aimer_retained_channels is None or not 1 <= int(aimer_retained_channels) <= channel_count:
            raise ValueError("AIMER retained-channel cutoff must be valid.")
        aimer_set = aimer_order[: int(aimer_retained_channels)]
    else:
        aimer_set = None
    selected_mask = torch.zeros(channel_count, dtype=torch.bool)
    selected_parts = []
    diagnostics: dict[str, float] = {}
    for name, quota, order in source_orders:
        if order.ndim != 1 or int(order.numel()) != channel_count:
            raise ValueError("all source orders must be aligned one-dimensional tensors.")
        order = order.to(dtype=torch.long, device="cpu")
        if not torch.equal(torch.sort(order).values, torch.arange(channel_count)):
            raise ValueError("each source order must be a permutation of all channel indices.")
        available = order[~selected_mask[order]]
        if int(available.numel()) < int(quota):
            raise ValueError(f"source {name} cannot fill its unique channel quota.")
        chosen = available[: int(quota)]
        selected_parts.append(chosen)
        selected_mask[chosen] = True
        if aimer_set is not None:
            already = int(torch.isin(chosen, aimer_set).sum().item())
            diagnostics[f"{name}_already_count"] = float(already)
            diagnostics[f"{name}_rescue_count"] = float(int(chosen.numel()) - already)
            diagnostics[f"{name}_rescue_ratio"] = float((int(chosen.numel()) - already) / int(chosen.numel()))
        chosen_positions = torch.nonzero(torch.isin(order, chosen), as_tuple=False).flatten()
        diagnostics[f"{name}_quota"] = float(quota)
        diagnostics[f"{name}_scan_depth"] = float(chosen_positions.max().item() + 1)
    protected = torch.cat(selected_parts)
    if protected.unique().numel() != protected.numel():
        raise ValueError("multi-source protection must contain unique channels.")
    diagnostics["protected_channels"] = float(protected.numel())
    if aimer_set is not None:
        already = int(torch.isin(protected, aimer_set).sum().item())
        diagnostics["aimer_already_count"] = float(already)
        diagnostics["aimer_rescue_count"] = float(int(protected.numel()) - already)
        diagnostics["aimer_already_ratio"] = float(already / int(protected.numel()))
        diagnostics["aimer_rescue_ratio"] = float((int(protected.numel()) - already) / int(protected.numel()))
    return protected, diagnostics


def summarize_records(records: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = sorted(
        key
        for key in records[0]
        if key.endswith(("_scan_depth", "_count", "_ratio"))
    )
    summary = {}
    for key in keys:
        values = torch.tensor([record[key] for record in records], dtype=torch.float64)
        summary[key] = {
            "mean": float(values.mean().item()),
            "p10": float(torch.quantile(values, 0.10).item()),
            "median": float(values.median().item()),
            "p90": float(torch.quantile(values, 0.90).item()),
            "min": float(values.min().item()),
            "max": float(values.max().item()),
        }
    return summary


def main() -> int:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    aimer_path = args.aimer_cache.expanduser().resolve()
    source_specs = [parse_source_spec(spec) for spec in args.source]
    source_names = [name for name, _, _ in source_specs]
    if len(source_names) != len(set(source_names)):
        raise ValueError("source names must be unique.")

    aimer_orders = cache_orders(torch.load(aimer_path, map_location="cpu", weights_only=True))
    source_orders = []
    for name, quota, path in source_specs:
        orders = cache_orders(torch.load(path, map_location="cpu", weights_only=True))
        if orders.shape != aimer_orders.shape:
            raise ValueError("all source caches must match the AIMER cache dimensions.")
        source_orders.append((name, quota, path, orders))

    channel_count = int(aimer_orders.shape[-1])
    total_protected = sum(quota for _, quota, _, _ in source_orders)
    expected_protected = int(round(channel_count * float(args.protection_ratio)))
    if total_protected != expected_protected:
        raise ValueError(f"source quotas must sum to the fixed protection budget {expected_protected}.")

    combined_orders = []
    records = []
    for layer_id in range(int(aimer_orders.shape[0])):
        layer_orders = []
        for expert_id in range(int(aimer_orders.shape[1])):
            expert_sources = [
                (name, quota, orders[layer_id, expert_id]) for name, quota, _, orders in source_orders
            ]
            protected, diagnostics = select_quota_protection(
                expert_sources,
                aimer_order=aimer_orders[layer_id, expert_id],
                aimer_retained_channels=int(args.retained_blocks) * int(args.channel_block_size),
            )
            layer_orders.append(build_aimer_filled_order(aimer_orders[layer_id, expert_id], protected))
            diagnostics.update({"layer_id": float(layer_id), "expert_id": float(expert_id)})
            records.append(diagnostics)
        combined_orders.append(torch.stack(layer_orders))
    orders = torch.stack(combined_orders)

    sources_metadata = [
        {"name": name, "quota": quota, "path": str(path), "sha256": file_sha256(path)}
        for name, quota, path, _ in source_orders
    ]
    source_identity = json.dumps(sources_metadata, sort_keys=True).encode("utf-8")
    metadata = {
        "method": args.method,
        "selection_policy": "ordered_unique_quota_fill",
        "total_protected": total_protected,
        "sources": sources_metadata,
        "aimer_cache_sha256": file_sha256(aimer_path),
        "diagnostics": summarize_records(records),
    }
    channel, profile = build_protected_artifacts(
        model_path=model_path,
        orders=orders,
        method=args.method,
        backbone="multi_source_protection",
        retained_blocks=int(args.retained_blocks),
        protection_ratio=total_protected / channel_count,
        block_size=int(args.channel_block_size),
        backbone_cache_sha256=file_sha256(aimer_path),
        pseudo_cache_sha256=hashlib.sha256(source_identity).hexdigest(),
    )
    channel["multi_source_protection"] = metadata
    profile["multi_source_protection"] = metadata

    args.output_channel_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(channel, args.output_channel_cache)
    profile["cache_provenance"] = {
        "channel": {"sha256": file_sha256(args.output_channel_cache), "role": args.method}
    }
    args.output_profile.parent.mkdir(parents=True, exist_ok=True)
    torch.save(profile, args.output_profile)
    profile_summary = {key: value for key, value in profile.items() if key != "profile_widths"}
    profile_summary["width_histogram"] = {
        str(int(width)): int(count)
        for width, count in zip(*torch.unique(profile["profile_widths"], return_counts=True))
    }
    args.output_profile.with_suffix(".json").write_text(
        json.dumps(profile_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    diagnostics_payload = {
        "method": args.method,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "expert_count": len(records),
        "summary": metadata["diagnostics"],
        "per_expert": records,
    }
    args.diagnostics_output.parent.mkdir(parents=True, exist_ok=True)
    args.diagnostics_output.write_text(
        json.dumps(diagnostics_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(args.output_channel_cache.resolve())
    print(args.output_profile.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())