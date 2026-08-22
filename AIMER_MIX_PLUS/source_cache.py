from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

import torch

from AIMER_MIX_PLUS.plus_core import PseudoSource


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _layer_row(table: dict[Any, Any], layer_id: int) -> dict[str, Any]:
    if layer_id in table:
        row = table[layer_id]
    elif str(layer_id) in table:
        row = table[str(layer_id)]
    else:
        raise KeyError(f"Pseudo ranking cache is missing layer {layer_id}")
    if not isinstance(row, dict):
        raise ValueError(f"Pseudo ranking layer {layer_id} must be a mapping")
    return row


def _stack_orders(
    payload: dict[str, Any],
    layer_ids: Sequence[int],
    num_experts: int,
    channels: int,
) -> torch.Tensor:
    table = payload.get("table")
    if not isinstance(table, dict):
        raise ValueError("Pseudo ranking cache must contain a table mapping")
    rows = []
    expected = torch.arange(channels)
    for layer_id in layer_ids:
        row = _layer_row(table, int(layer_id))
        order = row.get("ranked_indices")
        if not isinstance(order, torch.Tensor) or tuple(order.shape) != (num_experts, channels):
            raise ValueError(
                f"Pseudo ranking layer {layer_id} must have ranked_indices shape "
                f"{(num_experts, channels)}"
            )
        order = order.to(torch.long).cpu()
        if not torch.equal(torch.sort(order, dim=1).values, expected.expand(num_experts, -1)):
            raise ValueError(f"Pseudo ranking layer {layer_id} rows must be complete permutations")
        rows.append(order)
    return torch.stack(rows)


def _layer_coverage(
    payload: dict[str, Any],
    layer_ids: Sequence[int],
    num_experts: int,
    floor: float,
) -> torch.Tensor:
    if not 0.0 <= floor <= 1.0:
        raise ValueError("coverage floor must be in [0, 1]")
    values = torch.ones((len(layer_ids), num_experts), dtype=torch.float32)
    metadata = payload.get("naps")
    records = metadata.get("coverage") if isinstance(metadata, dict) else None
    if not isinstance(records, list):
        return values
    by_layer = {
        int(record["layer_id"]): record
        for record in records
        if isinstance(record, dict) and "layer_id" in record
    }
    for position, layer_id in enumerate(layer_ids):
        record = by_layer.get(int(layer_id))
        if record is None:
            continue
        covered = float(record.get("experts_covered", num_experts)) / float(max(num_experts, 1))
        confidence = floor + (1.0 - floor) * max(0.0, min(1.0, covered))
        values[position].fill_(confidence)
    return values


def _tensor_confidence(
    payload: dict[str, Any],
    key: str,
    layer_ids: Sequence[int],
    num_experts: int,
) -> torch.Tensor | None:
    """Read optional ``[layer, expert]`` confidence tensors from a cache."""

    raw = payload.get(key)
    if raw is None:
        metadata = payload.get("pseudo_source")
        raw = metadata.get(key) if isinstance(metadata, dict) else None
    if raw is None:
        return None
    if not isinstance(raw, torch.Tensor):
        raw = torch.as_tensor(raw)
    if tuple(raw.shape) == (len(layer_ids), num_experts):
        values = raw.to(dtype=torch.float32, device="cpu")
    elif tuple(raw.shape) == (max(layer_ids) + 1, num_experts):
        values = raw.index_select(0, torch.tensor(layer_ids)).to(dtype=torch.float32, device="cpu")
    else:
        raise ValueError(f"Pseudo source {key} must have shape [layers, experts]")
    if not bool(torch.isfinite(values).all()) or bool(((values < 0) | (values > 1)).any()):
        raise ValueError(f"Pseudo source {key} must be finite and in [0, 1]")
    return values


def load_pseudo_source(
    *,
    name: str,
    cache_path: Path,
    layer_ids: Sequence[int],
    num_experts: int,
    channels: int,
    model_path: Path | None = None,
    base_weight: float = 1.0,
    coverage_floor: float = 0.35,
    stability: float = 1.0,
    strict_model_path: bool = True,
) -> PseudoSource:
    """Load a ranking-only PP, PRP, or LayerProp artifact as an AMP source."""

    path = cache_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Pseudo ranking cache must contain a mapping")
    if payload.get("test_metrics_used") is True:
        raise ValueError("Pseudo ranking sources must not use test metrics")
    calibration_sequences = int(payload.get("calibration_sequences", 0) or 0)
    if calibration_sequences != 0:
        raise ValueError("AIMER-Mix-Plus accepts only data-free pseudo ranking sources")
    source_model = payload.get("model_path")
    if strict_model_path and model_path is not None and source_model:
        if Path(str(source_model)).expanduser().resolve() != model_path.expanduser().resolve():
            raise ValueError("Pseudo ranking cache was built for a different model path")
    if model_path is not None:
        provenance = payload.get("model_provenance")
        if isinstance(provenance, dict):
            config_sha = provenance.get("config_sha256")
            index_sha = provenance.get("weight_index_sha256")
            if config_sha is not None and config_sha != file_sha256(model_path / "config.json"):
                raise ValueError("Checkpoint config changed after pseudo ranking construction")
            index_path = model_path / "model.safetensors.index.json"
            if index_sha is not None and index_sha != file_sha256(index_path):
                raise ValueError("Checkpoint weight index changed after pseudo ranking construction")
    order = _stack_orders(payload, layer_ids, num_experts, channels)
    coverage = _tensor_confidence(payload, "coverage", layer_ids, num_experts)
    if coverage is None:
        coverage = _layer_coverage(payload, layer_ids, num_experts, coverage_floor)
    stability_values = _tensor_confidence(payload, "stability", layer_ids, num_experts)
    if stability_values is None:
        stability_values = torch.full_like(coverage, float(stability))
    source = PseudoSource(
        name=name,
        order=order,
        coverage=coverage,
        stability=stability_values,
        base_weight=float(base_weight),
        metadata={
            "cache_path": str(path),
            "cache_sha256": file_sha256(path),
            "purpose": payload.get("purpose"),
            "method": payload.get("method"),
            "model_path": source_model,
        },
    )
    source.validate(len(layer_ids), num_experts, channels)
    return source
