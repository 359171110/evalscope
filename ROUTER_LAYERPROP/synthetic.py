"""Deterministic, data-free synthetic token and embedding helpers."""

from __future__ import annotations

import torch


def synthetic_input_ids(
    *,
    vocab_size: int,
    num_sequences: int,
    sequence_length: int,
    bos_token_id: int = 2,
    pad_token_id: int = 0,
) -> torch.Tensor:
    """Build a deterministic vocabulary lattice without reading a corpus."""

    if vocab_size < 3 or num_sequences <= 0 or sequence_length <= 0:
        raise ValueError("vocab_size must be >= 3 and sequence dimensions must be positive")
    usable = torch.tensor(
        [idx for idx in range(int(vocab_size)) if idx not in {int(bos_token_id), int(pad_token_id)}],
        dtype=torch.long,
    )
    if usable.numel() == 0:
        raise ValueError("No non-special vocabulary ids are available")
    total = int(num_sequences) * int(sequence_length)
    lattice = torch.arange(total, dtype=torch.long).remainder(usable.numel())
    ids = usable.index_select(0, lattice).view(int(num_sequences), int(sequence_length))
    ids[:, 0] = int(bos_token_id)
    return ids


def embedding_probe_scale(model: torch.nn.Module, sample_rows: int = 4096) -> float:
    """Estimate source-0 raw residual scale from a fixed embedding subset."""

    embeddings = model.get_input_embeddings().weight.detach().float()
    rows = min(int(sample_rows), embeddings.shape[0])
    if rows <= 0:
        raise ValueError("Model has no embedding rows")
    scale = embeddings[:rows].square().mean(dim=-1).sqrt().median()
    config = getattr(model, "config", object())
    model_type = str(getattr(config, "model_type", "")).lower()
    if model_type.startswith("gemma"):
        scale = scale * float(embeddings.shape[-1]) ** 0.5
    return float(scale.clamp_min(1.0e-6).item())
