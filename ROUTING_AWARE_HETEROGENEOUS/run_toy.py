"""Run a small end-to-end smoke test for the pruning method."""

from __future__ import annotations

import argparse

import torch

from .config import MethodConfig
from .core import RoutingAwarePruner
from .toy import ToyAdapter


def main() -> int:
    """Run the toy pipeline and print compact diagnostics."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    adapter = ToyAdapter(device=args.device)
    config = MethodConfig(
        natural_sequences=4,
        guided_sequences=2,
        sequence_length=16,
        min_samples_per_expert=2,
        safe_samples_per_expert=1,
        max_samples_per_expert=8,
        width_levels=(1.0, 0.75, 0.5, 0.25),
        retention=0.5,
        device=args.device,
    )
    input_ids = torch.arange(64, device=adapter.device).reshape(4, 16)
    natural_mass_before = None
    result = RoutingAwarePruner(adapter, config).run(input_ids)
    natural_mass_before = result.natural_mass.clone()
    assert int(result.widths.sum(dim=1)[0].item()) == 12
    assert torch.equal(result.natural_mass, natural_mass_before)
    print({"device": str(adapter.device), "widths": result.widths.cpu().tolist(), "mass_shape": list(result.natural_mass.shape)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())