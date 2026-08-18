from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.aimer_selector import build_aimer_keep_table_for_model
from src.amp_proxy import build_amp_table_for_model
from src.expert_priors import build_prior_payload
from src.model_loading import load_supported_moe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build independent AMP and AIMER expert-prior caches."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--amp-output", type=Path, required=True)
    parser.add_argument("--aimer-output", type=Path, required=True)
    return parser.parse_args()


@torch.no_grad()
def main() -> int:
    args = parse_args()
    model, _ = load_supported_moe(args.model_path)
    amp = build_prior_payload(
        method="top_p_method1",
        model_path=args.model_path,
        table=build_amp_table_for_model(model),
    )
    aimer = build_prior_payload(
        method="top_p_aimer",
        model_path=args.model_path,
        table=build_aimer_keep_table_for_model(model),
    )
    args.amp_output.parent.mkdir(parents=True, exist_ok=True)
    args.aimer_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(amp, args.amp_output)
    torch.save(aimer, args.aimer_output)
    print(args.amp_output.resolve())
    print(args.aimer_output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
