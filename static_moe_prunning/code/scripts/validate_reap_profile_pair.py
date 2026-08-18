from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from src.protocol_comparison import validate_profile_pair
from src.static_expert_pruning import validate_static_profile_payload


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit a frozen official REAP and static-prefix profile pair."
    )
    parser.add_argument("--reap-profile", type=Path, required=True)
    parser.add_argument("--candidate-profile", type=Path, required=True)
    parser.add_argument(
        "--group",
        choices=("method_native", "per_layer_controlled"),
        required=True,
    )
    parser.add_argument("--evaluation-cache", type=Path, required=True)
    parser.add_argument("--expected-evaluation-cache-sha256", required=True)
    parser.add_argument("--output-audit", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reap_path = args.reap_profile.expanduser().resolve()
    candidate_path = args.candidate_profile.expanduser().resolve()
    evaluation_path = args.evaluation_cache.expanduser().resolve()
    for path in (reap_path, candidate_path, evaluation_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    reap_profile = torch.load(reap_path, map_location="cpu", weights_only=True)
    candidate_profile = torch.load(candidate_path, map_location="cpu", weights_only=True)
    validate_static_profile_payload(reap_profile)
    validate_static_profile_payload(candidate_profile)
    evaluation_hash = file_sha256(evaluation_path)
    audit = validate_profile_pair(
        reap_profile,
        candidate_profile,
        group=args.group,
        evaluation_cache_sha256=evaluation_hash,
        expected_evaluation_cache_sha256=args.expected_evaluation_cache_sha256,
    )
    audit.update(
        {
            "reap_profile_path": str(reap_path),
            "reap_profile_file_sha256": file_sha256(reap_path),
            "candidate_profile_path": str(candidate_path),
            "candidate_profile_file_sha256": file_sha256(candidate_path),
            "evaluation_cache_path": str(evaluation_path),
        }
    )
    args.output_audit.parent.mkdir(parents=True, exist_ok=True)
    args.output_audit.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(args.output_audit.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())