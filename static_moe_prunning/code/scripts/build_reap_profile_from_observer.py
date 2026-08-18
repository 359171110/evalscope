from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from src.reap_bridge import build_reap_profile_payload
from src.source_identity import source_tree_identity


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive an exact-budget REAP profile from a frozen official observer artifact."
    )
    parser.add_argument("--observer-artifact", type=Path, required=True)
    parser.add_argument("--calibration-cache", type=Path, required=True)
    parser.add_argument("--channel-cache", type=Path, required=True)
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument("--experts-to-prune-per-layer", type=int, required=True)
    parser.add_argument("--num-blocks", type=int, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    observer_path = args.observer_artifact.expanduser().resolve()
    calibration_path = args.calibration_cache.expanduser().resolve()
    channel_path = args.channel_cache.expanduser().resolve()
    observer = torch.load(observer_path, map_location="cpu", weights_only=True)
    calibration = torch.load(calibration_path, map_location="cpu", weights_only=True)
    if observer.get("method") != "official_reap_observer":
        raise ValueError("observer artifact is not an official REAP observer artifact.")
    if observer.get("renormalize_router_weights") is not True:
        raise ValueError("observer artifact must use renormalized router weights.")
    calibration_hash = file_sha256(calibration_path)
    if observer.get("calibration_cache_sha256") != calibration_hash:
        raise ValueError("observer calibration cache SHA256 does not match the requested artifact.")
    if observer.get("input_ids_sha256") != calibration.get("input_ids_sha256"):
        raise ValueError("observer token IDs do not match the requested calibration artifact.")
    profile = build_reap_profile_payload(
        observer_data=observer["observer_data"],
        model_path=str(observer["model_path"]),
        calibration_payload=calibration,
        calibration_file_sha256=calibration_hash,
        channel_file_sha256=file_sha256(channel_path),
        official_reap_commit=str(observer["official_reap_commit"]),
        num_blocks=args.num_blocks,
        experts_to_prune_per_layer=args.experts_to_prune_per_layer,
        top_k=args.top_k,
        renormalize_router_weights=True,
    )
    profile["cache_provenance"]["observer"] = {
        "path": str(observer_path),
        "sha256": file_sha256(observer_path),
        "official_reap_commit": observer["official_reap_commit"],
    }
    profile["bridge_source_identity"] = source_tree_identity(
        Path(__file__).resolve().parents[2],
        pathspecs=("code/src", "code/scripts/build_reap_profile_from_observer.py"),
    )
    profile["cache_provenance"]["channel"]["path"] = str(channel_path)
    args.output_profile.parent.mkdir(parents=True, exist_ok=True)
    torch.save(profile, args.output_profile)
    summary = {
        key: value
        for key, value in profile.items()
        if key not in {"profile_widths", "retained_expert_mask", "official_reap_saliency"}
    }
    summary["profile_file_sha256"] = file_sha256(args.output_profile)
    args.output_profile.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(args.output_profile.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())