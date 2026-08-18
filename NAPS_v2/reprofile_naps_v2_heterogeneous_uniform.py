from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a heterogeneous artifact to a uniform medium-width profile.")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, help="Uniform width; defaults to the artifact medium width.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact_dir = args.artifact_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    profile = torch.load(artifact_dir / "profile.pt", map_location="cpu", weights_only=True)
    width_options = tuple(int(value) for value in profile["width_options"].tolist())
    if len(width_options) != 3:
        raise ValueError("Expected small, medium, and large width options")
    uniform_width = width_options[1] if args.width is None else int(args.width)
    if uniform_width not in width_options:
        raise ValueError(f"Width {uniform_width} is not present in artifact width options {width_options}")
    block_size = int(profile["channel_block_size"])
    if uniform_width % block_size:
        raise ValueError("Uniform width is not block-aligned")
    profile_widths = torch.full_like(profile["profile_widths"], uniform_width // block_size)
    profile["profile_widths"] = profile_widths
    profile["total_blocks"] = int(profile_widths.sum().item())
    profile["allocation_scope"] = "per_expert_uniform_medium"
    profile["naps"] = {
        **profile.get("naps", {}),
        "width_assignment": "uniform_medium_control",
        "uniform_width": uniform_width,
        "source_artifact": str(artifact_dir),
    }
    torch.save(profile, output_dir / "profile.pt")

    for name in ("rankings.pt", "routing_audit.pt"):
        shutil.copy2(artifact_dir / name, output_dir / name)

    diagnostics = json.loads((artifact_dir / "diagnostics.json").read_text(encoding="utf-8"))
    diagnostics["profile_widths"] = (profile_widths * block_size).tolist()
    diagnostics["profile_construction"] = "uniform_medium_control"
    diagnostics["source_artifact"] = str(artifact_dir)
    for record in diagnostics["records"]:
        record["assigned_width"] = uniform_width
    (output_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())