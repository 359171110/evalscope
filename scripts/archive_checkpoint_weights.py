from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import stat
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RECOVERY_MANIFEST = "weight_recovery_manifest.json"
RESTORE_SCRIPT = "restore_weights.sh"


@dataclass(frozen=True)
class CheckpointPlan:
    checkpoint_dir: Path
    export_manifest: dict[str, Any]
    export_script: Path
    export_arguments: list[str]
    shards: list[Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create recovery files beside exported checkpoints and optionally delete their weight shards."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Checkpoint directory, experiment directory, or root containing pruning_export_manifest.json files.",
    )
    parser.add_argument("--delete", action="store_true", help="Delete weight shards after writing recovery files.")
    parser.add_argument("--yes", action="store_true", help="Confirm deletion; required together with --delete.")
    parser.add_argument(
        "--recovery-python",
        type=Path,
        default=Path(sys.executable),
        help="Python interpreter recorded for checkpoint regeneration.",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_checkpoint_dirs(paths: list[Path]) -> list[Path]:
    checkpoint_dirs: set[Path] = set()
    for raw_path in paths:
        path = raw_path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_file():
            if path.name != "pruning_export_manifest.json":
                raise ValueError(f"Expected pruning_export_manifest.json, got: {path}")
            checkpoint_dirs.add(path.parent)
            continue
        if (path / "pruning_export_manifest.json").is_file():
            checkpoint_dirs.add(path)
            continue
        checkpoint_dirs.update(manifest.parent for manifest in path.rglob("pruning_export_manifest.json"))
    return sorted(checkpoint_dirs)


def require_dependency(path_value: Any, label: str) -> Path:
    if not path_value:
        raise ValueError(f"Export manifest does not record {label}.")
    path = Path(path_value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Required {label} is missing: {path}")
    return path


def validate_recovery_python(path: Path) -> Path:
    recovery_python = path.expanduser().resolve()
    if not recovery_python.is_file():
        raise FileNotFoundError(f"Recovery Python is missing: {recovery_python}")
    subprocess.run(
        [str(recovery_python), "-c", "import torch, safetensors"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return recovery_python


def resolve_exporter(export_manifest: dict[str, Any]) -> tuple[Path, list[str]]:
    source_model = require_dependency(export_manifest.get("source_model"), "source_model")
    channel_cache = require_dependency(export_manifest.get("channel_cache"), "channel_cache")
    retained_channels = export_manifest.get("retained_channels")
    if not isinstance(retained_channels, int):
        raise ValueError("Export manifest does not record an integer retained_channels.")

    export_script_value = export_manifest.get("export_script")
    if export_script_value:
        export_script = require_dependency(export_script_value, "export_script")
    else:
        export_script = (Path(__file__).resolve().parents[1] / "WICK/export_uniform_qwen3_moe.py").resolve()
        if not export_script.is_file():
            raise FileNotFoundError(f"Fallback exporter is missing: {export_script}")

    arguments = [
        "--model-path",
        str(source_model),
        "--channel-cache",
        str(channel_cache),
        "--output-dir",
        "__OUTPUT_DIR__",
        "--retained-channels",
        str(retained_channels),
    ]
    profile_value = export_manifest.get("profile")
    if profile_value:
        profile = require_dependency(profile_value, "profile")
        arguments[2:2] = ["--profile", str(profile)]
    return export_script, arguments


def listed_weight_shards(checkpoint_dir: Path, export_manifest: dict[str, Any]) -> list[Path]:
    exported_shards = export_manifest.get("exported_shards")
    if isinstance(exported_shards, dict) and exported_shards:
        shard_names = sorted(exported_shards)
    else:
        shard_names = sorted(path.name for path in checkpoint_dir.glob("model-*.safetensors"))
    shards = [checkpoint_dir / name for name in shard_names]
    missing = [path for path in shards if not path.is_file()]
    if missing and len(missing) != len(shards):
        raise FileNotFoundError(f"Checkpoint is partially archived; missing {len(missing)} of {len(shards)} shards: {checkpoint_dir}")
    return shards


def build_recovery_manifest(
    checkpoint_dir: Path,
    export_manifest: dict[str, Any],
    export_script: Path,
    export_arguments: list[str],
    shards: list[Path],
    recovery_python: Path,
) -> dict[str, Any]:
    exported_hashes = export_manifest.get("exported_shards", {})
    shard_records = []
    for shard in shards:
        recorded_hash = exported_hashes.get(shard.name)
        shard_records.append(
            {
                "name": shard.name,
                "size_bytes": shard.stat().st_size if shard.exists() else None,
                "sha256": recorded_hash or file_sha256(shard) if shard.exists() else recorded_hash,
            }
        )
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_dir": str(checkpoint_dir),
        "export_manifest": "pruning_export_manifest.json",
        "export_manifest_sha256": file_sha256(checkpoint_dir / "pruning_export_manifest.json"),
        "python_executable": str(recovery_python),
        "export_script": str(export_script),
        "export_arguments": export_arguments,
        "weight_shards": shard_records,
        "total_weight_bytes": sum(record["size_bytes"] or 0 for record in shard_records),
    }


def render_restore_script() -> str:
    return """#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


checkpoint_dir = Path(__file__).resolve().parent
manifest = json.loads((checkpoint_dir / "weight_recovery_manifest.json").read_text(encoding="utf-8"))
existing = [item["name"] for item in manifest["weight_shards"] if (checkpoint_dir / item["name"]).exists()]
if existing:
    raise SystemExit(f"Refusing to overwrite existing weight shards: {existing[:3]}")

with tempfile.TemporaryDirectory(prefix="restore-weights-", dir=checkpoint_dir.parent) as temporary:
    output_dir = Path(temporary) / "checkpoint"
    arguments = [str(output_dir) if value == "__OUTPUT_DIR__" else value for value in manifest["export_arguments"]]
    subprocess.run([manifest["python_executable"], manifest["export_script"], *arguments], check=True)
    for item in manifest["weight_shards"]:
        generated = output_dir / item["name"]
        if not generated.is_file():
            raise FileNotFoundError(generated)
        expected_hash = item.get("sha256")
        if expected_hash and sha256(generated) != expected_hash:
            raise RuntimeError(f"Checksum mismatch for {item['name']}")
    for item in manifest["weight_shards"]:
        (output_dir / item["name"]).replace(checkpoint_dir / item["name"])

print(f"Restored {len(manifest['weight_shards'])} shards in {checkpoint_dir}")
"""


def build_checkpoint_plan(checkpoint_dir: Path) -> CheckpointPlan:
    export_manifest_path = checkpoint_dir / "pruning_export_manifest.json"
    export_manifest = json.loads(export_manifest_path.read_text(encoding="utf-8"))
    export_script, export_arguments = resolve_exporter(export_manifest)
    shards = listed_weight_shards(checkpoint_dir, export_manifest)
    return CheckpointPlan(checkpoint_dir, export_manifest, export_script, export_arguments, shards)


def archive_checkpoint(plan: CheckpointPlan, delete: bool, recovery_python: Path) -> tuple[int, int]:
    checkpoint_dir = plan.checkpoint_dir
    shards = plan.shards
    if not shards:
        print(f"SKIP no weight shards recorded: {checkpoint_dir}")
        return 0, 0

    recovery_manifest = build_recovery_manifest(
        checkpoint_dir, plan.export_manifest, plan.export_script, plan.export_arguments, shards, recovery_python
    )
    recovery_path = checkpoint_dir / RECOVERY_MANIFEST
    recovery_path.write_text(json.dumps(recovery_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    restore_path = checkpoint_dir / RESTORE_SCRIPT
    restore_path.write_text(render_restore_script(), encoding="utf-8")
    restore_path.chmod(restore_path.stat().st_mode | stat.S_IXUSR)

    total_bytes = recovery_manifest["total_weight_bytes"]
    if delete:
        for shard in shards:
            shard.unlink()
        print(f"ARCHIVED {checkpoint_dir} ({len(shards)} shards, {total_bytes / 2**30:.2f} GiB deleted)")
    else:
        command = " ".join(shlex.quote(value) for value in [str(Path(__file__).resolve()), str(checkpoint_dir), "--delete", "--yes"])
        print(f"DRY RUN {checkpoint_dir} ({len(shards)} shards, {total_bytes / 2**30:.2f} GiB); delete with:\n  {command}")
    return len(shards), total_bytes


def main() -> int:
    args = parse_args()
    if args.delete != args.yes:
        raise SystemExit("Deletion requires both --delete and --yes; use neither for a dry run.")
    checkpoint_dirs = find_checkpoint_dirs(args.paths)
    if not checkpoint_dirs:
        raise SystemExit("No pruning_export_manifest.json files found.")
    recovery_python = validate_recovery_python(args.recovery_python)
    plans = [build_checkpoint_plan(checkpoint_dir) for checkpoint_dir in checkpoint_dirs]
    if args.delete and any(not plan.shards for plan in plans):
        raise SystemExit("Deletion refused because at least one checkpoint has no present weight shards.")

    total_shards = 0
    total_bytes = 0
    for plan in plans:
        shard_count, checkpoint_bytes = archive_checkpoint(plan, delete=args.delete, recovery_python=recovery_python)
        total_shards += shard_count
        total_bytes += checkpoint_bytes
    action = "deleted" if args.delete else "eligible"
    print(f"TOTAL {len(checkpoint_dirs)} checkpoints, {total_shards} shards, {total_bytes / 2**30:.2f} GiB {action}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())