from __future__ import annotations

import argparse
import shutil
from pathlib import Path


MODEL_ID = "qwen3-mixed-512x1024-global-quick9-50pct-aimer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consolidate non-overlapping AIMER subset prediction shards.")
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--sources", type=Path, nargs="+", required=True)
    parser.add_argument("--expected-records-per-file", type=int, default=None)
    return parser.parse_args()


def _record_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def consolidate_predictions(
    destination: Path,
    sources: list[Path],
    expected_records_per_file: int | None = None,
) -> list[Path]:
    if expected_records_per_file is not None and expected_records_per_file <= 0:
        raise ValueError("expected_records_per_file must be positive.")
    target_root = destination.expanduser().resolve() / "predictions" / MODEL_ID
    target_root.mkdir(parents=True, exist_ok=True)
    candidates: dict[str, list[Path]] = {}
    for source in sources:
        source_root = source.expanduser().resolve() / "predictions" / MODEL_ID
        if not source_root.is_dir():
            raise FileNotFoundError(f"Prediction source does not exist: {source_root}")
        for path in sorted(source_root.glob("*.jsonl")):
            if expected_records_per_file is not None and _record_count(path) != expected_records_per_file:
                continue
            candidates.setdefault(path.name, []).append(path)

    copied = []
    for name, paths in sorted(candidates.items()):
        selected = paths[0]
        if any(path.read_bytes() != selected.read_bytes() for path in paths[1:]):
            if expected_records_per_file is not None:
                raise ValueError(f"Conflicting complete prediction shards for {name}")
            raise ValueError(f"Conflicting prediction shard for {name}")
        target = target_root / name
        if target.exists():
            if target.read_bytes() != selected.read_bytes():
                raise ValueError(f"Conflicting prediction shard for {name}")
            continue
        shutil.copy2(selected, target)
        copied.append(target)
    return copied


def main() -> int:
    args = parse_args()
    for path in consolidate_predictions(
        args.destination,
        list(args.sources),
        expected_records_per_file=args.expected_records_per_file,
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())