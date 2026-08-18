from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.crossfit_parent_selection import select_stable_candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a frozen static profile from train-only cross-fit PPL folds."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_result(path: Path) -> dict:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != 1:
        raise ValueError("each selection result must contain exactly one row.")
    row = rows[0]
    if row.get("split") != "train":
        raise ValueError("cross-fit selection results must use the train split.")
    if not row.get("profile_frozen_before_evaluation"):
        raise ValueError("candidate profile must be frozen before selection.")
    if row.get("test_metrics_used_for_profile") is not False:
        raise ValueError("candidate profile must be test-independent.")
    return row


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    fallback = str(manifest["fallback"])
    candidates = manifest["candidates"]
    fold_ppl = {}
    provenance = {}
    for candidate, spec in candidates.items():
        profile_path = Path(spec["profile_path"]).expanduser().resolve()
        expected_profile_digest = str(spec["profile_file_sha256"])
        if not profile_path.is_file():
            raise FileNotFoundError(f"candidate profile does not exist: {profile_path}")
        if _file_sha256(profile_path) != expected_profile_digest:
            raise ValueError(f"candidate {candidate} profile SHA256 mismatch.")
        result_paths = [Path(path) for path in spec["result_paths"]]
        rows = [_load_result(path) for path in result_paths]
        for row in rows:
            if Path(str(row.get("profile_path", ""))).expanduser().resolve() != profile_path:
                raise ValueError(f"candidate {candidate} result profile path mismatch.")
            if row.get("profile_file_sha256") != expected_profile_digest:
                raise ValueError(f"candidate {candidate} result profile SHA256 mismatch.")
        fold_ppl[candidate] = [float(row["ppl"]) for row in rows]
        provenance[candidate] = {
            "profile_path": str(profile_path),
            "profile_file_sha256": expected_profile_digest,
            "selection_results": [str(path.resolve()) for path in result_paths],
            "selection_protocols": [row.get("protocol_name") for row in rows],
        }
    decision = select_stable_candidate(fold_ppl=fold_ppl, fallback=fallback)
    selected = decision["selected"]
    payload = {
        "schema_version": 1,
        "method": "crossfit_stable_parent_selection",
        "selection_split": "train",
        "test_metrics_used": False,
        **decision,
        "selected_profile_path": provenance[selected]["profile_path"],
        "selected_profile_file_sha256": provenance[selected][
            "profile_file_sha256"
        ],
        "candidate_provenance": provenance,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
