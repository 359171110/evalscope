#!/usr/bin/env bash
set -Eeuo pipefail

# Usage:
#   1. Preview/archive: set TARGET_PATH to a checkpoint, experiment, or result root;
#      set ACTION="preview" or ACTION="delete", then run:
#        bash scripts/archive_checkpoint_weights.sh
#   2. Restore: set TARGET_PATH to an archived checkpoint or experiment directory;
#      set ACTION="restore", then run the same command. A batch result root is also supported.
#   3. Restore one checkpoint directly without this wrapper:
#        /path/to/checkpoint/restore_weights.sh
#      The direct command still verifies every regenerated shard against its recorded SHA-256.
#
# Change only these settings for normal use.
TARGET_PATH="${TARGET_PATH:-/data01/home/xinpei.gao/evalscope/result}"
ACTION="${ACTION:-delete}"  # preview | delete | restore
# ALLOW_DELETE="${ALLOW_DELETE:-false}"
ALLOW_DELETE="true"

ROOT="/data01/home/xinpei.gao/evalscope"
RESULT_ROOT="$ROOT/result"
ARCHIVER="$ROOT/scripts/archive_checkpoint_weights.py"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RECOVERY_PYTHON="${RECOVERY_PYTHON:-/data01/home/xuzk/anaconda3/envs/xhquant/bin/python}"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

[[ -f "$ARCHIVER" ]] || die "Archiver not found: $ARCHIVER"
[[ -x "$RECOVERY_PYTHON" ]] || die "Recovery Python is not executable: $RECOVERY_PYTHON"
[[ "$ACTION" == "preview" || "$ACTION" == "delete" || "$ACTION" == "restore" ]] \
    || die "ACTION must be preview, delete, or restore."
[[ -e "$TARGET_PATH" ]] || die "TARGET_PATH does not exist: $TARGET_PATH"

TARGET_PATH="$($PYTHON_BIN -c 'import pathlib, sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve())' "$TARGET_PATH")"
RESULT_ROOT="$($PYTHON_BIN -c 'import pathlib, sys; print(pathlib.Path(sys.argv[1]).resolve())' "$RESULT_ROOT")"
case "$TARGET_PATH" in
    "$RESULT_ROOT"|"$RESULT_ROOT"/*) ;;
    *) die "Refusing a target outside RESULT_ROOT: $TARGET_PATH" ;;
esac

mapfile -t CHECKPOINT_DIRS < <(
    "$PYTHON_BIN" - "$TARGET_PATH" <<'PY'
from pathlib import Path
import sys

target = Path(sys.argv[1])
if target.is_file():
    if target.name != "pruning_export_manifest.json":
        raise SystemExit(f"Unsupported file target: {target}")
    print(target.parent)
elif (target / "pruning_export_manifest.json").is_file():
    print(target)
else:
    for manifest in sorted(target.rglob("pruning_export_manifest.json")):
        print(manifest.parent)
PY
)

((${#CHECKPOINT_DIRS[@]} > 0)) || die "No checkpoint export manifests found under: $TARGET_PATH"

if ((${#CHECKPOINT_DIRS[@]} == 1)); then
    SCOPE="single checkpoint/experiment"
else
    SCOPE="batch (${#CHECKPOINT_DIRS[@]} checkpoints)"
fi

echo "Target: $TARGET_PATH"
echo "Detected scope: $SCOPE"

if [[ "$ACTION" == "restore" ]]; then
    RESTORE_SUMMARY="$($PYTHON_BIN - "${CHECKPOINT_DIRS[@]}" <<'PY'
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

checkpoint_dirs = [Path(value).resolve() for value in sys.argv[1:]]
restore_dirs = []
total_bytes = 0
total_shards = 0
for checkpoint_dir in checkpoint_dirs:
    recovery_path = checkpoint_dir / "weight_recovery_manifest.json"
    restore_path = checkpoint_dir / "restore_weights.sh"
    if not recovery_path.is_file() or not restore_path.is_file():
        raise SystemExit(f"Recovery files are incomplete: {checkpoint_dir}")
    if not os.access(restore_path, os.X_OK):
        raise SystemExit(f"Restore script is not executable: {restore_path}")

    recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
    if Path(recovery["checkpoint_dir"]).resolve() != checkpoint_dir:
        raise SystemExit(f"Recovery manifest points to another directory: {recovery_path}")
    for dependency in (recovery["python_executable"], recovery["export_script"], *recovery["export_arguments"]):
        if isinstance(dependency, str) and dependency.startswith("/") and not Path(dependency).exists():
            raise SystemExit(f"Missing recovery dependency: {dependency}")

    shard_records = recovery.get("weight_shards", [])
    if not shard_records or any(not item.get("sha256") for item in shard_records):
        raise SystemExit(f"Recovery manifest lacks shard checksums: {recovery_path}")
    present = [(checkpoint_dir / item["name"]).is_file() for item in shard_records]
    if any(present) and not all(present):
        raise SystemExit(f"Refusing partially present checkpoint: {checkpoint_dir}")
    if all(present):
        print(f"ALREADY_PRESENT\t{checkpoint_dir}", file=sys.stderr)
        continue
    if any(item.get("size_bytes") is None for item in shard_records):
        raise SystemExit(f"Recovery manifest lacks shard sizes: {recovery_path}")
    restore_dirs.append(checkpoint_dir)
    total_shards += len(shard_records)
    total_bytes += sum(item["size_bytes"] for item in shard_records)

print(f"{len(restore_dirs)}\t{total_shards}\t{total_bytes}")
for checkpoint_dir in restore_dirs:
    print(checkpoint_dir)
PY
)" || die "Restore preflight failed. Nothing was restored."

    mapfile -t RESTORE_LINES <<<"$RESTORE_SUMMARY"
    RESTORE_HEADER="${RESTORE_LINES[0]}"
    RESTORE_DIRS=("${RESTORE_LINES[@]:1}")
    IFS=$'\t' read -r RESTORE_COUNT RESTORE_SHARDS RESTORE_BYTES <<<"$RESTORE_HEADER"
    if ((RESTORE_COUNT == 0)); then
        echo "All detected checkpoints already contain their weight shards. Nothing to restore."
        exit 0
    fi
    REQUIRED_GIB="$($PYTHON_BIN -c 'import sys; print(f"{int(sys.argv[1]) / 2**30:.2f}")' "$RESTORE_BYTES")"
    AVAILABLE_BYTES="$(df -PB1 "$TARGET_PATH" | awk 'NR == 2 {print $4}')"
    ((AVAILABLE_BYTES >= RESTORE_BYTES)) \
        || die "Insufficient disk space: need $REQUIRED_GIB GiB before temporary export overhead."

    echo "Restore validated: $RESTORE_COUNT checkpoints, $RESTORE_SHARDS shards, $REQUIRED_GIB GiB"
    [[ -t 0 ]] || die "Restore requires an interactive terminal."
    CONFIRMATION="RESTORE ${RESTORE_COUNT} CHECKPOINTS ${RESTORE_SHARDS} SHARDS"
    printf 'Type exactly "%s" to continue: ' "$CONFIRMATION"
    read -r ANSWER
    [[ "$ANSWER" == "$CONFIRMATION" ]] || die "Confirmation did not match. Nothing was restored."

    for checkpoint_dir in "${RESTORE_DIRS[@]}"; do
        [[ -n "$checkpoint_dir" ]] || continue
        "$checkpoint_dir/restore_weights.sh"
    done

    "$PYTHON_BIN" - "${CHECKPOINT_DIRS[@]}" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

verified = 0
for checkpoint_dir_value in sys.argv[1:]:
    checkpoint_dir = Path(checkpoint_dir_value)
    recovery = json.loads((checkpoint_dir / "weight_recovery_manifest.json").read_text(encoding="utf-8"))
    for item in recovery["weight_shards"]:
        shard = checkpoint_dir / item["name"]
        if not shard.is_file() or shard.stat().st_size != item["size_bytes"]:
            raise SystemExit(f"Restored shard is missing or has the wrong size: {shard}")
        if sha256(shard) != item["sha256"]:
            raise SystemExit(f"Restored shard checksum mismatch: {shard}")
        verified += 1
print(f"Post-restore verification passed: {verified} shards match their recorded SHA-256.")
PY
    exit 0
fi

FILTERED_CHECKPOINTS="$($PYTHON_BIN - "${CHECKPOINT_DIRS[@]}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

for checkpoint_dir_value in sys.argv[1:]:
    checkpoint_dir = Path(checkpoint_dir_value)
    export_path = checkpoint_dir / "pruning_export_manifest.json"
    export_manifest = json.loads(export_path.read_text(encoding="utf-8"))
    exported_shards = export_manifest.get("exported_shards")
    if isinstance(exported_shards, dict) and exported_shards:
        shard_paths = [checkpoint_dir / name for name in sorted(exported_shards)]
    else:
        shard_paths = sorted(checkpoint_dir.glob("model-*.safetensors"))
    if not shard_paths:
        print(f"SKIP no weight shards recorded: {checkpoint_dir}", file=sys.stderr)
        continue

    present = [path.is_file() for path in shard_paths]
    if any(present) and not all(present):
        missing_count = len(present) - sum(present)
        raise SystemExit(
            f"Checkpoint is partially archived; missing {missing_count} of {len(present)} shards: {checkpoint_dir}"
        )
    if not any(present):
        print(f"SKIP weight shards already absent: {checkpoint_dir}", file=sys.stderr)
        continue
    print(checkpoint_dir)
PY
)" || die "Checkpoint-state preflight failed. Nothing was deleted."

mapfile -t CHECKPOINT_DIRS <<<"$FILTERED_CHECKPOINTS"
if ((${#CHECKPOINT_DIRS[@]} == 0)) || [[ -z "${CHECKPOINT_DIRS[0]}" ]]; then
    echo "All detected checkpoints already have no weight shards. Nothing to delete."
    exit 0
fi

echo "Running recovery-manifest preflight..."
"$PYTHON_BIN" "$ARCHIVER" "${CHECKPOINT_DIRS[@]}" --recovery-python "$RECOVERY_PYTHON"

SUMMARY="$($PYTHON_BIN - "${CHECKPOINT_DIRS[@]}" <<'PY'
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

checkpoint_dirs = [Path(value).resolve() for value in sys.argv[1:]]
total_bytes = 0
total_shards = 0
for checkpoint_dir in checkpoint_dirs:
    export_path = checkpoint_dir / "pruning_export_manifest.json"
    recovery_path = checkpoint_dir / "weight_recovery_manifest.json"
    restore_path = checkpoint_dir / "restore_weights.sh"
    if not export_path.is_file() or not recovery_path.is_file() or not restore_path.is_file():
        raise SystemExit(f"Recovery files are incomplete: {checkpoint_dir}")
    if not os.access(restore_path, os.X_OK):
        raise SystemExit(f"Restore script is not executable: {restore_path}")

    recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
    if Path(recovery["checkpoint_dir"]).resolve() != checkpoint_dir:
        raise SystemExit(f"Recovery manifest points to another directory: {recovery_path}")
    for dependency in (recovery["python_executable"], recovery["export_script"], *recovery["export_arguments"]):
        if isinstance(dependency, str) and dependency.startswith("/") and not Path(dependency).exists():
            raise SystemExit(f"Missing recovery dependency: {dependency}")

    shard_records = recovery.get("weight_shards", [])
    if not shard_records:
        raise SystemExit(f"No weight shards recorded: {recovery_path}")
    for record in shard_records:
        shard = checkpoint_dir / record["name"]
        if not shard.is_file():
            raise SystemExit(f"Weight shard is missing before deletion: {shard}")
        expected_size = record.get("size_bytes")
        if expected_size is None or shard.stat().st_size != expected_size:
            raise SystemExit(f"Weight shard size mismatch: {shard}")
        total_bytes += expected_size
        total_shards += 1

print(f"{len(checkpoint_dirs)}\t{total_shards}\t{total_bytes}")
PY
)" || die "Independent recovery validation failed. Nothing was deleted."

IFS=$'\t' read -r CHECKPOINT_COUNT SHARD_COUNT TOTAL_BYTES <<<"$SUMMARY"
TOTAL_GIB="$($PYTHON_BIN -c 'import sys; print(f"{int(sys.argv[1]) / 2**30:.2f}")' "$TOTAL_BYTES")"
echo "Validated: $CHECKPOINT_COUNT checkpoints, $SHARD_COUNT shards, $TOTAL_GIB GiB"

if [[ "$ACTION" == "preview" ]]; then
    echo "Preview complete. No weights were deleted."
    echo "To delete this exact scope, set ACTION=delete and ALLOW_DELETE=true, then run this script again."
    exit 0
fi

[[ "$ALLOW_DELETE" == "true" ]] || die "Deletion is locked. Set ALLOW_DELETE=true after reviewing preview output."
[[ -t 0 ]] || die "Deletion requires an interactive terminal."

CONFIRMATION="DELETE ${CHECKPOINT_COUNT} CHECKPOINTS ${SHARD_COUNT} SHARDS"
echo
echo "This will delete $TOTAL_GIB GiB of regenerable weight shards only."
echo "Recovery manifests, restore scripts, configs, tokenizers, reports, and logs will remain."
printf 'Type exactly "%s" to continue: ' "$CONFIRMATION"
read -r ANSWER
[[ "$ANSWER" == "$CONFIRMATION" ]] || die "Confirmation did not match. Nothing was deleted."

"$PYTHON_BIN" "$ARCHIVER" "${CHECKPOINT_DIRS[@]}" --recovery-python "$RECOVERY_PYTHON" --delete --yes

"$PYTHON_BIN" - "${CHECKPOINT_DIRS[@]}" <<'PY'
from pathlib import Path
import json
import sys

for checkpoint_dir_value in sys.argv[1:]:
    checkpoint_dir = Path(checkpoint_dir_value)
    recovery_path = checkpoint_dir / "weight_recovery_manifest.json"
    restore_path = checkpoint_dir / "restore_weights.sh"
    if not recovery_path.is_file() or not restore_path.is_file():
        raise SystemExit(f"Post-delete recovery files are missing: {checkpoint_dir}")
    recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
    remaining = [item["name"] for item in recovery["weight_shards"] if (checkpoint_dir / item["name"]).exists()]
    if remaining:
        raise SystemExit(f"Post-delete validation found remaining recorded shards: {checkpoint_dir}")

print("Post-delete validation passed: all recorded shards are absent and recovery files remain.")
PY