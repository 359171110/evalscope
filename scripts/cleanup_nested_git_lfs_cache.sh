#!/usr/bin/env bash
set -Eeuo pipefail

# Removes only nested Git LFS object caches under result/ and WICK/.
# The repository's top-level .git directory and checkpoint working files are untouched.
#
# Preview:
#   bash scripts/cleanup_nested_git_lfs_cache.sh
# Delete after reviewing the preview:
#   ACTION=delete bash scripts/cleanup_nested_git_lfs_cache.sh

ROOT="/data01/home/xinpei.gao/evalscope"
ACTION="${ACTION:-preview}"  # preview | delete
SEARCH_ROOTS=("$ROOT/result" "$ROOT/WICK")

die() {
    echo "ERROR: $*" >&2
    exit 1
}

[[ "$ACTION" == "preview" || "$ACTION" == "delete" ]] || die "ACTION must be preview or delete."

for search_root in "${SEARCH_ROOTS[@]}"; do
    [[ -d "$search_root" ]] || die "Search root does not exist: $search_root"
done

mapfile -d '' -t LFS_OBJECT_DIRS < <(
    find "${SEARCH_ROOTS[@]}" -xdev -type d -path '*/.git/lfs/objects' -prune -print0 2>/dev/null
)

if ((${#LFS_OBJECT_DIRS[@]} == 0)); then
    echo "No nested Git LFS object caches found."
    exit 0
fi

TOTAL_BYTES=0
echo "Nested Git LFS object caches:"
for object_dir in "${LFS_OBJECT_DIRS[@]}"; do
    case "$object_dir" in
        "$ROOT/result"/*/\.git/lfs/objects|"$ROOT/WICK"/*/\.git/lfs/objects) ;;
        *) die "Refusing unexpected path: $object_dir" ;;
    esac

    bytes="$(du -s -B1 "$object_dir" | awk '{print $1}')"
    TOTAL_BYTES=$((TOTAL_BYTES + bytes))
    gib="$(awk -v bytes="$bytes" 'BEGIN {printf "%.2f", bytes / 2^30}')"
    printf '  %8s GiB  %s\n' "$gib" "$object_dir"
done

TOTAL_GIB="$(awk -v bytes="$TOTAL_BYTES" 'BEGIN {printf "%.2f", bytes / 2^30}')"
COUNT="${#LFS_OBJECT_DIRS[@]}"
echo
echo "Total: $COUNT caches, $TOTAL_GIB GiB"
echo "Only .git/lfs/objects contents will be removed; model files and the top-level repository .git are preserved."

if [[ "$ACTION" == "preview" ]]; then
    echo "Preview complete. Run with ACTION=delete to remove these caches."
    exit 0
fi

[[ -t 0 ]] || die "Deletion requires an interactive terminal."
CONFIRMATION="DELETE ${COUNT} LFS CACHES"
printf 'Type exactly "%s" to continue: ' "$CONFIRMATION"
read -r answer
[[ "$answer" == "$CONFIRMATION" ]] || die "Confirmation did not match. Nothing was deleted."

for object_dir in "${LFS_OBJECT_DIRS[@]}"; do
    find "$object_dir" -mindepth 1 -delete
done

remaining=0
for object_dir in "${LFS_OBJECT_DIRS[@]}"; do
    if find "$object_dir" -mindepth 1 -print -quit | grep -q .; then
        echo "NOT EMPTY: $object_dir" >&2
        remaining=$((remaining + 1))
    fi
done

((remaining == 0)) || die "$remaining Git LFS object caches were not fully cleared."
echo "Deleted $COUNT nested Git LFS object caches and released approximately $TOTAL_GIB GiB."