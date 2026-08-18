from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_tree_identity(repository: Path, *, pathspecs: tuple[str, ...]) -> dict:
    """Hash tracked and untracked runtime files under explicit Git pathspecs."""

    root = repository.expanduser().resolve()
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    files_output = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            *pathspecs,
        ],
        check=True,
        capture_output=True,
    ).stdout
    relative_paths = sorted(
        path.decode("utf-8")
        for path in files_output.split(b"\0")
        if path
    )
    digest = hashlib.sha256()
    file_count = 0
    for relative_path in relative_paths:
        path = root / relative_path
        if not path.is_file():
            continue
        path_bytes = relative_path.encode("utf-8")
        digest.update(len(path_bytes).to_bytes(8, byteorder="big"))
        digest.update(path_bytes)
        digest.update(bytes.fromhex(_file_sha256(path)))
        file_count += 1
    status = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            *pathspecs,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "repository": str(root),
        "commit": commit,
        "runtime_pathspecs": list(pathspecs),
        "runtime_tree_dirty": bool(status.strip()),
        "runtime_tree_sha256": digest.hexdigest(),
        "runtime_file_count": file_count,
    }