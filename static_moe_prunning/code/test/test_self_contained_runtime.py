from __future__ import annotations

from pathlib import Path


def test_runtime_sources_do_not_import_historical_package() -> None:
    code_dir = Path(__file__).resolve().parents[1]
    offenders = []
    for source_dir in (code_dir / "src", code_dir / "scripts"):
        for path in source_dir.rglob("*.py"):
            if "moe_prune_v2" in path.read_text(encoding="utf-8"):
                offenders.append(path.relative_to(code_dir).as_posix())

    assert offenders == []