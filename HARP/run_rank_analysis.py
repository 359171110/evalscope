"""Run HARP rank-structure analysis on the five eval ranking caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from HARP.rank_analysis_core import analyze_model
from HARP.search_combo_five import MODELS

CACHE_ROOT = Path("/data/xinpeigao/evalscope_results/_artifacts/harp")
OUT_ROOT = Path("/data/xinpeigao/evalscope_results/_artifacts/harp_rank_analysis")
REPORT = Path("/home/xinpeigao/evalscope/HARP/rank_analysis_generated.md")
NARRATIVE = Path("/home/xinpeigao/evalscope/HARP/rank_analysis_report.md")


def main() -> int:
    """Analyze all five families unless --model is set."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODELS), default=None)
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT)
    args = parser.parse_args()
    names = [args.model] if args.model else list(MODELS)
    results = {}
    for name in names:
        cache = CACHE_ROOT / name / "harp_rankings.pt"
        if not cache.is_file():
            raise FileNotFoundError(f"missing ranking cache: {cache}")
        dest = args.output_root / name
        print(f"ANALYZE {name} -> {dest}")
        results[name] = analyze_model(cache, dest, name)
        print(results[name]["conclusion"])
        print()
    if args.model is None:
        write_report(results, args.output_root)
    return 0


def write_report(results: dict[str, object], output_root: Path) -> None:
    """Write the cross-model conclusion markdown."""

    lines = [
        "# HARP rank-structure analysis",
        "",
        "Generated from existing `harp_rankings.pt` caches. Allocator was not changed.",
        f"Artifacts: `{output_root}`",
        "",
        "## Per-model conclusions",
        "",
    ]
    for name, payload in results.items():
        lines.append("```text")
        lines.append(str(payload["conclusion"]))  # type: ignore[index]
        lines.append("```")
        lines.append("")
        special = payload["special_layers"]  # type: ignore[index]
        lines.append(
            f"- special layer sets: top10={special['top10']}, z2={special['z2']}, "
            f"robust={special['robust_z2']}, strong={special['strong']}"
        )
        links = payload["links"]  # type: ignore[index]
        lines.append(f"- layer driver counts: {links['expert_to_layer']['driver_counts']}")
        nulls = payload["nulls"]  # type: ignore[index]
        lines.append(
            f"- layer prefix Jaccard @0.5% noise: {nulls['layer_prefix_jaccard_0.5pct']}"
        )
        lines.append("")
    lines.extend([
        "## Cross-model reading",
        "",
        "- A natural Layer-SP segment is declared only when the change-point gap is in the upper tail, the boundary is not a single endpoint, and perturbation keeps the rank within two places most of the time.",
        "- Channel->Expert and Expert->Layer claims require Spearman/Pearson above the within-parent permutation null.",
        "- `use segment` means a stable head interval could become a HARP budget cut; `use quantile only` means the order is informative but the score curve has no stable gap; `no reliable segment` means neither.",
        "",
        json.dumps({name: payload["manifest"] for name, payload in results.items()}, indent=2),  # type: ignore[index]
        "",
    ])
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {REPORT}")
    print(f"narrative report: {NARRATIVE}")


if __name__ == "__main__":
    raise SystemExit(main())
