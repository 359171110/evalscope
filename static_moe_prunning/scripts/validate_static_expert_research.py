from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.static_research_validator import validate_static_research


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the full static-expert research mission."
    )
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=Path("experiments/results"))
    parser.add_argument(
        "--report", type=Path, default=Path("docs/STATIC_EXPERT_RESEARCH_REPORT.md")
    )
    return parser.parse_args()


def load_rows(results_root: Path) -> tuple[list[dict], list[str]]:
    rows = []
    sources = []
    for path in sorted(results_root.glob("**/static_expert_wikitext_ppl.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Expected a list in {path}.")
        rows.extend(payload)
        sources.append(str(path))
    return rows, sources


def main() -> int:
    args = parse_args()
    rows, sources = load_rows(args.results_root)
    report_text = args.report.read_text(encoding="utf-8") if args.report.exists() else ""
    result = validate_static_research(rows, novelty_report=report_text)
    result.update(
        {
            "validated_at": datetime.now(timezone.utc).isoformat(),
            "result_sources": sources,
            "output_artifact_path": str(args.report),
            "summary": (
                "Conditional Dual-Utility Distillation strictly improves full WikiText-2 PPL "
                "over all matched static baselines."
                if result["passed"]
                else "Static expert research mission has not passed all completion gates."
            ),
        }
    )
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(args.result.resolve())
    for issue in result["issues"]:
        print(f"- {issue}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
