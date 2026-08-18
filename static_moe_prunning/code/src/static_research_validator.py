from __future__ import annotations

from typing import Iterable, Mapping


REQUIRED_BASELINES = (
    "uniform",
    "rms",
    "route_rms",
    "dual_route_rms",
    "dynamic_regret",
    "dynamic_expected_utility",
    "expected_utility_gate",
    "expected_utility_top_p",
)
NOVELTY_REFERENCES = (
    "moe-slimming",
    "mose",
    "pop",
    "reap",
    "maestro",
    "flap",
    "moe-pruner",
    "mixture compressor",
    "dtop-p",
)


def _protocol_issues(row: Mapping[str, object]) -> list[str]:
    mode = str(row.get("mode", "<missing>"))
    issues = []
    expected = {
        "windows": 114,
        "tokens": 233368,
        "sequence_length": 2048,
        "dataset": "wikitext-2-raw-v1",
        "split": "test",
        "standard_protocol": True,
        "profile_frozen_before_evaluation": True,
        "test_metrics_used_for_profile": False,
    }
    for key, value in expected.items():
        if row.get(key) != value:
            issues.append(f"{mode}: {key} must equal {value!r}.")
    for key in ("profile_sha256", "profile_file_sha256"):
        if not row.get(key):
            issues.append(f"{mode}: missing {key}.")
    provenance = row.get("cache_provenance")
    channel = provenance.get("channel", {}) if isinstance(provenance, dict) else {}
    if channel.get("split") != "train" or not channel.get("sha256"):
        issues.append(f"{mode}: channel provenance must be train-only with SHA256.")
    try:
        total = int(row.get("total_profile_blocks", -1))
        maximum = int(row.get("maximum_profile_blocks", -1))
        ratio = float(row.get("structural_pruning_ratio", -1.0))
        ppl = float(row.get("ppl", float("nan")))
    except (TypeError, ValueError):
        issues.append(f"{mode}: numeric metric fields are malformed.")
        return issues
    if total <= 0 or maximum <= 0 or total > maximum:
        issues.append(f"{mode}: invalid structural block budget.")
    elif abs(ratio - (1.0 - total / maximum)) > 1.0e-9:
        issues.append(f"{mode}: structural pruning ratio does not match block budget.")
    if not (ppl > 0.0 and ppl < float("inf")):
        issues.append(f"{mode}: PPL must be finite and positive.")
    return issues


def _compact(row: Mapping[str, object]) -> dict[str, object]:
    return {
        key: row.get(key)
        for key in (
            "mode",
            "ppl",
            "total_profile_blocks",
            "maximum_profile_blocks",
            "structural_pruning_ratio",
            "routed_compute_pruning_ratio",
            "correction_mode",
            "profile_sha256",
            "profile_path",
        )
    }


def validate_static_research(
    rows: Iterable[Mapping[str, object]],
    *,
    novelty_report: str,
) -> dict[str, object]:
    rows = list(rows)
    issues: list[str] = []
    valid_rows: list[Mapping[str, object]] = []
    for row in rows:
        # Smoke artifacts coexist with formal results by design.  They are
        # ignored as evidence rather than treated as mission failures.
        if row.get("standard_protocol") is not True:
            continue
        row_issues = _protocol_issues(row)
        issues.extend(row_issues)
        if not row_issues:
            valid_rows.append(row)

    selected: dict[str, Mapping[str, object]] = {}
    for mode in (*REQUIRED_BASELINES, "expected_utility_dual"):
        candidates = [row for row in valid_rows if row.get("mode") == mode]
        if not candidates:
            issues.append(f"Missing valid full-protocol result for {mode}.")
            continue
        selected[mode] = min(candidates, key=lambda row: float(row["ppl"]))

    best_baseline = None
    candidate = selected.get("expected_utility_dual")
    available_baseline_modes = [
        mode for mode in REQUIRED_BASELINES if mode in selected
    ]
    if available_baseline_modes:
        baselines = [selected[mode] for mode in available_baseline_modes]
        best_baseline = min(baselines, key=lambda row: float(row["ppl"]))
        if candidate is not None:
            matched = (
                int(candidate["total_profile_blocks"])
                == int(best_baseline["total_profile_blocks"])
                and int(candidate["maximum_profile_blocks"])
                == int(best_baseline["maximum_profile_blocks"])
                and candidate.get("correction_mode")
                == best_baseline.get("correction_mode")
            )
            if not matched:
                issues.append(
                    "expected_utility_dual and the strongest baseline do not have a matched "
                    "structural budget and correction mode."
                )
            if float(candidate["ppl"]) >= float(best_baseline["ppl"]):
                issues.append(
                    "expected_utility_dual must strictly beat the strongest matched static baseline."
                )

    report_lower = novelty_report.lower()
    missing_references = [
        reference for reference in NOVELTY_REFERENCES if reference not in report_lower
    ]
    if missing_references:
        issues.append(
            "novelty report must explicitly separate the method from: "
            + ", ".join(missing_references)
            + "."
        )

    return {
        "passed": not issues,
        "status": "passed" if not issues else "in_progress",
        "issues": issues,
        "required_baselines": list(REQUIRED_BASELINES),
        "best_baseline": None if best_baseline is None else _compact(best_baseline),
        "candidate": None if candidate is None else _compact(candidate),
        "full_protocol_rows_seen": len(valid_rows),
    }
