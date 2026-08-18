from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def validate_disjoint_token_ranges(
    *,
    calibration_range: tuple[int, int],
    selection_ranges: Sequence[tuple[int, int]],
) -> None:
    """Validate half-open calibration/selection token intervals."""

    cal_start, cal_end = (int(value) for value in calibration_range)
    if cal_start < 0 or cal_end <= cal_start:
        raise ValueError("calibration range must be a non-empty non-negative interval.")
    normalized = []
    for start, end in selection_ranges:
        start, end = int(start), int(end)
        if start < 0 or end <= start:
            raise ValueError("selection ranges must be non-empty non-negative intervals.")
        if max(start, cal_start) < min(end, cal_end):
            raise ValueError("selection range overlaps calibration range.")
        normalized.append((start, end))
    for index, left in enumerate(normalized):
        for right in normalized[index + 1 :]:
            if max(left[0], right[0]) < min(left[1], right[1]):
                raise ValueError("selection ranges overlap each other.")


def select_stable_candidate(
    *,
    fold_ppl: Mapping[str, Sequence[float]],
    fallback: str,
) -> dict:
    """Select a refinement only when it wins a strict majority and mean PPL.

    Ties are assigned to the fallback, making the decision conservative without
    introducing a tuned improvement threshold.
    """

    if fallback not in fold_ppl:
        raise ValueError("fallback candidate is missing.")
    candidates = sorted(str(candidate) for candidate in fold_ppl)
    fold_counts = {len(fold_ppl[candidate]) for candidate in candidates}
    if len(fold_counts) != 1 or not fold_counts or next(iter(fold_counts)) == 0:
        raise ValueError("candidates must have the same non-zero fold count.")
    fold_count = next(iter(fold_counts))
    values = {
        candidate: [float(value) for value in fold_ppl[candidate]]
        for candidate in candidates
    }
    if any(
        not math.isfinite(value) or value <= 0.0
        for rows in values.values()
        for value in rows
    ):
        raise ValueError("fold PPL values must be finite and positive.")

    win_counts = {candidate: 0 for candidate in candidates}
    fold_winners = []
    for fold_idx in range(fold_count):
        minimum = min(values[candidate][fold_idx] for candidate in candidates)
        tied = [
            candidate
            for candidate in candidates
            if values[candidate][fold_idx] == minimum
        ]
        winner = fallback if fallback in tied else tied[0]
        win_counts[winner] += 1
        fold_winners.append(winner)

    means = {
        candidate: sum(rows) / fold_count for candidate, rows in values.items()
    }
    majority = fold_count // 2 + 1
    eligible = [
        candidate
        for candidate in candidates
        if candidate != fallback
        and win_counts[candidate] >= majority
        and means[candidate] < means[fallback]
    ]
    if eligible:
        selected = min(eligible, key=lambda candidate: (means[candidate], candidate))
        reason = "stable_majority_and_lower_mean"
    else:
        selected = fallback
        reason = "fallback_no_stable_refinement"
    return {
        "selected": selected,
        "fallback": fallback,
        "selection_reason": reason,
        "fold_count": fold_count,
        "required_majority": majority,
        "fold_winners": fold_winners,
        "win_counts": win_counts,
        "mean_ppl": means,
    }
