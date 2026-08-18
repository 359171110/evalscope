from __future__ import annotations

from NAPS_v2.build_label_free_calibration import (
    Candidate,
    DomainSpec,
    candidates_from_rows,
    deduplicate_candidates,
    render_arc,
    render_code,
    render_gsm8k,
    render_hellaswag,
    render_mmlu_pro,
    select_splits,
)


def test_renderers_exclude_answer_and_output_fields() -> None:
    code = render_code({"instruction": "Add values", "input": "1 2", "output": "SECRET_CODE"})
    gsm8k = render_gsm8k({"question": "What is 1 + 1?", "answer": "SECRET_MATH"})
    arc = render_arc({
        "question": "Which option?",
        "choices": {"label": ["A", "B"], "text": ["First", "Second"]},
        "answerKey": "SECRET_ARC",
    })
    hellaswag = render_hellaswag({
        "ctx_a": "A person starts",
        "ctx_b": "finishing the task",
        "endings": ["one", "two"],
        "label": "SECRET_HELLASWAG",
    })
    mmlu = render_mmlu_pro({
        "question": "Choose one",
        "options": ["left", "right"],
        "answer": "SECRET_MMLU",
        "answer_index": 1,
        "cot_content": "SECRET_COT",
    })
    rendered = "\n".join((code, gsm8k, arc, hellaswag, mmlu))
    assert "SECRET" not in rendered
    assert "Task: Add values" in code
    assert "Input: 1 2" in code
    assert "A. First" in arc
    assert "Candidate continuations" in hellaswag
    assert "Option 2: right" in mmlu


def test_global_prompt_deduplication_is_deterministic() -> None:
    candidates = [
        Candidate("code", "b", "same prompt", "2", None, None),
        Candidate("gsm8k", "a", "same   prompt", "1", None, None),
        Candidate("arc", "c", "different", "3", None, None),
    ]
    unique, duplicate_count = deduplicate_candidates(candidates)
    assert [(candidate.domain, candidate.source_id) for candidate in unique] == [
        ("gsm8k", "a"),
        ("arc", "c"),
    ]
    assert duplicate_count == 1


def test_candidates_never_serialize_unallowed_record_fields() -> None:
    spec = DomainSpec(
        name="gsm8k",
        path=None,
        columns=("question", "answer"),
        allowed_fields=("question",),
        forbidden_fields=("answer",),
        renderer_version="test",
        render=render_gsm8k,
        stable_id=lambda row: row["question"],
    )
    candidates = candidates_from_rows(
        spec,
        [{"question": "Visible question", "answer": "HIDDEN_ANSWER"}],
        "test-protocol",
    )
    assert candidates[0].prompt == "Question: Visible question"
    assert "HIDDEN_ANSWER" not in repr(candidates[0])


def test_split_selection_keeps_groups_disjoint_and_strata_balanced() -> None:
    quotas = {
        "code": {"fit": 1, "holdout": 1},
        "gsm8k": {"fit": 1, "holdout": 1},
        "arc": {"fit": 1, "holdout": 1},
        "hellaswag": {"fit": 1, "holdout": 1},
        "mmlu_pro": {"fit": 2, "holdout": 2},
    }
    candidates = []
    for domain in ("code", "gsm8k", "arc"):
        for index in range(2):
            candidates.append(Candidate(domain, str(index), f"{domain}-{index}", str(index), None, None))
    candidates.extend([
        Candidate("hellaswag", "0", "hs-0", "0", "video-a", None),
        Candidate("hellaswag", "1", "hs-1", "1", "video-a", None),
        Candidate("hellaswag", "2", "hs-2", "2", "video-b", None),
        Candidate("hellaswag", "3", "hs-3", "3", "video-c", None),
    ])
    candidates.extend([
        Candidate("mmlu_pro", "0", "mmlu-0", "0", None, "biology"),
        Candidate("mmlu_pro", "1", "mmlu-1", "1", None, "math"),
        Candidate("mmlu_pro", "2", "mmlu-2", "2", None, "biology"),
        Candidate("mmlu_pro", "3", "mmlu-3", "3", None, "math"),
    ])

    splits = select_splits(candidates, quotas)

    assert len(splits["fit"]) == 6
    assert len(splits["holdout"]) == 6
    fit_ids = {(candidate.domain, candidate.source_id) for candidate in splits["fit"]}
    holdout_ids = {(candidate.domain, candidate.source_id) for candidate in splits["holdout"]}
    assert not fit_ids & holdout_ids
    hellaswag_groups = [
        candidate.group_id
        for split in splits.values()
        for candidate in split
        if candidate.domain == "hellaswag"
    ]
    assert len(hellaswag_groups) == len(set(hellaswag_groups))
    fit_mmlu_strata = {
        candidate.stratum for candidate in splits["fit"] if candidate.domain == "mmlu_pro"
    }
    holdout_mmlu_strata = {
        candidate.stratum for candidate in splits["holdout"] if candidate.domain == "mmlu_pro"
    }
    assert fit_mmlu_strata == {"biology", "math"}
    assert holdout_mmlu_strata == {"biology", "math"}