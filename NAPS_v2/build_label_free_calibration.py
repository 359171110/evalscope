from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


PROTOCOL_NAME = "channel-label-free-128-v1"
DOMAIN_ORDER = ("code", "gsm8k", "arc", "hellaswag", "mmlu_pro")
SPLIT_QUOTAS = {
    "code": {"fit": 18, "holdout": 6},
    "gsm8k": {"fit": 18, "holdout": 6},
    "arc": {"fit": 18, "holdout": 6},
    "hellaswag": {"fit": 18, "holdout": 6},
    "mmlu_pro": {"fit": 24, "holdout": 8},
}
WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class DomainSpec:
    name: str
    path: Path
    columns: tuple[str, ...]
    allowed_fields: tuple[str, ...]
    forbidden_fields: tuple[str, ...]
    renderer_version: str
    render: Callable[[dict[str, Any]], str]
    stable_id: Callable[[dict[str, Any]], str]
    group_id: Callable[[dict[str, Any]], str] | None = None
    stratum: Callable[[dict[str, Any]], str] | None = None


@dataclass(frozen=True)
class Candidate:
    domain: str
    source_id: str
    prompt: str
    order_key: str
    group_id: str | None
    stratum: str | None


def normalize_text(value: Any) -> str:
    return WHITESPACE.sub(" ", str(value or "").strip())


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def selection_key(protocol_name: str, domain: str, source_id: str) -> str:
    payload = f"{protocol_name}\0{domain}\0{source_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_code(record: dict[str, Any]) -> str:
    instruction = normalize_text(record["instruction"])
    input_text = normalize_text(record.get("input", ""))
    if not instruction:
        raise ValueError("Code calibration row has an empty instruction")
    return f"Task: {instruction}" if not input_text else f"Task: {instruction}\nInput: {input_text}"


def render_gsm8k(record: dict[str, Any]) -> str:
    question = normalize_text(record["question"])
    if not question:
        raise ValueError("GSM8K calibration row has an empty question")
    return f"Question: {question}"


def render_arc(record: dict[str, Any]) -> str:
    question = normalize_text(record["question"])
    choices = record["choices"]
    labels = choices["label"]
    texts = choices["text"]
    if not question or len(labels) != len(texts) or not texts:
        raise ValueError("ARC calibration row has invalid question choices")
    options = "\n".join(
        f"{normalize_text(label)}. {normalize_text(text)}" for label, text in zip(labels, texts)
    )
    return f"Question: {question}\nOptions:\n{options}"


def _capitalize_first(text: str) -> str:
    return text if not text else text[0].upper() + text[1:]


def render_hellaswag(record: dict[str, Any]) -> str:
    context_a = normalize_text(record.get("ctx_a", ""))
    context_b = _capitalize_first(normalize_text(record.get("ctx_b", "")))
    context = normalize_text(f"{context_a} {context_b}")
    endings = [normalize_text(value) for value in record["endings"]]
    if not context or not endings or any(not ending for ending in endings):
        raise ValueError("HellaSwag calibration row has invalid context or endings")
    options = "\n".join(f"Option {index + 1}: {ending}" for index, ending in enumerate(endings))
    return f"Context: {context}\nCandidate continuations:\n{options}"


def render_mmlu_pro(record: dict[str, Any]) -> str:
    question = normalize_text(record["question"])
    options = [normalize_text(value) for value in record["options"]]
    if not question or not options or any(not option for option in options):
        raise ValueError("MMLU-Pro calibration row has invalid question or options")
    rendered = "\n".join(f"Option {index + 1}: {option}" for index, option in enumerate(options))
    return f"Question: {question}\nOptions:\n{rendered}"


def build_domain_specs(data_root: Path) -> list[DomainSpec]:
    return [
        DomainSpec(
            name="code",
            path=data_root / "python_code_instructions_18k_alpaca/train-00000-of-00001.parquet",
            columns=("instruction", "input", "output"),
            allowed_fields=("instruction", "input"),
            forbidden_fields=("output", "prompt"),
            renderer_version="code-instruction-input-v1",
            render=render_code,
            stable_id=lambda row: prompt_sha256(
                f"{normalize_text(row['instruction'])}\0{normalize_text(row.get('input', ''))}"
            ),
        ),
        DomainSpec(
            name="gsm8k",
            path=data_root / "gsm8k/main/train-00000-of-00001.parquet",
            columns=("question", "answer"),
            allowed_fields=("question",),
            forbidden_fields=("answer",),
            renderer_version="gsm8k-question-v1",
            render=render_gsm8k,
            stable_id=lambda row: prompt_sha256(normalize_text(row["question"])),
        ),
        DomainSpec(
            name="arc",
            path=data_root / "arc/ARC-Challenge/train-00000-of-00001.parquet",
            columns=("id", "question", "choices", "answerKey"),
            allowed_fields=("id", "question", "choices"),
            forbidden_fields=("answerKey",),
            renderer_version="arc-question-unlabeled-choices-v1",
            render=render_arc,
            stable_id=lambda row: normalize_text(row["id"]),
        ),
        DomainSpec(
            name="hellaswag",
            path=data_root / "hellaswag/data/train-00000-of-00001.parquet",
            columns=("ind", "ctx_a", "ctx_b", "endings", "source_id", "label"),
            allowed_fields=("ind", "ctx_a", "ctx_b", "endings", "source_id"),
            forbidden_fields=("label",),
            renderer_version="hellaswag-context-unlabeled-endings-v1",
            render=render_hellaswag,
            stable_id=lambda row: f"{normalize_text(row['source_id'])}:{int(row['ind'])}",
            group_id=lambda row: normalize_text(row["source_id"]),
        ),
        DomainSpec(
            name="mmlu_pro",
            path=data_root / "mmlu_pro/data/validation-00000-of-00001.parquet",
            columns=("question_id", "question", "options", "answer", "answer_index", "cot_content", "category"),
            allowed_fields=("question_id", "question", "options", "category"),
            forbidden_fields=("answer", "answer_index", "cot_content"),
            renderer_version="mmlu-pro-question-unlabeled-options-v1",
            render=render_mmlu_pro,
            stable_id=lambda row: str(int(row["question_id"])),
            stratum=lambda row: normalize_text(row["category"]),
        ),
    ]


def rows_from_parquet(spec: DomainSpec) -> tuple[list[dict[str, Any]], int]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("pyarrow is required to build label-free calibration prompts") from error

    path = spec.path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Calibration source does not exist: {path}")
    table = parquet.read_table(path, columns=list(spec.columns))
    return table.to_pylist(), int(table.num_rows)


def candidates_from_rows(
    spec: DomainSpec,
    rows: list[dict[str, Any]],
    protocol_name: str,
) -> list[Candidate]:
    candidates = []
    for row in rows:
        source_id = spec.stable_id(row)
        if not source_id:
            raise ValueError(f"{spec.name} calibration row has an empty stable id")
        prompt = spec.render(row)
        candidates.append(
            Candidate(
                domain=spec.name,
                source_id=source_id,
                prompt=prompt,
                order_key=selection_key(protocol_name, spec.name, source_id),
                group_id=spec.group_id(row) if spec.group_id is not None else None,
                stratum=spec.stratum(row) if spec.stratum is not None else None,
            )
        )
    return candidates


def deduplicate_candidates(candidates: list[Candidate]) -> tuple[list[Candidate], int]:
    unique = []
    seen_prompts = set()
    duplicate_count = 0
    for candidate in sorted(candidates, key=lambda item: (item.order_key, item.domain, item.source_id)):
        canonical_prompt = normalize_text(candidate.prompt)
        if canonical_prompt in seen_prompts:
            duplicate_count += 1
            continue
        seen_prompts.add(canonical_prompt)
        unique.append(candidate)
    return unique, duplicate_count


def enforce_unique_groups(candidates: list[Candidate]) -> list[Candidate]:
    selected = []
    seen_groups = set()
    for candidate in sorted(candidates, key=lambda item: (item.order_key, item.source_id)):
        if candidate.group_id is not None and candidate.group_id in seen_groups:
            continue
        if candidate.group_id is not None:
            seen_groups.add(candidate.group_id)
        selected.append(candidate)
    return selected


def stratified_order(candidates: list[Candidate]) -> list[Candidate]:
    if not candidates or all(candidate.stratum is None for candidate in candidates):
        return sorted(candidates, key=lambda item: (item.order_key, item.source_id))
    groups: dict[str, deque[Candidate]] = defaultdict(deque)
    for candidate in sorted(candidates, key=lambda item: (item.order_key, item.source_id)):
        groups[str(candidate.stratum)].append(candidate)
    ordered = []
    while groups:
        for stratum in sorted(tuple(groups)):
            ordered.append(groups[stratum].popleft())
            if not groups[stratum]:
                del groups[stratum]
    return ordered


def select_splits(
    candidates: list[Candidate],
    quotas: dict[str, dict[str, int]],
) -> dict[str, list[Candidate]]:
    by_domain: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_domain[candidate.domain].append(candidate)
    splits: dict[str, list[Candidate]] = {"fit": [], "holdout": []}
    for domain in DOMAIN_ORDER:
        domain_candidates = stratified_order(enforce_unique_groups(by_domain[domain]))
        fit_count = int(quotas[domain]["fit"])
        holdout_count = int(quotas[domain]["holdout"])
        required = fit_count + holdout_count
        if len(domain_candidates) < required:
            raise ValueError(f"Domain {domain} has {len(domain_candidates)} usable rows, requires {required}")
        splits["fit"].extend(domain_candidates[:fit_count])
        splits["holdout"].extend(domain_candidates[fit_count:required])
    return splits


def round_robin(candidates: list[Candidate]) -> list[Candidate]:
    by_domain = {
        domain: deque(candidate for candidate in candidates if candidate.domain == domain)
        for domain in DOMAIN_ORDER
    }
    ordered = []
    while any(by_domain.values()):
        for domain in DOMAIN_ORDER:
            if by_domain[domain]:
                ordered.append(by_domain[domain].popleft())
    return ordered


def split_records(split: str, candidates: list[Candidate]) -> list[dict[str, Any]]:
    return [
        {
            "sequence_id": f"{split}-{index:03d}",
            "split": split,
            "domain": candidate.domain,
            "source_id": candidate.source_id,
            "text": candidate.prompt,
        }
        for index, candidate in enumerate(round_robin(candidates))
    ]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    content = "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records)
    path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic label-free CHANNEL calibration prompts.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/data01/datasets/evalscope_benchmarks"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol-name", default=PROTOCOL_NAME)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = build_domain_specs(data_root)
    all_candidates = []
    source_manifest = []
    for spec in specs:
        rows, row_count = rows_from_parquet(spec)
        all_candidates.extend(candidates_from_rows(spec, rows, args.protocol_name))
        source_path = spec.path.expanduser().resolve()
        source_manifest.append(
            {
                "domain": spec.name,
                "path": str(source_path),
                "rows": row_count,
                "size_bytes": int(source_path.stat().st_size),
                "sha256": file_sha256(source_path),
                "split": "validation" if spec.name == "mmlu_pro" else "train",
                "allowed_fields": list(spec.allowed_fields),
                "forbidden_fields": list(spec.forbidden_fields),
                "renderer_version": spec.renderer_version,
            }
        )

    unique_candidates, duplicate_count = deduplicate_candidates(all_candidates)
    splits = select_splits(unique_candidates, SPLIT_QUOTAS)
    fit_records = split_records("fit", splits["fit"])
    holdout_records = split_records("holdout", splits["holdout"])
    fit_ids = {(record["domain"], record["source_id"]) for record in fit_records}
    holdout_ids = {(record["domain"], record["source_id"]) for record in holdout_records}
    if fit_ids & holdout_ids:
        raise RuntimeError("Calibration fit and holdout source IDs overlap")
    fit_prompts = {normalize_text(record["text"]) for record in fit_records}
    holdout_prompts = {normalize_text(record["text"]) for record in holdout_records}
    if fit_prompts & holdout_prompts:
        raise RuntimeError("Calibration fit and holdout prompts overlap")

    fit_path = output_dir / "fit.jsonl"
    holdout_path = output_dir / "holdout.jsonl"
    write_jsonl(fit_path, fit_records)
    write_jsonl(holdout_path, holdout_records)
    manifest = {
        "schema_version": 1,
        "purpose": "label_free_channel_calibration",
        "protocol_name": args.protocol_name,
        "selection": "sha256_order_global_prompt_dedup_then_domain_split",
        "domain_order": list(DOMAIN_ORDER),
        "quotas": SPLIT_QUOTAS,
        "counts": {
            "fit": len(fit_records),
            "holdout": len(holdout_records),
            "total": len(fit_records) + len(holdout_records),
            "global_duplicate_prompts_removed": duplicate_count,
        },
        "fit_holdout_source_overlap": 0,
        "fit_holdout_prompt_overlap": 0,
        "sources": source_manifest,
        "outputs": {
            "fit": {"path": str(fit_path), "sha256": file_sha256(fit_path)},
            "holdout": {"path": str(holdout_path), "sha256": file_sha256(holdout_path)},
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["counts"], indent=2), flush=True)
    print(output_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())