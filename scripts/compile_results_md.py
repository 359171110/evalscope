#!/usr/bin/env python3
"""Compile per-model full8 comparison tables into Results.md.

Edit SOURCES below to add or remove experiments. Each model has:
  - dense: one directory or a list of directories (scores are merged; later paths win)
  - runs:  (sparsity_percent, method_name, path_or_paths)

Then regenerate:

    python scripts/compile_results_md.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from html import escape
from pathlib import Path
from typing import Any, Sequence, Union

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / 'Results.md'

PathLike = Union[str, Path]
PathSpec = Union[PathLike, Sequence[PathLike]]

DATASETS: tuple[tuple[str, str], ...] = (
    ('arc', 'ARC'),
    ('hellaswag', 'HellaSwag'),
    ('winogrande', 'WinoGrande'),
    ('gsm8k', 'GSM8K'),
    ('math_500', 'MATH-500'),
    ('mmlu', 'MMLU'),
    ('humaneval', 'HumanEval'),
    ('mbpp', 'MBPP'),
)

SPARSITY_ORDER: tuple[int, ...] = (25, 50)

# ---------------------------------------------------------------------------
# Result paths. Add or delete entries here to change the comparison.
# ---------------------------------------------------------------------------

SOURCES: list[dict[str, Any]] = [
    {
        'title': 'Qwen3-30B-A3B-Instruct-2507',
        'dense': [
            '/data/xinpeigao/evalscope_results/Qwen330BA3BInstruct_0_vllm_CalibrationFree_full6_v1_Dense_202608182316_42',
            '/data/xinpeigao/evalscope_results/humaneval/Qwen330BA3BInstruct_0_vllm_CalibrationFree_humaneval_Dense_202608190038_42',
            '/data/xinpeigao/evalscope_results/mbpp/Qwen330BA3BInstruct_0_vllm_CalibrationFree_mbpp_Dense_202608190059_42',
        ],
        'runs': [
            (25, 'Random',
             '/home/xinpeigao/evalscope/results/Qwen330BA3BInstruct_25_vllm_CalibrationFree_full8_v1_Random_202608191559_42'),
            (25, 'Magnitude',
             '/home/xinpeigao/evalscope/results/Qwen330BA3BInstruct_25_vllm_CalibrationFree_full8_v1_Magnitude_202608200100_42'),
            (25, 'AIMERChannel',
             '/home/xinpeigao/evalscope/results/Qwen330BA3BInstruct_25_vllm_CalibrationFree_full8_v1_AIMERChannel_202608201113_42'),
            (25, 'Wanda',
             '/home/xinpeigao/evalscope/results/Qwen330BA3BInstruct_25_vllm_WikiText128x2048_full8_v1_Wanda_202608201410_42'),
            (25, 'ENP',
                '/home/xinpeigao/evalscope/results/Qwen330BA3BInstruct_25_vllm_WikiText128x2048_full8_v1_ENP_202608201808_42'),
            (25, 'AIMERMix',
                '/home/xinpeigao/evalscope/results/Qwen330BA3BInstruct_25_vllm_CalibrationFree_full8_v1_AIMERMix_202608202313_42'),
            (50, 'Random',
             '/home/xinpeigao/evalscope/results/Qwen330BA3BInstruct_50_vllm_CalibrationFree_full8_v1_Random_202608191559_42'),
            (50, 'Magnitude',
             '/home/xinpeigao/evalscope/results/Qwen330BA3BInstruct_50_vllm_CalibrationFree_full8_v1_Magnitude_202608200100_42'),
            (50, 'AIMERChannel',
             '/home/xinpeigao/evalscope/results/Qwen330BA3BInstruct_50_vllm_CalibrationFree_full8_v1_AIMERChannel_202608201113_42'),
            (50, 'Wanda',
             '/home/xinpeigao/evalscope/results/Qwen330BA3BInstruct_50_vllm_WikiText128x2048_full8_v1_Wanda_202608201410_42'),
             (50, 'ENP',
                '/home/xinpeigao/evalscope/results/Qwen330BA3BInstruct_50_vllm_WikiText128x2048_full8_v1_ENP_202608201808_42'),
            (50, 'AIMERMix',
                '/home/xinpeigao/evalscope/results/Qwen330BA3BInstruct_50_vllm_CalibrationFree_full8_v1_AIMERMix_202608202313_42'),
        ],
    },
    {
        'title': 'Gemma4-26B-A4B-it',
        'dense': [
            '/data/xinpeigao/evalscope_results/Gemma4-26B-A4B-it_0_vllm_CalibrationFree_full6_v1_Dense_202608190417_42',
            '/data/xinpeigao/evalscope_results/humaneval/Gemma4-26B-A4B-it_0_vllm_CalibrationFree_humaneval_Dense_202608191343_42',
            '/data/xinpeigao/evalscope_results/mbpp/Gemma4-26B-A4B-it_0_vllm_CalibrationFree_mbpp_Dense_202608191348_42',
        ],
        'runs': [
            (25, 'Random',
             '/home/xinpeigao/evalscope/results/Gemma4-26B-A4B_25_vllm_CalibrationFree_full8_v1_Random_202608191559_42'),
            (25, 'Magnitude',
             '/home/xinpeigao/evalscope/results/Gemma4-26B-A4B_25_vllm_CalibrationFree_full8_v1_Magnitude_202608200100_42'),
            (25, 'AIMERChannel',
             '/home/xinpeigao/evalscope/results/Gemma4-26B-A4B_25_vllm_CalibrationFree_full8_v1_AIMERChannel_202608201113_42'),
            (25, 'Wanda',
             '/home/xinpeigao/evalscope/results/Gemma4-26B-A4B_25_vllm_WikiText128x2048_full8_v1_Wanda_202608201410_42'),
            (25, 'ENP',
                '/home/xinpeigao/evalscope/results/Gemma4-26B-A4B_25_vllm_WikiText128x2048_full8_v1_ENP_202608201808_42'),
            (25, 'AIMERMix',
                '/home/xinpeigao/evalscope/results/Gemma4-26B-A4B_25_vllm_CalibrationFree_full8_v1_AIMERMix_202608202313_42'),
            (25, 'Product',
                '/home/xinpeigao/evalscope/results/Gemma4-26B-A4B_25_vllm_CalibrationFree_full8_v1_Product_202608211630_42'),
            (50, 'Random',
             '/home/xinpeigao/evalscope/results/Gemma4-26B-A4B_50_vllm_CalibrationFree_full8_v1_Random_202608191559_42'),
            (50, 'Magnitude',
             '/home/xinpeigao/evalscope/results/Gemma4-26B-A4B_50_vllm_CalibrationFree_full8_v1_Magnitude_202608200100_42'),
            (50, 'AIMERChannel',
             '/home/xinpeigao/evalscope/results/Gemma4-26B-A4B_50_vllm_CalibrationFree_full8_v1_AIMERChannel_202608201113_42'),
            (50, 'Wanda',
             '/home/xinpeigao/evalscope/results/Gemma4-26B-A4B_50_vllm_WikiText128x2048_full8_v1_Wanda_202608201410_42'),
             (50, 'ENP',
                '/home/xinpeigao/evalscope/results/Gemma4-26B-A4B_50_vllm_WikiText128x2048_full8_v1_ENP_202608201808_42'),
            (50, 'AIMERMix',
                '/home/xinpeigao/evalscope/results/Gemma4-26B-A4B_50_vllm_CalibrationFree_full8_v1_AIMERMix_202608202313_42'),
        ],
    },
    {
        'title': 'Qwen3.6-35B-A3B',
        'dense': [
            '/data/xinpeigao/evalscope_results/Qwen36-35B-A3B_0_vllm_CalibrationFree_full6_v1_Dense_202608182346_42',
            '/data/xinpeigao/evalscope_results/humaneval/Qwen36-35B-A3B_0_vllm_CalibrationFree_humaneval_Dense_202608190038_42',
            '/data/xinpeigao/evalscope_results/mbpp/Qwen36-35B-A3B_0_vllm_CalibrationFree_mbpp_Dense_202608190059_42',
        ],
        'runs': [
            (25, 'Random',
             '/home/xinpeigao/evalscope/results/Qwen3.6-35B-A3B_25_vllm_CalibrationFree_full8_v1_Random_202608191559_42'),
            (25, 'Magnitude',
             '/home/xinpeigao/evalscope/results/Qwen3.6-35B-A3B_25_vllm_CalibrationFree_full8_v1_Magnitude_202608200100_42'),
            (25, 'AIMERChannel',
             '/home/xinpeigao/evalscope/results/Qwen3.6-35B-A3B_25_vllm_CalibrationFree_full8_v1_AIMERChannel_202608201113_42'),
            (25, 'Wanda',
             '/home/xinpeigao/evalscope/results/Qwen3.6-35B-A3B_25_vllm_WikiText128x2048_full8_v1_Wanda_202608201410_42'),
            (25, 'ENP',
                '/home/xinpeigao/evalscope/results/Qwen3.6-35B-A3B_25_vllm_WikiText128x2048_full8_v1_ENP_202608201808_42'),
            (25, 'AIMERMix',
                '/home/xinpeigao/evalscope/results/Qwen3.6-35B-A3B_25_vllm_CalibrationFree_full8_v1_AIMERMix_202608202313_42'),
            (50, 'Random',
             '/home/xinpeigao/evalscope/results/Qwen3.6-35B-A3B_50_vllm_CalibrationFree_full8_v1_Random_202608191559_42'),
            (50, 'Magnitude',
             '/home/xinpeigao/evalscope/results/Qwen3.6-35B-A3B_50_vllm_CalibrationFree_full8_v1_Magnitude_202608200100_42'),
            (50, 'AIMERChannel',
             '/home/xinpeigao/evalscope/results/Qwen3.6-35B-A3B_50_vllm_CalibrationFree_full8_v1_AIMERChannel_202608201113_42'),
            (50, 'Wanda',
             '/home/xinpeigao/evalscope/results/Qwen3.6-35B-A3B_50_vllm_WikiText128x2048_full8_v1_Wanda_202608201410_42'),
             (50, 'ENP',
                '/home/xinpeigao/evalscope/results/Qwen3.6-35B-A3B_50_vllm_WikiText128x2048_full8_v1_ENP_202608201808_42'),
            (50, 'AIMERMix',
                '/home/xinpeigao/evalscope/results/Qwen3.6-35B-A3B_50_vllm_CalibrationFree_full8_v1_AIMERMix_202608202313_42'),
        ],
    },
    {
        'title': 'DeepSeek-V2-Lite-Chat',
        'dense': '/data/xinpeigao/evalscope_results/DeepSeek-V2-Lite-Chat_0_vllm_CalibrationFree_full8_v1_Dense_202608200101_42',
        'runs': [
            (25, 'Random',
             '/home/xinpeigao/evalscope/results/DeepSeek-V2-Lite-Chat_25_vllm_CalibrationFree_full8_v1_Random_202608191826_42'),
            (25, 'Magnitude',
             '/home/xinpeigao/evalscope/results/DeepSeek-V2-Lite-Chat_25_vllm_CalibrationFree_full8_v1_Magnitude_202608200100_42'),
            (25, 'AIMERChannel',
             '/home/xinpeigao/evalscope/results/DeepSeek-V2-Lite-Chat_25_vllm_CalibrationFree_full8_v1_AIMERChannel_202608201113_42'),
            (25, 'Wanda',
             '/home/xinpeigao/evalscope/results/DeepSeek-V2-Lite-Chat_25_vllm_WikiText128x2048_full8_v1_Wanda_202608201410_42'),
            (25, 'ENP',
                '/home/xinpeigao/evalscope/results/DeepSeek-V2-Lite-Chat_25_vllm_WikiText128x2048_full8_v1_ENP_202608201808_42'),
            (25, 'AIMERMix',
                '/home/xinpeigao/evalscope/results/DeepSeek-V2-Lite-Chat_25_vllm_CalibrationFree_full8_v1_AIMERMix_202608202313_42'),
            (50, 'Random',
             '/home/xinpeigao/evalscope/results/DeepSeek-V2-Lite-Chat_50_vllm_CalibrationFree_full8_v1_Random_202608191826_42'),
            (50, 'Magnitude',
             '/home/xinpeigao/evalscope/results/DeepSeek-V2-Lite-Chat_50_vllm_CalibrationFree_full8_v1_Magnitude_202608200100_42'),
            (50, 'AIMERChannel',
             '/home/xinpeigao/evalscope/results/DeepSeek-V2-Lite-Chat_50_vllm_CalibrationFree_full8_v1_AIMERChannel_202608201113_42'),
            (50, 'Wanda',
             '/home/xinpeigao/evalscope/results/DeepSeek-V2-Lite-Chat_50_vllm_WikiText128x2048_full8_v1_Wanda_202608201410_42'),
             (50, 'ENP',
                '/home/xinpeigao/evalscope/results/DeepSeek-V2-Lite-Chat_50_vllm_WikiText128x2048_full8_v1_ENP_202608201808_42'),
            (50, 'AIMERMix',
                '/home/xinpeigao/evalscope/results/DeepSeek-V2-Lite-Chat_50_vllm_CalibrationFree_full8_v1_AIMERMix_202608202313_42'),
        ],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compile per-model pruning comparison tables into Results.md.')
    parser.add_argument(
        '--output',
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f'Markdown output path. Default: {DEFAULT_OUTPUT}',
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=None,
        help='Optional JSON file that replaces the in-script SOURCES list.',
    )
    return parser.parse_args()


def as_paths(spec: PathSpec) -> list[Path]:
    if isinstance(spec, (str, Path)):
        return [Path(spec)]
    return [Path(item) for item in spec]


def find_report(experiment: Path, dataset: str) -> Path | None:
    candidates = sorted(
        path for path in experiment.rglob(f'{dataset}.json')
        if 'reports' in path.parts and dataset in path.parts and '_broken' not in path.parts
    )
    return candidates[0] if candidates else None


def read_score(path: Path) -> float | None:
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None
    score = payload.get('score')
    return float(score) if isinstance(score, (int, float)) else None


def load_scores(spec: PathSpec, warnings: list[str]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for experiment in as_paths(spec):
        if not experiment.is_dir():
            warnings.append(f'missing experiment directory: {experiment}')
            continue
        for dataset, _label in DATASETS:
            report = find_report(experiment, dataset)
            if report is None:
                continue
            score = read_score(report)
            if score is None:
                warnings.append(f'missing numeric score: {report}')
                continue
            scores[dataset] = score
    return scores


def format_score(value: float | None) -> str:
    if value is None:
        return '-'
    return f'{value:.4f}'


def retained_ratio(method: float | None, dense: float | None) -> float | None:
    if method is None or dense is None or dense == 0:
        return None
    return 100.0 * method / dense


def format_retained(method: float | None, dense: float | None) -> str:
    ratio = retained_ratio(method, dense)
    return '-' if ratio is None else f'{ratio:.1f}%'


def mean_retained(method_scores: dict[str, float], dense_scores: dict[str, float]) -> str:
    ratios = [
        retained_ratio(method_scores.get(name), dense_scores.get(name)) for name, _label in DATASETS
    ]
    present = [ratio for ratio in ratios if ratio is not None]
    if not present:
        return '-'
    return f'{sum(present) / len(present):.1f}%'


def html_cell(text: str, *, header: bool = False, align: str | None = None, colspan: int | None = None) -> str:
    tag = 'th' if header else 'td'
    attrs: list[str] = []
    if align:
        attrs.append(f'align="{align}"')
    if colspan is not None:
        attrs.append(f'colspan="{colspan}"')
    attr = f' {" ".join(attrs)}' if attrs else ''
    return f'<{tag}{attr}>{escape(text)}</{tag}>'


def html_row(cells: Sequence[str], *, header: bool = False, numeric: bool = False) -> str:
    rendered: list[str] = []
    for index, text in enumerate(cells):
        align = 'right' if numeric and index > 0 else None
        rendered.append(html_cell(text, header=header, align=align))
    return '<tr>' + ''.join(rendered) + '</tr>'


def html_span_row(label: str, n_cols: int) -> str:
    return '<tr>' + html_cell(label, header=True, align='center', colspan=n_cols) + '</tr>'


def group_runs(runs: Sequence[tuple[Any, ...]]) -> OrderedDict[int, list[tuple[str, PathSpec]]]:
    grouped: OrderedDict[int, list[tuple[str, PathSpec]]] = OrderedDict()
    for sparsity, method, spec in runs:
        grouped.setdefault(int(sparsity), []).append((str(method), spec))
    ordered: OrderedDict[int, list[tuple[str, PathSpec]]] = OrderedDict()
    for sparsity in SPARSITY_ORDER:
        if sparsity in grouped:
            ordered[sparsity] = grouped[sparsity]
    for sparsity, items in grouped.items():
        if sparsity not in ordered:
            ordered[sparsity] = items
    return ordered


def render_model(model: dict[str, Any], warnings: list[str]) -> str:
    title = str(model['title'])
    dense_scores = load_scores(model['dense'], warnings)
    grouped = group_runs(model.get('runs') or [])
    labels = [label for _name, label in DATASETS]
    header = ['Method', *labels, 'Mean retained %']
    n_cols = len(header)
    rows = [
        html_row(header, header=True, numeric=True),
        html_row(['Dense', *[format_score(dense_scores.get(name)) for name, _label in DATASETS], '-'], numeric=True),
    ]
    missing_dense = [label for name, label in DATASETS if name not in dense_scores]
    if missing_dense:
        warnings.append(f'{title}: Dense missing {", ".join(missing_dense)}')

    for sparsity, methods in grouped.items():
        rows.append(html_span_row(f'{sparsity}% pruning', n_cols))
        for method, spec in methods:
            scores = load_scores(spec, warnings)
            missing = [label for name, label in DATASETS if name not in scores]
            if missing:
                warnings.append(f'{title} {sparsity}% {method}: missing {", ".join(missing)}')
            rows.append(
                html_row(
                    [method, *[format_score(scores.get(name)) for name, _label in DATASETS], '-'],
                    numeric=True,
                )
            )
            rows.append(
                html_row(
                    [
                        f'{method} retained %',
                        *[format_retained(scores.get(name), dense_scores.get(name)) for name, _label in DATASETS],
                        mean_retained(scores, dense_scores),
                    ],
                    numeric=True,
                )
            )
    table = '<table>\n' + '\n'.join(rows) + '\n</table>\n'
    return f'## {title}\n\n{table}'


def load_sources(config_path: Path | None) -> list[dict[str, Any]]:
    if config_path is None:
        return SOURCES
    payload = json.loads(config_path.read_text(encoding='utf-8'))
    if not isinstance(payload, list):
        raise ValueError(f'config must be a JSON list of model objects: {config_path}')
    return payload


def render_markdown(sources: list[dict[str, Any]], warnings: list[str]) -> str:
    parts = [
        '<!-- Generated by scripts/compile_results_md.py. Edit SOURCES in that script (or pass --config), then rerun. -->',
        '',
        '# Downstream pruning results',
        '',
        'Each table is one model. The first data row is the unpruned Dense baseline. '
        'A full-width row marks each sparsity. Each method then occupies two rows: raw accuracy, '
        'then the fraction of Dense kept on that dataset. The rightmost column averages those '
        'per-dataset retained percentages into an overall retention rate.',
        '',
        'Regenerate:',
        '',
        '```bash',
        'python scripts/compile_results_md.py',
        '```',
        '',
    ]
    for model in sources:
        parts.append(render_model(model, warnings))
    return '\n'.join(parts).rstrip() + '\n'


def main() -> int:
    args = parse_args()
    warnings: list[str] = []
    sources = load_sources(args.config)
    markdown = render_markdown(sources, warnings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding='utf-8')
    print(f'Wrote {args.output}')
    for item in warnings:
        print(f'WARNING: {item}', file=sys.stderr)
    return 1 if warnings else 0


if __name__ == '__main__':
    raise SystemExit(main())
