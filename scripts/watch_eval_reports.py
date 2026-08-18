from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any


PROTOCOL_DIR = Path(__file__).resolve().parents[1] / 'eval_protocol'
DEFAULT_PROTOCOL = 'quick9'
DATASET_LABELS = {
    'arc': 'ARC',
    'hellaswag': 'HellaSwag',
    'winogrande': 'WinoGrande',
    'gsm8k': 'GSM8K',
    'math_500': 'MATH-500',
    'mmlu': 'MMLU',
}


def load_protocol_spec(name: str) -> dict[str, Any]:
    payload = json.loads((PROTOCOL_DIR / f'{name}.json').read_text(encoding='utf-8'))
    return payload


def dataset_specs(protocol_name: str) -> list[tuple[str, str, int]]:
    protocol = load_protocol_spec(protocol_name)
    return [
        (item['name'], str(item.get('label') or DATASET_LABELS[item['name']]), int(item['expected_samples']))
        for item in protocol['datasets']
    ]


EXPERIMENT_NAME_RE = re.compile(
    r'^(?P<model>[A-Za-z0-9.-]+)_'
    r'(?P<pruning>[A-Za-z0-9.-]+)_'
    r'(?P<inference>vllm|transformer)_'
    r'(?P<calibration>[A-Za-z0-9.-]+)_'
    r'(?P<protocol>quick9|full6_v1|full6_unlimited)_'
    r'(?P<method>[A-Za-z0-9-]+)_'
    r'(?P<timestamp>\d{12})_'
    r'(?P<seed>\d+)$'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compare frozen EvalScope protocol reports.')
    parser.add_argument('experiments', nargs='*', type=Path)
    parser.add_argument('--result-root', type=Path, default=None)
    parser.add_argument('--watch', type=float, default=0.0)
    parser.add_argument('--details', action='store_true')
    parser.add_argument('--allow-invalid', action='store_true')
    return parser.parse_args()


def find_report(experiment: Path, dataset: str) -> Path | None:
    candidates = sorted(
        path for path in experiment.rglob(f'{dataset}.json') if 'reports' in path.parts and dataset in path.parts
    )
    return candidates[0] if candidates else None


def read_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload.get('score'), (int, float)):
        raise ValueError(f'missing numeric top-level score: {path}')
    return payload


def sample_count(payload: dict[str, Any]) -> int | None:
    metrics = payload.get('metrics') or []
    if not metrics:
        return None
    num = metrics[0].get('num')
    return int(num) if isinstance(num, (int, float)) else None


def discover_experiments(result_root: Path) -> list[Path]:
    return sorted(path for path in result_root.iterdir() if path.is_dir() and not path.name.startswith('.'))


def load_manifest(experiment: Path) -> dict[str, Any] | None:
    manifest_path = experiment / 'experiment_manifest.json'
    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def validate_experiment(experiment: Path) -> list[str]:
    errors: list[str] = []
    match = EXPERIMENT_NAME_RE.match(experiment.name)
    if match is None:
        errors.append('directory name does not match the frozen experiment identity format')
        return errors

    manifest = load_manifest(experiment)
    if manifest is None:
        errors.append('missing or invalid experiment_manifest.json')
    else:
        expected = {
            'target_model': match.group('model'),
            'pruning_ratio_label': match.group('pruning'),
            'inference': match.group('inference'),
            'calibration': match.group('calibration'),
            'evaluation_protocol': match.group('protocol'),
            'method': match.group('method'),
            'started_at_minute': match.group('timestamp'),
        }
        for key, value in expected.items():
            if str(manifest.get(key, '')) != value:
                errors.append(f'manifest field {key} != {value}')

    protocol_name = match.group('protocol')
    for dataset, _label, expected_num in dataset_specs(protocol_name):
        report_path = find_report(experiment, dataset)
        if report_path is None:
            errors.append(f'missing {dataset} report')
            continue
        try:
            payload = read_report(report_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            errors.append(f'invalid {dataset} report: {error}')
            continue
        num = sample_count(payload)
        if num != expected_num:
            errors.append(f'{dataset} sample count {num} != {expected_num}')
    return errors


def row_label(experiment: Path) -> str:
    match = EXPERIMENT_NAME_RE.match(experiment.name)
    if match is None:
        return experiment.name
    return f'{match.group("pruning")}% {match.group("calibration")} {match.group("method")}'


def collect_scores(experiment: Path, protocol_name: str) -> dict[str, str]:
    values: dict[str, str] = {}
    scores: list[float] = []
    for dataset, _label, _expected in dataset_specs(protocol_name):
        report_path = find_report(experiment, dataset)
        if report_path is None:
            values[dataset] = '-'
            continue
        try:
            payload = read_report(report_path)
            score = float(payload['score'])
        except (OSError, ValueError, json.JSONDecodeError):
            values[dataset] = 'ERROR'
            continue
        values[dataset] = f'{score:.4f}'
        scores.append(score)
    values['macro'] = f'{sum(scores) / len(scores):.4f}' if scores else '-'
    return values


def render(experiments: list[Path], allow_invalid: bool, show_details: bool) -> str:
    lines = ['Protocol comparison (macro is unweighted mean of six dataset scores).', '']
    header_specs = dataset_specs(DEFAULT_PROTOCOL)
    header = f'{"experiment":<48} ' + ' '.join(f'{label:<12}' for _name, label, _num in header_specs) + f' {"Macro":<12}'
    lines.append(header)
    lines.append('-' * len(header))

    rejected: list[str] = []
    for experiment in experiments:
        match = EXPERIMENT_NAME_RE.match(experiment.name)
        protocol_name = match.group('protocol') if match else DEFAULT_PROTOCOL
        errors = validate_experiment(experiment)
        if errors and not allow_invalid:
            rejected.append(f'{experiment.name}: ' + '; '.join(errors))
            continue
        scores = collect_scores(experiment, protocol_name)
        row = f'{row_label(experiment)[:48]:<48} '
        row += ' '.join(f'{scores[name]:<12}' for name, _label, _num in header_specs)
        row += f' {scores["macro"]:<12}'
        if errors:
            row += ' INVALID'
        lines.append(row)
        if show_details:
            for error in errors:
                lines.append(f'  - {error}')

    if rejected:
        lines.append('')
        lines.append('Rejected experiments:')
        lines.extend(f'- {item}' for item in rejected)
    if not experiments:
        lines.append('No experiment directories found.')
    return '\n'.join(lines)


def main() -> None:
    args = parse_args()
    result_root = (args.result_root or Path(os.environ.get('RESULT_ROOT', 'result'))).expanduser().resolve()
    if args.experiments:
        experiments = [path.expanduser().resolve() for path in args.experiments]
        missing = [str(path) for path in experiments if not path.is_dir()]
        if missing:
            raise SystemExit('experiment directory does not exist: ' + ', '.join(missing))
    else:
        if not result_root.is_dir():
            raise SystemExit(f'result root does not exist: {result_root}')
        experiments = discover_experiments(result_root)

    while True:
        if args.watch > 0:
            print('\033[2J\033[H', end='')
        print(render(experiments, args.allow_invalid, args.details), flush=True)
        if args.watch <= 0:
            return
        try:
            time.sleep(args.watch)
        except KeyboardInterrupt:
            return


if __name__ == '__main__':
    main()
