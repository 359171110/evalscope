#!/usr/bin/env python3
"""List or delete exported checkpoints whose downstream eval has finished.

Rankings/profiles are never touched. Default is list-only.

    python scripts/list_evaluated_checkpoints.py
    python scripts/list_evaluated_checkpoints.py --paths-only
    python scripts/list_evaluated_checkpoints.py --delete
    python scripts/list_evaluated_checkpoints.py --delete --yes
    python scripts/list_evaluated_checkpoints.py --delete --rm-dir --only random
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_DIR = ROOT / 'eval_protocol'
DEFAULT_ARTIFACT_ROOTS = (
    Path('/data/xinpeigao/evalscope_results/_artifacts'),
)
DEFAULT_RESULT_ROOTS = (
    Path('/home/xinpeigao/evalscope/results'),
    Path('/data/xinpeigao/evalscope_results'),
)
WEIGHT_GLOBS = ('model-*.safetensors', 'model.safetensors')
SKIP_DIR_NAMES = {'_broken', '_logs', '_launchers', '_calibration'}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='List exported checkpoints with a complete eval. Optionally delete their weights.'
    )
    parser.add_argument(
        '--artifact-root',
        action='append',
        type=Path,
        default=None,
        help='Root that contains exported checkpoint_* dirs. Repeatable. Default: _artifacts.',
    )
    parser.add_argument(
        '--result-root',
        action='append',
        type=Path,
        default=None,
        help='Root that contains experiment dirs. Repeatable.',
    )
    parser.add_argument(
        '--paths-only',
        action='store_true',
        help='Print only checkpoint paths that are complete, have weights, and are not in use.',
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Also print incomplete, already-empty, unused, and in-use checkpoints.',
    )
    parser.add_argument(
        '--only',
        action='append',
        default=[],
        help='Substring filter on the checkpoint path. Repeatable (OR). Example: --only random.',
    )
    parser.add_argument(
        '--delete',
        action='store_true',
        help='Delete weight shards of listed complete checkpoints after confirmation.',
    )
    parser.add_argument(
        '--rm-dir',
        action='store_true',
        help='With --delete, remove the whole checkpoint_* directory instead of only *.safetensors.',
    )
    parser.add_argument(
        '--yes',
        action='store_true',
        help='Skip the interactive confirmation. Required for non-interactive --delete.',
    )
    return parser.parse_args()


def protocol_datasets(name: str) -> tuple[str, ...]:
    path = PROTOCOL_DIR / f'{name}.json'
    if not path.is_file():
        return ('arc', 'hellaswag', 'winogrande', 'gsm8k', 'math_500', 'mmlu', 'humaneval', 'mbpp')
    payload = json.loads(path.read_text(encoding='utf-8'))
    return tuple(str(item['name']) for item in payload['datasets'])


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


def weight_files(checkpoint: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in WEIGHT_GLOBS:
        files.extend(sorted(checkpoint.glob(pattern)))
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in files:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def dir_bytes(paths: Iterable[Path]) -> int:
    total = 0
    for path in paths:
        try:
            total += path.stat().st_size
        except OSError:
            continue
    return total


def format_gib(n_bytes: int) -> str:
    return f'{n_bytes / (1024 ** 3):.1f} GiB'


def is_exported_checkpoint(path: Path, artifact_roots: list[Path]) -> bool:
    resolved = path.resolve()
    for root in artifact_roots:
        if not root.exists():
            continue
        try:
            resolved.relative_to(root.resolve())
            return True
        except ValueError:
            continue
    if resolved.name.startswith('checkpoint_'):
        return True
    return (resolved / 'pruning_export_manifest.json').is_file()


def is_checkpoint_dir(path: Path) -> bool:
    if not path.is_dir() or not (path / 'config.json').is_file():
        return False
    if weight_files(path):
        return True
    return (path / 'model.safetensors.index.json').is_file() or (path / 'pruning_export_manifest.json').is_file()


def discover_checkpoints(roots: list[Path]) -> list[Path]:
    found: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for config in root.rglob('config.json'):
            if any(part in SKIP_DIR_NAMES or part.startswith('.') for part in config.parts):
                continue
            checkpoint = config.parent
            if is_checkpoint_dir(checkpoint):
                found.add(checkpoint.resolve())
    return sorted(found)


def load_manifest(experiment: Path) -> dict[str, Any] | None:
    path = experiment / 'experiment_manifest.json'
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def experiment_checkpoint_paths(experiment: Path, manifest: dict[str, Any] | None) -> list[Path]:
    paths: list[Path] = []
    if manifest:
        raw = manifest.get('checkpoint')
        if raw:
            paths.append(Path(str(raw)))
    checkpoint_root = experiment / 'checkpoints'
    if checkpoint_root.is_dir():
        for child in checkpoint_root.iterdir():
            if child.is_symlink() or child.is_dir():
                paths.append(child)
    resolved: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        try:
            real = path.resolve()
        except OSError:
            continue
        if real in seen or not real.is_dir():
            continue
        seen.add(real)
        resolved.append(real)
    return resolved


def discover_experiments(roots: list[Path]) -> list[Path]:
    experiments: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for manifest in root.glob('*/experiment_manifest.json'):
            if manifest.parent.name.startswith('_'):
                continue
            experiments.append(manifest.parent)
        for manifest in root.glob('*/*/experiment_manifest.json'):
            if any(part.startswith('_') for part in manifest.relative_to(root).parts):
                continue
            experiments.append(manifest.parent)
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in experiments:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


@dataclass
class ExperimentEval:
    path: Path
    protocol: str
    datasets: tuple[str, ...]
    present: tuple[str, ...]
    missing: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.missing


def evaluate_experiment(experiment: Path, manifest: dict[str, Any] | None) -> ExperimentEval:
    protocol = 'full8_v1'
    if manifest:
        protocol = str(manifest.get('evaluation_protocol') or protocol)
    datasets = protocol_datasets(protocol)
    present_list: list[str] = []
    for name in datasets:
        report = find_report(experiment, name)
        if report is not None and read_score(report) is not None:
            present_list.append(name)
    present = tuple(present_list)
    missing = tuple(name for name in datasets if name not in present)
    return ExperimentEval(
        path=experiment,
        protocol=protocol,
        datasets=datasets,
        present=present,
        missing=missing,
    )


def running_model_paths() -> set[Path]:
    found: set[Path] = set()
    pattern = re.compile(r'--model\s+(\S+)')
    try:
        proc = Path('/proc')
        for pid_dir in proc.iterdir():
            if not pid_dir.name.isdigit():
                continue
            try:
                cmdline = (pid_dir / 'cmdline').read_bytes().replace(b'\x00', b' ').decode('utf-8', 'replace')
            except OSError:
                continue
            if 'vllm' not in cmdline and 'evalscope' not in cmdline:
                continue
            for match in pattern.finditer(cmdline):
                path = Path(match.group(1))
                try:
                    found.add(path.resolve())
                except OSError:
                    found.add(path)
    except OSError:
        return found
    return found


@dataclass
class CheckpointReport:
    path: Path
    weight_bytes: int
    in_use: bool
    experiments: list[ExperimentEval] = field(default_factory=list)

    @property
    def has_weights(self) -> bool:
        return self.weight_bytes > 0

    @property
    def complete(self) -> bool:
        return any(item.complete for item in self.experiments)

    @property
    def deletable(self) -> bool:
        return self.complete and self.has_weights and not self.in_use


def build_reports(
    checkpoints: list[Path],
    experiments: list[Path],
    in_use: set[Path],
    artifact_roots: list[Path],
) -> list[CheckpointReport]:
    by_checkpoint: dict[Path, CheckpointReport] = {}
    for checkpoint in checkpoints:
        by_checkpoint[checkpoint] = CheckpointReport(
            path=checkpoint,
            weight_bytes=dir_bytes(weight_files(checkpoint)),
            in_use=checkpoint in in_use,
        )

    for experiment in experiments:
        manifest = load_manifest(experiment)
        eval_status = evaluate_experiment(experiment, manifest)
        for checkpoint in experiment_checkpoint_paths(experiment, manifest):
            if not is_exported_checkpoint(checkpoint, artifact_roots):
                continue
            report = by_checkpoint.get(checkpoint)
            if report is None:
                report = CheckpointReport(
                    path=checkpoint,
                    weight_bytes=dir_bytes(weight_files(checkpoint)),
                    in_use=checkpoint in in_use,
                )
                by_checkpoint[checkpoint] = report
            report.experiments.append(eval_status)

    return sorted(by_checkpoint.values(), key=lambda item: str(item.path))


def matches_only(path: Path, filters: list[str]) -> bool:
    if not filters:
        return True
    text = str(path)
    return any(token in text for token in filters)


def assert_under_artifact_root(path: Path, artifact_roots: list[Path]) -> None:
    resolved = path.resolve()
    for root in artifact_roots:
        if not root.exists():
            continue
        try:
            resolved.relative_to(root.resolve())
            return
        except ValueError:
            continue
    raise PermissionError(f'Refusing to delete a path outside artifact roots: {resolved}')


def delete_checkpoint(item: CheckpointReport, *, rm_dir: bool, artifact_roots: list[Path]) -> int:
    assert_under_artifact_root(item.path, artifact_roots)
    if rm_dir:
        shutil.rmtree(item.path)
        return item.weight_bytes
    removed = 0
    for shard in weight_files(item.path):
        assert_under_artifact_root(shard, artifact_roots)
        removed += shard.stat().st_size
        shard.unlink()
    return removed


def confirm_delete(count: int, n_bytes: int, rm_dir: bool) -> bool:
    target = 'directories' if rm_dir else 'weight-shard sets'
    phrase = f'DELETE {count} {target.upper()}'
    print()
    print(f'This will remove {format_gib(n_bytes)} from {count} checkpoint {target}.')
    print('Rankings/profiles and eval reports are not deleted.')
    if not sys.stdin.isatty():
        print('Refusing --delete without --yes in a non-interactive shell.', file=sys.stderr)
        return False
    answer = input(f'Type exactly "{phrase}" to continue: ')
    return answer == phrase


def print_human(reports: list[CheckpointReport], show_all: bool, heading: str) -> None:
    deletable = [item for item in reports if item.deletable]
    others = [item for item in reports if not item.deletable]
    total = sum(item.weight_bytes for item in deletable)
    print(f'Complete evals with weights still on disk: {len(deletable)}  ({format_gib(total)})')
    print(heading)
    print()
    for item in deletable:
        experiments = ', '.join(exp.path.name for exp in item.experiments if exp.complete) or '-'
        print(f'{format_gib(item.weight_bytes):>8}  {item.path}')
        print(f'          eval: {experiments}')
    if not show_all:
        skipped = len(others)
        if skipped:
            print()
            print(f'Skipped {skipped} checkpoints (incomplete, empty, unused, or in use). Pass --all to list them.')
        return
    print()
    print('Other checkpoints:')
    for item in others:
        reasons: list[str] = []
        if item.in_use:
            reasons.append('IN_USE')
        if not item.experiments:
            reasons.append('NO_EVAL')
        elif not item.complete:
            missing = sorted({name for exp in item.experiments for name in exp.missing})
            reasons.append(f'INCOMPLETE:{",".join(missing) if missing else "?"}')
        if not item.has_weights:
            reasons.append('NO_WEIGHTS')
        print(f'{format_gib(item.weight_bytes):>8}  [{" ".join(reasons)}]  {item.path}')


def main() -> int:
    args = parse_args()
    if args.rm_dir and not args.delete:
        print('ERROR: --rm-dir requires --delete.', file=sys.stderr)
        return 2
    if args.paths_only and args.delete:
        print('ERROR: --paths-only cannot be combined with --delete.', file=sys.stderr)
        return 2

    artifact_roots = args.artifact_root or list(DEFAULT_ARTIFACT_ROOTS)
    result_roots = args.result_root or list(DEFAULT_RESULT_ROOTS)
    checkpoints = discover_checkpoints(artifact_roots)
    experiments = discover_experiments(result_roots)
    reports = build_reports(checkpoints, experiments, running_model_paths(), artifact_roots)
    if args.only:
        reports = [item for item in reports if matches_only(item.path, args.only)]
    deletable = [item for item in reports if item.deletable]

    if args.paths_only:
        for item in deletable:
            print(item.path)
        return 0

    heading = 'Nothing was deleted. Copy a path below, or rerun with --delete.'
    print_human(reports, show_all=args.all, heading=heading)
    if not args.delete:
        return 0
    if not deletable:
        print('No complete checkpoints with weights to delete.')
        return 0

    total = sum(item.weight_bytes for item in deletable)
    if not args.yes and not confirm_delete(len(deletable), total, args.rm_dir):
        print('Aborted. Nothing was deleted.')
        return 1

    removed = 0
    failed = 0
    for item in deletable:
        try:
            removed += delete_checkpoint(item, rm_dir=args.rm_dir, artifact_roots=artifact_roots)
            print(f'DELETED  {item.path}')
        except (OSError, PermissionError) as exc:
            failed += 1
            print(f'FAILED   {item.path}: {exc}', file=sys.stderr)
    print(f'Removed {format_gib(removed)} from {len(deletable) - failed} checkpoints.' + (
        f' {failed} failed.' if failed else ''
    ))
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
