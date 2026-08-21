#!/usr/bin/env python3
"""Run a frozen vLLM EvalScope protocol against a local OpenAI-compatible server."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


PROTOCOL_DIR = Path(__file__).resolve().parent
SUPPORTED_PROTOCOLS = ('quick9', 'full6_v1', 'full8_v1')
PATH_ENV_FALLBACK = {
    'ARC_PATH': 'arc',
    'HELLASWAG_PATH': 'hellaswag',
    'WINOGRANDE_PATH': 'winogrande/winogrande_1.1.zip',
    'GSM8K_PATH': 'gsm8k',
    'MATH_500_PATH': 'math_500',
    'MMLU_PATH': 'mmlu',
    'HUMANEVAL_PATH': 'humaneval',
    'MBPP_PATH': 'mbpp',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True, choices=SUPPORTED_PROTOCOLS)
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--api-url', required=True)
    parser.add_argument('--method', required=True)
    parser.add_argument('--experiment-dir', required=True, type=Path)
    parser.add_argument('--datasets', default='', help='Optional comma-separated subset of protocol datasets.')
    parser.add_argument('--python-bin', default=os.environ.get('PYTHON_BIN', sys.executable))
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def load_protocol(name: str) -> dict[str, Any]:
    payload = json.loads((PROTOCOL_DIR / f'{name}.json').read_text(encoding='utf-8'))
    if payload.get('name') != name:
        raise ValueError(f'Protocol file name mismatch: {name}')
    return payload


def selected_datasets(protocol: dict[str, Any], requested: str) -> list[dict[str, Any]]:
    datasets = {item['name']: item for item in protocol['datasets']}
    order = list(protocol['dataset_order'])
    if requested.strip():
        order = [name.strip() for name in requested.split(',') if name.strip()]
    missing = [name for name in order if name not in datasets]
    if missing:
        raise ValueError(f'Unknown datasets for this protocol: {missing}')
    return [datasets[name] for name in order]


def dataset_args_json(item: dict[str, Any]) -> str:
    path_env = str(item['path_env'])
    local_path = os.environ.get(path_env, '').strip()
    if not local_path:
        dataset_root = os.environ.get('DATASET_ROOT', '').strip()
        subdir = PATH_ENV_FALLBACK.get(path_env, '')
        if dataset_root and subdir:
            local_path = str(Path(dataset_root) / subdir)
    if not local_path:
        raise ValueError(f'{path_env} is not set. Source eval_protocol/env.sh first.')
    args = {'local_path': local_path}
    args.update(item.get('dataset_args') or {})
    return json.dumps({item['name']: args}, ensure_ascii=False)


def generation_config(protocol: dict[str, Any], max_tokens: int) -> str:
    return json.dumps(
        {
            'max_tokens': max_tokens,
            'temperature': protocol['temperature'],
            'do_sample': protocol['do_sample'],
            'extra_body': {'chat_template_kwargs': {'enable_thinking': protocol['enable_thinking']}},
        },
        ensure_ascii=False,
    )


def build_command(
    args: argparse.Namespace,
    protocol: dict[str, Any],
    item: dict[str, Any],
    work_dir: Path,
) -> list[str]:
    api_url = args.api_url.rstrip('/')
    if not api_url.endswith('/v1/chat/completions'):
        api_url = f'{api_url}/v1/chat/completions'
    command = [
        args.python_bin,
        '-m',
        'evalscope.cli.cli',
        'eval',
        '--model',
        args.model_id,
        '--model-id',
        f'{args.model_id}-{item["name"]}',
        '--eval-type',
        'openai_api',
        '--api-url',
        api_url,
        '--api-key',
        'EMPTY',
        '--datasets',
        item['name'],
        '--dataset-args',
        dataset_args_json(item),
        '--generation-config',
        generation_config(protocol, int(item['max_tokens'])),
        '--eval-batch-size',
        str(protocol['eval_batch_size']),
        '--seed',
        str(protocol['seed']),
        '--timeout',
        str(protocol['timeout']),
        '--work-dir',
        str(work_dir),
        '--no-timestamp',
    ]
    if item.get('limit') is not None and protocol.get('sample_policy') != 'full_split':
        command.extend(['--limit', str(item['limit'])])
    if (work_dir / 'predictions').is_dir():
        command.extend(['--use-cache', str(work_dir)])
    return command


def main() -> None:
    args = parse_args()
    protocol = load_protocol(args.protocol)
    experiment_dir = args.experiment_dir.expanduser().resolve()
    experiment_dir.mkdir(parents=True, exist_ok=True)

    repo_root = PROTOCOL_DIR.parent
    env = os.environ.copy()
    pythonpath = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = f'{repo_root}{os.pathsep}{pythonpath}' if pythonpath else str(repo_root)

    for item in selected_datasets(protocol, args.datasets):
        work_dir = experiment_dir / args.method / item['name']
        command = build_command(args, protocol, item, work_dir)
        print(' '.join(command), flush=True)
        if args.dry_run:
            continue
        subprocess.run(command, env=env, check=True)


if __name__ == '__main__':
    main()
