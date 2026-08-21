from __future__ import annotations

import json
from pathlib import Path

PROTOCOL_DIR = Path(__file__).resolve().parents[2] / 'eval_protocol'
QUICK9_EXPECTED = {
    'arc': 600,
    'hellaswag': 1000,
    'winogrande': 400,
    'gsm8k': 128,
    'math_500': 100,
    'mmlu': 570,
}
FULL6_EXPECTED = {
    'arc': 3548,
    'hellaswag': 10042,
    'winogrande': 1267,
    'gsm8k': 1319,
    'math_500': 500,
    'mmlu': 14042,
}
MAX_TOKENS = {
    'arc': 2048,
    'hellaswag': 512,
    'winogrande': 1024,
    'gsm8k': 2048,
    'math_500': 4096,
    'mmlu': 2048,
}
FULL8_EXPECTED = {
    **FULL6_EXPECTED,
    'humaneval': 164,
    'mbpp': 500,
}
FULL8_MAX_TOKENS = {
    **MAX_TOKENS,
    'humaneval': 1024,
    'mbpp': 1024,
}


def load_protocol(name: str) -> dict:
    payload = json.loads((PROTOCOL_DIR / f'{name}.json').read_text(encoding='utf-8'))
    assert payload['name'] == name
    return payload


def test_quick9_is_subset_screening() -> None:
    protocol = load_protocol('quick9')
    assert protocol['sample_policy'] == 'subset'
    assert protocol['dataset_order'] == ['arc', 'hellaswag', 'winogrande', 'gsm8k', 'math_500', 'mmlu']
    assert protocol['seed'] == 42
    assert protocol['enable_thinking'] is False
    counts = {item['name']: item['expected_samples'] for item in protocol['datasets']}
    tokens = {item['name']: item['max_tokens'] for item in protocol['datasets']}
    assert counts == QUICK9_EXPECTED
    assert tokens == MAX_TOKENS
    assert protocol['expected_total'] == sum(QUICK9_EXPECTED.values())


def test_full6_v1_is_full_split_with_generation_caps() -> None:
    protocol = load_protocol('full6_v1')
    assert protocol['sample_policy'] == 'full_split'
    assert protocol['dataset_order'] == ['arc', 'hellaswag', 'winogrande', 'gsm8k', 'math_500', 'mmlu']
    counts = {item['name']: item['expected_samples'] for item in protocol['datasets']}
    tokens = {item['name']: item['max_tokens'] for item in protocol['datasets']}
    assert counts == FULL6_EXPECTED
    assert tokens == MAX_TOKENS
    assert protocol['expected_total'] == sum(FULL6_EXPECTED.values())
    assert all(item.get('limit') is None for item in protocol['datasets'])
    assert protocol['expected_total'] != sum(QUICK9_EXPECTED.values())


def test_full8_v1_extends_full6_with_code_benchmarks() -> None:
    protocol = load_protocol('full8_v1')
    assert protocol['sample_policy'] == 'full_split'
    assert protocol['dataset_order'] == [
        'arc',
        'hellaswag',
        'winogrande',
        'gsm8k',
        'math_500',
        'mmlu',
        'humaneval',
        'mbpp',
    ]
    counts = {item['name']: item['expected_samples'] for item in protocol['datasets']}
    tokens = {item['name']: item['max_tokens'] for item in protocol['datasets']}
    assert counts == FULL8_EXPECTED
    assert tokens == FULL8_MAX_TOKENS
    assert protocol['expected_total'] == sum(FULL8_EXPECTED.values())
    assert protocol['expected_total'] == sum(FULL6_EXPECTED.values()) + 164 + 500
    assert all(item.get('limit') is None for item in protocol['datasets'])
    humaneval = next(item for item in protocol['datasets'] if item['name'] == 'humaneval')
    mbpp = next(item for item in protocol['datasets'] if item['name'] == 'mbpp')
    assert humaneval['path_env'] == 'HUMANEVAL_PATH'
    assert mbpp['path_env'] == 'MBPP_PATH'
    assert mbpp.get('dataset_args', {}).get('few_shot_num') == 3
