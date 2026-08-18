from __future__ import annotations

import json
from pathlib import Path

PROTOCOL_DIR = Path(__file__).resolve().parents[2] / 'eval_protocol'
SUBSET_EXPECTED = {
    'arc': 600,
    'hellaswag': 1000,
    'winogrande': 400,
    'gsm8k': 128,
    'math_500': 100,
    'mmlu': 570,
}


def load_protocol(name: str) -> dict:
    payload = json.loads((PROTOCOL_DIR / f'{name}.json').read_text(encoding='utf-8'))
    assert payload['name'] == name
    return payload


def test_quick9_and_full6_v1_share_subset_counts() -> None:
    for name in ('quick9', 'full6_v1'):
        protocol = load_protocol(name)
        assert protocol['dataset_order'] == ['arc', 'hellaswag', 'winogrande', 'gsm8k', 'math_500', 'mmlu']
        assert protocol['seed'] == 42
        assert protocol['enable_thinking'] is False
        counts = {item['name']: item['expected_samples'] for item in protocol['datasets']}
        assert counts == SUBSET_EXPECTED
        assert protocol['expected_total'] == sum(SUBSET_EXPECTED.values())


def test_full6_unlimited_is_full_split_and_not_comparable() -> None:
    protocol = load_protocol('full6_unlimited')
    counts = {item['name']: item['expected_samples'] for item in protocol['datasets']}
    assert counts['arc'] == 3548
    assert counts['math_500'] == 500
    assert counts['mmlu'] == 14042
    assert protocol['expected_total'] != sum(SUBSET_EXPECTED.values())
