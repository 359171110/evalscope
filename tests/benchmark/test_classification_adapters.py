# Copyright (c) Alibaba, Inc. and its affiliates.

import json
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.boolq.boolq_adapter import BoolQAdapter
from evalscope.benchmarks.openbookqa.openbookqa_adapter import OpenBookQAAdapter
from evalscope.benchmarks.rte.rte_adapter import RTEAdapter
from evalscope.config import TaskConfig


def make_config(name: str, local_path: Path, dataset_dir: Path, **dataset_args) -> TaskConfig:
    return TaskConfig(
        model='mock-model',
        datasets=[name],
        eval_type='mock_llm',
        dataset_dir=str(dataset_dir),
        dataset_args={name: {
            'local_path': str(local_path),
            **dataset_args
        }},
    )


def test_boolq_registration_and_sample_conversion(tmp_path: Path) -> None:
    adapter = get_benchmark('boolq', make_config('boolq', tmp_path, tmp_path / 'cache'))
    sample = adapter.record_to_sample({
        'question': 'is the sky blue',
        'passage': 'The daytime sky appears blue.',
        'label': True,
        'idx': 7,
    })

    assert isinstance(adapter, BoolQAdapter)
    assert sample.input == 'Passage: The daytime sky appears blue.\nQuestion: is the sky blue'
    assert sample.choices == ['No', 'Yes']
    assert sample.target == 'B'
    assert sample.metadata['id'] == 7


def test_boolq_accepts_canonical_answer_field(tmp_path: Path) -> None:
    adapter = get_benchmark('boolq', make_config('boolq', tmp_path, tmp_path / 'cache'))
    sample = adapter.record_to_sample({'question': 'q', 'passage': 'p', 'answer': False})

    assert sample.target == 'A'


def test_openbookqa_registration_and_label_mapping(tmp_path: Path) -> None:
    adapter = get_benchmark('openbookqa', make_config('openbookqa', tmp_path, tmp_path / 'cache'))
    sample = adapter.record_to_sample({
        'id': 'question-1',
        'question_stem': 'Which object conducts electricity?',
        'choices': {
            'text': ['rubber', 'copper', 'glass', 'wood'],
            'label': ['D', 'B', 'A', 'C'],
        },
        'answerKey': 'B',
    })

    assert isinstance(adapter, OpenBookQAAdapter)
    assert sample.choices == ['rubber', 'copper', 'glass', 'wood']
    assert sample.target == 'B'
    assert sample.metadata['id'] == 'question-1'


def test_openbookqa_loads_subset_parquet_directory(tmp_path: Path) -> None:
    subset_dir = tmp_path / 'main'
    subset_dir.mkdir()
    table = pa.Table.from_pylist([{
        'id': 'question-1',
        'question_stem': 'Which object conducts electricity?',
        'choices': {
            'text': ['rubber', 'copper', 'glass', 'wood'],
            'label': ['A', 'B', 'C', 'D'],
        },
        'answerKey': 'B',
    }])
    pq.write_table(table, subset_dir / 'validation-00000-of-00001.parquet')
    config = make_config(
        'openbookqa',
        tmp_path,
        tmp_path / 'cache',
        subset_list=['main'],
        eval_split='validation',
    )

    dataset = get_benchmark('openbookqa', config).load_dataset()

    assert len(dataset['main']) == 1
    assert dataset['main'][0].target == 'B'


def test_rte_registration_and_glue_label_mapping(tmp_path: Path) -> None:
    adapter = get_benchmark('rte', make_config('rte', tmp_path, tmp_path / 'cache', subset_list=['rte']))
    entailment = adapter.record_to_sample({
        'sentence1': 'A dog is running.',
        'sentence2': 'An animal is moving.',
        'label': 0,
        'idx': 3,
    })
    not_entailment = adapter.record_to_sample({
        'sentence1': 'A dog is running.',
        'sentence2': 'No animal is moving.',
        'label': 1,
        'idx': 4,
    })

    assert isinstance(adapter, RTEAdapter)
    assert entailment.choices == ['Entailment', 'Not entailment']
    assert entailment.target == 'A'
    assert not_entailment.target == 'B'


def test_rte_loads_local_parquet_directory(tmp_path: Path) -> None:
    table = pa.Table.from_pylist([{
        'sentence1': 'A dog is running.',
        'sentence2': 'An animal is moving.',
        'label': 0,
        'idx': 3,
    }])
    pq.write_table(table, tmp_path / 'validation-00000-of-00001.parquet')
    config = make_config(
        'rte',
        tmp_path,
        tmp_path / 'cache',
        subset_list=['rte'],
        eval_split='validation',
    )

    dataset = get_benchmark('rte', config).load_dataset()

    assert len(dataset['rte']) == 1
    assert dataset['rte'][0].target == 'A'


def test_rte_rejects_unlabeled_test_rows(tmp_path: Path) -> None:
    adapter = get_benchmark('rte', make_config('rte', tmp_path, tmp_path / 'cache', subset_list=['rte']))

    try:
        adapter.record_to_sample({'sentence1': 'premise', 'sentence2': 'hypothesis', 'label': -1})
    except ValueError as exc:
        assert str(exc) == 'Unexpected GLUE RTE label: -1'
    else:
        raise AssertionError('Expected a ValueError for an unlabeled GLUE RTE row.')
