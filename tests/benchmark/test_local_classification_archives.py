# Copyright (c) Alibaba, Inc. and its affiliates.

import csv
import json
import zipfile
from pathlib import Path

from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig


def make_config(name: str, local_path: Path, **dataset_args) -> TaskConfig:
    return TaskConfig(
        model='mock-model',
        datasets=[name],
        eval_type='mock_llm',
        dataset_args={name: {
            'local_path': str(local_path),
            **dataset_args
        }},
    )


def write_mmlu_row(path: Path, answer: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as csv_file:
        csv.writer(csv_file).writerow(['Question?', 'one', 'two', 'three', 'four', answer])


def test_mmlu_loads_local_subject_csv_tree(tmp_path: Path) -> None:
    write_mmlu_row(tmp_path / 'dev' / 'abstract_algebra_dev.csv', 'A')
    write_mmlu_row(tmp_path / 'test' / 'abstract_algebra_test.csv', 'B')
    config = make_config('mmlu', tmp_path, subset_list=['abstract_algebra'], few_shot_num=1)

    adapter = get_benchmark('mmlu', config)
    dataset = adapter.load_dataset()

    assert len(dataset['abstract_algebra']) == 1
    assert dataset['abstract_algebra'][0].target == 'B'
    assert adapter.fewshot_dataset['abstract_algebra'][0].target == 'A'


def test_winogrande_loads_official_validation_from_zip(tmp_path: Path) -> None:
    archive_path = tmp_path / 'winogrande_1.1.zip'
    record = {
        'qID': 'question-1',
        'sentence': 'The trophy does not fit because _ is too large.',
        'option1': 'the trophy',
        'option2': 'the suitcase',
        'answer': '1',
    }
    with zipfile.ZipFile(archive_path, 'w') as archive:
        archive.writestr('winogrande_1.1/dev.jsonl', json.dumps(record) + '\n')
    config = make_config('winogrande', archive_path)

    dataset = get_benchmark('winogrande', config).load_dataset()

    assert len(dataset['default']) == 1
    assert dataset['default'][0].target == 'A'
    assert dataset['default'][0].metadata['id'] == 'question-1'
