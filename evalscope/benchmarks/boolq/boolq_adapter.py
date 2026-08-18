# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate


@register_benchmark(
    BenchmarkMeta(
        name='boolq',
        pretty_name='BoolQ',
        tags=[Tags.READING_COMPREHENSION, Tags.REASONING, Tags.YES_NO],
        description="""
## Overview

BoolQ evaluates yes/no question answering over short passages. Questions are naturally occurring queries paired with
Wikipedia passages, and answering them requires reading comprehension and inference.

## Task Description

- **Task Type**: Binary reading comprehension
- **Input**: Passage and question
- **Output**: No or Yes
- **Evaluation Split**: Validation

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Boolean labels are presented as a two-choice question and scored with accuracy
- The adapter accepts both the canonical `answer` field and mirrors that expose the label as `label`
""",
        dataset_id='google/boolq',
        metric_list=['acc'],
        few_shot_num=0,
        train_split='train',
        eval_split='validation',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER,
    )
)
class BoolQAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        label = record['answer'] if 'answer' in record else record['label']
        return Sample(
            input=f"Passage: {record['passage']}\nQuestion: {record['question']}",
            choices=['No', 'Yes'],
            target='B' if bool(label) else 'A',
            metadata={'id': record.get('idx', record.get('id', 'unknown'))},
        )