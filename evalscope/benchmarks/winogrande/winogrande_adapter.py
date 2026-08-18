import json
import os
import zipfile
from typing import Type

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import DataLoader, Dataset, DictDataLoader, Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate


@register_benchmark(
    BenchmarkMeta(
        name='winogrande',
        pretty_name='Winogrande',
        tags=[Tags.REASONING, Tags.MULTIPLE_CHOICE],
        description="""
## Overview

Winogrande is a large-scale benchmark for commonsense reasoning, specifically designed to test pronoun resolution in the Winograd Schema Challenge format. It contains 44K problems that require understanding of physical and social commonsense.

## Task Description

- **Task Type**: Pronoun Resolution / Commonsense Reasoning
- **Input**: Sentence with ambiguous pronoun and two options
- **Output**: Correct option (A or B) that resolves the pronoun
- **Format**: Binary choice between two noun phrases

## Key Features

- 44K Winograd-style pronoun resolution problems
- Adversarially filtered to reduce dataset biases
- Tests physical commonsense (object properties, actions)
- Tests social commonsense (intentions, emotions)
- Requires understanding context to resolve ambiguity

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Binary choice format (option1 vs option2)
- Answers are converted to A/B letter format
- Simple accuracy metric for evaluation
- Commonly used for commonsense reasoning assessment
""",
        dataset_id='AI-ModelScope/winogrande_val',
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='validation',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER,
    )
)
class WinograndeAdapter(MultiChoiceAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_subset(self, subset: str, data_loader: Type[DataLoader]) -> Dataset:
        if not os.path.isfile(self.dataset_id) or not zipfile.is_zipfile(self.dataset_id):
            return super().load_subset(subset, data_loader)

        with zipfile.ZipFile(self.dataset_id) as archive:
            with archive.open('winogrande_1.1/dev.jsonl') as jsonl_file:
                records = [json.loads(line) for line in jsonl_file]

        return DictDataLoader(
            dict_list=records,
            sample_fields=self.record_to_sample,
            filter_func=self.sample_filter,
            limit=self.limit,
            repeats=self.repeats,
            shuffle=self.shuffle,
            shuffle_choices=self.shuffle_choices,
            data_source=self.dataset_hub,
        ).load()

    def record_to_sample(self, record) -> Sample:
        return Sample(
            input=record['sentence'],
            choices=[record['option1'], record['option2']],
            target=chr(ord('A') + int(record['answer']) - 1),  # Convert 1,2 to A,B
            metadata={'id': record.get('id', record.get('qID', 'unknown'))},
        )
