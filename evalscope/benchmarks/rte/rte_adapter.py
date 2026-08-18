# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from typing import Any, Dict, Optional, Type

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import DataLoader, Dataset, Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate


@register_benchmark(
    BenchmarkMeta(
        name='rte',
        pretty_name='GLUE RTE',
        tags=[Tags.REASONING, Tags.READING_COMPREHENSION, Tags.MULTIPLE_CHOICE],
        description="""
## Overview

RTE is the Recognizing Textual Entailment task from GLUE. Given a premise and a hypothesis, a model determines whether
the premise entails the hypothesis.

## Task Description

- **Task Type**: Binary textual entailment
- **Input**: Premise and hypothesis
- **Output**: Entailment or Not entailment
- **Evaluation Split**: Validation

## Evaluation Notes

- This adapter follows the **GLUE RTE** label convention: `0` is entailment and `1` is not entailment
- Default configuration uses **0-shot** evaluation
- The validation split is used because public test labels are unavailable
""",
        dataset_id='nyu-mll/glue',
        subset_list=['rte'],
        default_subset='rte',
        metric_list=['acc'],
        few_shot_num=0,
        train_split='train',
        eval_split='validation',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER,
    )
)
class RTEAdapter(MultiChoiceAdapter):

    def _load_local_split(
        self,
        split: str,
        data_loader: Type[DataLoader],
        limit: Optional[int | float],
        repeats: int,
        shuffle: bool,
    ) -> Optional[Dataset]:
        if not os.path.isdir(self.dataset_id):
            return None

        return data_loader(
            data_id_or_path=self.dataset_id,
            split=split,
            subset='default',
            sample_fields=self.record_to_sample,
            filter_func=self.sample_filter,
            limit=limit,
            repeats=repeats,
            shuffle=shuffle,
            shuffle_choices=self.shuffle_choices,
            data_source=self.dataset_hub,
            force_redownload=self.force_redownload,
            dataset_dir=self.dataset_dir,
        ).load()

    def load_subset(self, subset: str, data_loader: Type[DataLoader]) -> Dataset:
        dataset = self._load_local_split(
            split=self.eval_split,
            data_loader=data_loader,
            limit=self.limit,
            repeats=self.repeats,
            shuffle=self.shuffle,
        )
        return dataset if dataset is not None else super().load_subset(subset, data_loader)

    def load_fewshot_subset(self, subset: str, data_loader: Type[DataLoader]) -> Dataset:
        dataset = self._load_local_split(
            split=self.train_split,
            data_loader=data_loader,
            limit=self.few_shot_num,
            repeats=1,
            shuffle=self.few_shot_random,
        )
        return dataset if dataset is not None else super().load_fewshot_subset(subset, data_loader)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        label = int(record['label'])
        if label not in (0, 1):
            raise ValueError(f'Unexpected GLUE RTE label: {label}')

        return Sample(
            input=f"Premise: {record['sentence1']}\nHypothesis: {record['sentence2']}",
            choices=['Entailment', 'Not entailment'],
            target='A' if label == 0 else 'B',
            metadata={'id': record.get('idx', record.get('id', 'unknown'))},
        )