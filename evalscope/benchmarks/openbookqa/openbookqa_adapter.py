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
        name='openbookqa',
        pretty_name='OpenBookQA',
        tags=[Tags.KNOWLEDGE, Tags.REASONING, Tags.MULTIPLE_CHOICE],
        description="""
## Overview

OpenBookQA is a multiple-choice science benchmark modeled after open-book exams. It combines a small collection of
elementary science facts with questions that often require additional commonsense reasoning.

## Task Description

- **Task Type**: Multiple-choice science question answering
- **Input**: Science question with four answer choices
- **Output**: Correct answer letter
- **Subsets**: Main and Additional

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- The Main subset is selected by default
- Local snapshots may store each subset in a separate `main/` or `additional/` Parquet directory
""",
        dataset_id='allenai/openbookqa',
        subset_list=['main'],
        default_subset='main',
        metric_list=['acc'],
        few_shot_num=0,
        train_split='train',
        eval_split='validation',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER,
    )
)
class OpenBookQAAdapter(MultiChoiceAdapter):

    def _load_local_split(
        self,
        subset: str,
        split: str,
        data_loader: Type[DataLoader],
        limit: Optional[int | float],
        repeats: int,
        shuffle: bool,
    ) -> Optional[Dataset]:
        if not os.path.isdir(self.dataset_id):
            return None

        subset_path = os.path.join(self.dataset_id, subset)
        data_path = subset_path if os.path.isdir(subset_path) else self.dataset_id
        return data_loader(
            data_id_or_path=data_path,
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
            subset=subset,
            split=self.eval_split,
            data_loader=data_loader,
            limit=self.limit,
            repeats=self.repeats,
            shuffle=self.shuffle,
        )
        return dataset if dataset is not None else super().load_subset(subset, data_loader)

    def load_fewshot_subset(self, subset: str, data_loader: Type[DataLoader]) -> Dataset:
        dataset = self._load_local_split(
            subset=subset,
            split=self.train_split,
            data_loader=data_loader,
            limit=self.few_shot_num,
            repeats=1,
            shuffle=self.few_shot_random,
        )
        return dataset if dataset is not None else super().load_fewshot_subset(subset, data_loader)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        choices = record['choices']
        answer_key = record['answerKey']
        labels = choices.get('label', [])
        if answer_key in labels:
            answer_key = chr(ord('A') + labels.index(answer_key))

        return Sample(
            input=record['question_stem'],
            choices=choices['text'],
            target=answer_key,
            metadata={'id': record.get('id', 'unknown')},
        )