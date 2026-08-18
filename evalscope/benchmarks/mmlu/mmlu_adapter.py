# Copyright (c) Alibaba, Inc. and its affiliates.

import csv
import os
from typing import Type

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import DataLoader, Dataset, DictDataLoader, Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import MultipleChoiceTemplate

logger = get_logger()

SUBJECT_MAPPING = {
    'abstract_algebra': ['Abstract Algebra', 'math', 'STEM'],
    'anatomy': ['Anatomy', 'health', 'Other'],
    'astronomy': ['Astronomy', 'physics', 'STEM'],
    'business_ethics': ['Business Ethics', 'business', 'Other'],
    'clinical_knowledge': ['Clinical Knowledge', 'health', 'Other'],
    'college_biology': ['College Biology', 'biology', 'STEM'],
    'college_chemistry': ['College Chemistry', 'chemistry', 'STEM'],
    'college_computer_science': ['College Computer Science', 'computer science', 'STEM'],
    'college_mathematics': ['College Mathematics', 'math', 'STEM'],
    'college_medicine': ['College Medicine', 'health', 'Other'],
    'college_physics': ['College Physics', 'physics', 'STEM'],
    'computer_security': ['Computer Security', 'computer science', 'STEM'],
    'conceptual_physics': ['Conceptual Physics', 'physics', 'STEM'],
    'econometrics': ['Econometrics', 'economics', 'Social Science'],
    'electrical_engineering': ['Electrical Engineering', 'engineering', 'STEM'],
    'elementary_mathematics': ['Elementary Mathematics', 'math', 'STEM'],
    'formal_logic': ['Formal Logic', 'philosophy', 'Humanities'],
    'global_facts': ['Global Facts', 'other', 'Other'],
    'high_school_biology': ['High School Biology', 'biology', 'STEM'],
    'high_school_chemistry': ['High School Chemistry', 'chemistry', 'STEM'],
    'high_school_computer_science': ['High School Computer Science', 'computer science', 'STEM'],
    'high_school_european_history': ['High School European History', 'history', 'Humanities'],
    'high_school_geography': ['High School Geography', 'geography', 'Social Science'],
    'high_school_government_and_politics': ['High School Government And Politics', 'politics', 'Social Science'],
    'high_school_macroeconomics': ['High School Macroeconomics', 'economics', 'Social Science'],
    'high_school_mathematics': ['High School Mathematics', 'math', 'STEM'],
    'high_school_microeconomics': ['High School Microeconomics', 'economics', 'Social Science'],
    'high_school_physics': ['High School Physics', 'physics', 'STEM'],
    'high_school_psychology': ['High School Psychology', 'psychology', 'Social Science'],
    'high_school_statistics': ['High School Statistics', 'math', 'STEM'],
    'high_school_us_history': ['High School Us History', 'history', 'Humanities'],
    'high_school_world_history': ['High School World History', 'history', 'Humanities'],
    'human_aging': ['Human Aging', 'health', 'Other'],
    'human_sexuality': ['Human Sexuality', 'culture', 'Social Science'],
    'international_law': ['International Law', 'law', 'Humanities'],
    'jurisprudence': ['Jurisprudence', 'law', 'Humanities'],
    'logical_fallacies': ['Logical Fallacies', 'philosophy', 'Humanities'],
    'machine_learning': ['Machine Learning', 'computer science', 'STEM'],
    'management': ['Management', 'business', 'Other'],
    'marketing': ['Marketing', 'business', 'Other'],
    'medical_genetics': ['Medical Genetics', 'health', 'Other'],
    'miscellaneous': ['Miscellaneous', 'other', 'Other'],
    'moral_disputes': ['Moral Disputes', 'philosophy', 'Humanities'],
    'moral_scenarios': ['Moral Scenarios', 'philosophy', 'Humanities'],
    'nutrition': ['Nutrition', 'health', 'Other'],
    'philosophy': ['Philosophy', 'philosophy', 'Humanities'],
    'prehistory': ['Prehistory', 'history', 'Humanities'],
    'professional_accounting': ['Professional Accounting', 'other', 'Other'],
    'professional_law': ['Professional Law', 'law', 'Humanities'],
    'professional_medicine': ['Professional Medicine', 'health', 'Other'],
    'professional_psychology': ['Professional Psychology', 'psychology', 'Social Science'],
    'public_relations': ['Public Relations', 'politics', 'Social Science'],
    'security_studies': ['Security Studies', 'politics', 'Social Science'],
    'sociology': ['Sociology', 'culture', 'Social Science'],
    'us_foreign_policy': ['Us Foreign Policy', 'politics', 'Social Science'],
    'virology': ['Virology', 'health', 'Other'],
    'world_religions': ['World Religions', 'philosophy', 'Humanities'],
}


@register_benchmark(
    BenchmarkMeta(
        name='mmlu',
        pretty_name='MMLU',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE],
        description="""
## Overview

MMLU (Massive Multitask Language Understanding) is a comprehensive evaluation benchmark designed to measure knowledge acquired during pretraining. It covers 57 subjects across STEM, humanities, social sciences, and other domains, ranging from elementary to professional difficulty levels.

## Task Description

- **Task Type**: Multiple-Choice Question Answering
- **Input**: Question with four answer choices (A, B, C, D)
- **Output**: Single correct answer letter
- **Subjects**: 57 subjects organized into 4 categories (STEM, Humanities, Social Sciences, Other)

## Key Features

- Covers diverse knowledge domains from elementary to advanced professional levels
- Tests both factual knowledge and reasoning abilities
- Includes subjects like abstract algebra, anatomy, astronomy, business ethics, and more
- Standard benchmark for measuring LLM knowledge breadth

## Evaluation Notes

- Default configuration uses **5-shot** examples from the dev split
- Supports Chain-of-Thought (CoT) prompting for improved reasoning
- Results can be aggregated by subject or category (STEM, Humanities, Social Sciences, Other)
- Use `subset_list` parameter to evaluate specific subjects
""",
        dataset_id='cais/mmlu',
        metric_list=['acc'],
        subset_list=list(SUBJECT_MAPPING.keys()),
        default_subset='all',
        few_shot_num=5,
        train_split='dev',
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class MMLUAdapter(MultiChoiceAdapter):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.reformat_subset = True
        self.category_map = {k: v[-1] for k, v in SUBJECT_MAPPING.items()}

    def _load_local_csv_split(self, split: str) -> Dataset:
        split_dir = os.path.join(self.dataset_id, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f'MMLU split directory does not exist: {split_dir}')

        records = []
        for subject in self.subset_list:
            csv_path = os.path.join(split_dir, f'{subject}_{split}.csv')
            if not os.path.isfile(csv_path):
                raise FileNotFoundError(f'MMLU subject file does not exist: {csv_path}')
            with open(csv_path, newline='', encoding='utf-8') as csv_file:
                for row in csv.reader(csv_file):
                    if len(row) != 6:
                        raise ValueError(f'Expected 6 columns in {csv_path}, got {len(row)}')
                    records.append({
                        'question': row[0],
                        'choices': row[1:5],
                        'answer': 'ABCD'.index(row[5]),
                        'subject': subject,
                    })

        return DictDataLoader(
            dict_list=records,
            sample_fields=self.record_to_sample,
            filter_func=self.sample_filter,
            shuffle=self.shuffle if split == self.eval_split else self.few_shot_random,
            shuffle_choices=self.shuffle_choices,
            data_source=self.dataset_hub,
        ).load()

    def load_subset(self, subset: str, data_loader: Type[DataLoader]) -> Dataset:
        if os.path.isdir(self.dataset_id):
            return self._load_local_csv_split(self.eval_split)
        return super().load_subset(subset, data_loader)

    def load_fewshot_subset(self, subset: str, data_loader: Type[DataLoader]) -> Dataset:
        if os.path.isdir(self.dataset_id):
            return self._load_local_csv_split(self.train_split)
        return super().load_fewshot_subset(subset, data_loader)

    def record_to_sample(self, record) -> Sample:
        return Sample(
            input=record['question'],
            choices=record['choices'],
            # converts 0 -> A, 1 -> B, etc.
            target=('ABCD'[record['answer']]),
            subset_key=record['subject'],
            metadata={'subject': record['subject']},
        )
