"""Frozen full6_v1 evaluation protocol for NAPS-v2 experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for one full6_v1 dataset shard."""

    name: str
    limit: int
    max_tokens: int
    dataset_args: str


PROTOCOL_NAME = "full6_v1"
SEED = 42
EVAL_BATCH_SIZE = 16
TIMEOUT = 1200
TEMPERATURE = 0.0
DO_SAMPLE = False
ENABLE_THINKING = False

DATASETS = (
    DatasetConfig(
        "arc",
        3548,
        2048,
        '{"arc":{"local_path":"/data01/datasets/evalscope_benchmarks/arc",'
        '"subset_list":["ARC-Challenge","ARC-Easy"]}}',
    ),
    DatasetConfig(
        "hellaswag",
        10042,
        512,
        '{"hellaswag":{"local_path":"/data01/datasets/evalscope_benchmarks/hellaswag"}}',
    ),
    DatasetConfig(
        "winogrande",
        1267,
        1024,
        '{"winogrande":{"local_path":"/data01/home/xuzk/workspace/OSTQuant/'
        'utils/my_utils/llm_pipeline/datasets/winogrande/data/winogrande_1.1.zip"}}',
    ),
    DatasetConfig(
        "gsm8k",
        1319,
        2048,
        '{"gsm8k":{"local_path":"/data01/datasets/evalscope_benchmarks/gsm8k",'
        '"few_shot_num":0}}',
    ),
    DatasetConfig(
        "math_500",
        500,
        4096,
        '{"math_500":{"local_path":"/data01/datasets/evalscope_benchmarks/math_500"}}',
    ),
    DatasetConfig(
        "mmlu",
        14042,
        2048,
        '{"mmlu":{"local_path":"/data01/home/xuzk/workspace/OSTQuant/'
        'utils/my_utils/llm_pipeline/datasets/mmlu"}}',
    ),
)


def generation_config(max_tokens: int) -> dict[str, object]:
    """Return the request generation payload for one dataset."""

    return {
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
        "do_sample": DO_SAMPLE,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": ENABLE_THINKING}},
    }


def validate_protocol() -> None:
    """Fail fast if the frozen dataset order or counts drift."""

    expected = (
        ("arc", 3548, 2048),
        ("hellaswag", 10042, 512),
        ("winogrande", 1267, 1024),
        ("gsm8k", 1319, 2048),
        ("math_500", 500, 4096),
        ("mmlu", 14042, 2048),
    )
    actual = tuple((item.name, item.limit, item.max_tokens) for item in DATASETS)
    if actual != expected:
        raise RuntimeError(f"{PROTOCOL_NAME} drifted: {actual!r}")


validate_protocol()