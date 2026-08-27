"""Unit tests for native calibration packing and degeneration gates."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("build_native_calibration.py")
SPEC = importlib.util.spec_from_file_location("build_native_calibration", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_repeat_metrics_detect_constant_token_failure() -> None:
    metrics = MODULE._repeat_metrics([7] * 100)
    assert metrics["distinct_token_ratio"] == 0.01
    assert metrics["dominant_token_ratio"] == 1.0
    assert metrics["max_run_ratio"] == 1.0
    assert metrics["repeated_4gram_ratio"] > 0.9
    assert metrics["periodic_loop_ratio"] == 1.0


def test_repeat_metrics_detect_short_period_loop() -> None:
    episode = MODULE.Episode(
        token_ids=[1, 2, 3, 4] * 32,
        user_tokens=32,
        assistant_tokens=64,
        user_terminated=False,
        assistant_terminated=False,
        seed=42,
    )
    valid, metrics = MODULE._is_valid_episode(episode)
    assert not valid
    assert metrics["user"]["periodic_loop_ratio"] == 1.0


def test_validity_gate_accepts_varied_episode() -> None:
    episode = MODULE.Episode(
        token_ids=list(range(64)),
        user_tokens=24,
        assistant_tokens=24,
        user_terminated=True,
        assistant_terminated=True,
        seed=42,
        user_token_ids=tuple(range(24)),
        assistant_token_ids=tuple(range(24, 48)),
    )
    valid, _ = MODULE._is_valid_episode(episode)
    assert valid


def test_text_metrics_detect_repeated_fragments() -> None:
    metrics = MODULE._text_metrics("implementation-implementation " * 20)
    assert metrics["repeated_word_ratio"] > 0.9

    metrics = MODULE._text_metrics("ትን-ትን-ትን " * 20)
    assert metrics["repeated_word_ratio"] > 0.9

    metrics = MODULE._text_metrics("de-spa-el de-spa-el " * 20)
    assert metrics["repeated_word_ratio"] > 0.9


def test_user_gate_rejects_token_level_phrase_loop() -> None:
    tokens = [1, 2, 3, 4, 5, 6] * 20
    valid, metrics = MODULE._is_valid_turn(tokens, role="user")
    assert not valid
    assert metrics["repeated_4gram_ratio"] > 0.6


def test_bad_user_is_rejected_before_response_quality_can_dilute_it() -> None:
    episode = MODULE.Episode(
        token_ids=list(range(24)) + [99] * 80,
        user_tokens=24,
        assistant_tokens=80,
        user_terminated=True,
        assistant_terminated=True,
        seed=42,
        user_token_ids=tuple([1, 2, 3, 4] * 6),
        assistant_token_ids=tuple(range(80)),
    )
    valid, metrics = MODULE._is_valid_episode(episode)
    assert not valid
    assert metrics["user"]["repeated_4gram_ratio"] > 0.6
    assert metrics["assistant"]["dominant_token_ratio"] < 0.1


def test_packing_preserves_fixed_blocks_and_episode_boundaries() -> None:
    episodes = [
        MODULE.Episode(list(range(20)), 4, 4, True, True, 1),
        MODULE.Episode(list(range(20, 45)), 5, 9, True, False, 2),
        MODULE.Episode(list(range(45, 65)), 5, 7, False, True, 3),
    ]
    tokens, episode_records, block_records = MODULE._pack_blocks(episodes, blocks=2, block_length=32)
    assert tuple(tokens.shape) == (1, 64)
    assert len(block_records) == 2
    assert block_records[-1]["end"] == 64
    assert episode_records[0]["start"] == 0
    assert episode_records[0]["end"] == 20
    assert any(record["complete"] is False for record in episode_records)


def test_prefix_temperature_warmup_restores_default_sampling() -> None:
    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeCandidate:
        def __init__(self, tokens, finish_reason):
            self.token_ids = tokens
            self.finish_reason = finish_reason

    class FakeOutput:
        def __init__(self, tokens, finish_reason):
            self.outputs = [FakeCandidate(tokens, finish_reason)]

    class FakeLlm:
        def __init__(self):
            self.temperatures = []

        def generate(self, prompts, params, use_tqdm=False):
            self.temperatures.extend(param.kwargs["temperature"] for param in params)
            return [FakeOutput(list(range(param.kwargs["max_tokens"])), "length") for param in params]

    llm = FakeLlm()
    results = MODULE._generate_batch(
        llm,
        FakeSamplingParams,
        [[1, 2, 3]],
        max_tokens=12,
        stop_token_id=99,
        seeds=[42],
        warmup_temperature=1.5,
        warmup_tokens=4,
    )
    assert llm.temperatures == [1.5, 1.0]
    assert len(results[0][0]) == 12