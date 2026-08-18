from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pytest
import torch

from scripts import run_evalscope_static_profile
from scripts.run_evalscope_static_profile import (
    normalize_dataset_args,
    parse_dataset_limits,
    parse_limit,
    parse_sha256,
    source_tree_identity,
    validate_visible_gpus,
)


def test_evalscope_runner_accepts_explicit_physical_gpus() -> None:
    assert validate_visible_gpus("0") == ["0"]
    assert validate_visible_gpus("4") == ["4"]
    assert validate_visible_gpus("0,4,5,7") == ["0", "4", "5", "7"]

    for value in (None, "", "4,4", "4,-1", "gpu4"):
        with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
            validate_visible_gpus(value)


def test_evalscope_runner_preserves_limit_count_and_fraction_semantics() -> None:
    assert parse_limit("1") == 1
    assert isinstance(parse_limit("1"), int)
    assert parse_limit("0.1") == pytest.approx(0.1)
    assert isinstance(parse_limit("0.1"), float)

    for value in ("0", "-1", "1.1"):
        with pytest.raises(Exception):
            parse_limit(value)


def test_evalscope_runner_parses_per_dataset_limits() -> None:
    limits = parse_dataset_limits('{"arc":400,"math_500":40,"mmlu_pro":100,"ifeval":541}')

    assert limits == {"arc": 400, "math_500": 40, "mmlu_pro": 100, "ifeval": 541}

    with pytest.raises(argparse.ArgumentTypeError, match="invalid limit"):
        parse_dataset_limits('{"arc":0}')


def test_evalscope_runner_applies_per_dataset_limits_to_dataset_args() -> None:
    normalized = normalize_dataset_args(
        ["arc", "math_500", "mmlu_pro"],
        {},
        {"arc": 400, "math_500": 40, "mmlu_pro": 100},
    )

    assert normalized["arc"]["limit"] == 400
    assert normalized["math_500"]["limit"] == 40
    assert normalized["mmlu_pro"]["limit"] == 100

    with pytest.raises(ValueError, match="not selected"):
        normalize_dataset_args(["arc"], {}, {"mmlu_pro": 100})


def test_evalscope_runner_uses_parseable_boxed_math_prompt() -> None:
    normalized = normalize_dataset_args(
        ["gsm8k", "math_500"],
        {"gsm8k": {"local_path": "/data/gsm8k"}},
    )

    assert normalized["gsm8k"]["few_shot_num"] == 0
    for dataset in ("gsm8k", "math_500"):
        prompt = normalized[dataset]["prompt_template"].format(question="Q")
        assert "\\boxed expression" in prompt
        assert "ANSWER" not in prompt
        assert "inside its braces" in prompt
        assert "empty box" in prompt
    assert normalized["gsm8k"]["local_path"] == "/data/gsm8k"


def test_evalscope_runner_preserves_explicit_math_prompt_override() -> None:
    normalized = normalize_dataset_args(
        ["gsm8k"],
        {"gsm8k": {"prompt_template": "{question}\nAnswer: NUMBER", "few_shot_num": 2}},
    )

    assert normalized["gsm8k"]["prompt_template"] == "{question}\nAnswer: NUMBER"
    assert normalized["gsm8k"]["few_shot_num"] == 2


def test_evalscope_runner_accepts_only_sha256_hex_values() -> None:
    digest = "a" * 64
    assert parse_sha256(digest) == digest

    for value in ("", "a" * 63, "a" * 65, "g" * 64):
        with pytest.raises(argparse.ArgumentTypeError, match="SHA256"):
            parse_sha256(value)


def test_evalscope_runner_hashes_dirty_runtime_source_tree(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    source = repository / "code" / "runtime.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(["git", "-C", str(repository), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repository), "config", "user.name", "Test User"], check=True)
    subprocess.run(["git", "-C", str(repository), "add", "code/runtime.py"], check=True)
    subprocess.run(["git", "-C", str(repository), "commit", "-qm", "initial"], check=True)

    clean = source_tree_identity(repository, pathspecs=("code",))
    source.write_text("VALUE = 2\n", encoding="utf-8")
    dirty = source_tree_identity(repository, pathspecs=("code",))

    assert clean["commit"] == dirty["commit"]
    assert clean["runtime_tree_dirty"] is False
    assert dirty["runtime_tree_dirty"] is True
    assert clean["runtime_tree_sha256"] != dirty["runtime_tree_sha256"]
    assert dirty["runtime_file_count"] == 1


def test_evalscope_runner_preflight_only_does_not_register_model_api(tmp_path: Path, monkeypatch) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    profile_path = tmp_path / "profile.pt"
    channel_path = tmp_path / "channel.pt"
    profile_path.write_bytes(b"profile")
    channel_path.write_bytes(b"channel")
    profile_hash = "a" * 64
    channel_hash = "b" * 64
    work_dir = tmp_path / "outputs"
    profile = {
        "method": "test",
        "mode": "test",
        "test_metrics_used_for_profile": False,
    }
    args = argparse.Namespace(
        model_path=str(model_path),
        model_id="test-model",
        model_family="qwen3",
        profile=profile_path,
        channel_cache=channel_path,
        expected_profile_file_sha256=profile_hash,
        expected_channel_file_sha256=channel_hash,
        work_dir=work_dir,
        stats_path=None,
        datasets=["gsm8k"],
        dataset_args={},
        dataset_dir=None,
        dataset_hub="modelscope",
        generation_config={},
        limit=1,
        eval_batch_size=1,
        seed=42,
        correction_mode="none",
        max_correction_ratio=0.2,
        moe_backend="torch_index_add",
        enable_thinking=False,
        sandbox=None,
        use_cache=None,
        rerun_review=False,
        no_timestamp=True,
        preflight_only=True,
    )
    monkeypatch.setattr(run_evalscope_static_profile, "parse_args", lambda: args)
    monkeypatch.setattr(run_evalscope_static_profile, "validate_visible_gpus", lambda value: ["4"])
    monkeypatch.setattr(
        run_evalscope_static_profile,
        "file_sha256",
        lambda path: profile_hash if Path(path) == profile_path else channel_hash,
    )
    monkeypatch.setattr(
        run_evalscope_static_profile,
        "validate_static_profile_artifacts",
        lambda **kwargs: (profile, {}, torch.ones(1, 1, dtype=torch.long), {}),
    )
    monkeypatch.setattr(
        run_evalscope_static_profile,
        "source_tree_identity",
        lambda *args, **kwargs: {"commit": "test", "runtime_tree_sha256": "c" * 64},
    )
    monkeypatch.setattr(run_evalscope_static_profile, "evalscope_source_root", lambda: tmp_path)
    monkeypatch.setattr(
        run_evalscope_static_profile,
        "register_static_expert_profile_api",
        lambda: pytest.fail("preflight-only must not register or load the model API"),
    )

    assert run_evalscope_static_profile.main() == 0
    manifest = (work_dir / "evalscope_static_profile_manifest.json").read_text(encoding="utf-8")

    assert '"preflight_only": true' in manifest
    assert profile_hash in manifest
    assert channel_hash in manifest