from __future__ import annotations

import subprocess
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_downstream_matrix.sh"


def _create_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    profile_root = tmp_path / "profiles"
    profile_root.mkdir()
    for name in (
        "aimer_50pct_per_layer.pt",
        "pure_pseudo_50pct_per_layer.pt",
        "enp_50pct_per_layer.pt",
        "reap_official_50pct_per_layer.pt",
        "route_tail_50pct_per_layer.pt",
        "tail_risk_50pct_per_layer.pt",
        "tenp_50pct_trapezoid.pt",
        "wick_kernel_50pct_per_layer.pt",
        "wick_kernel_merge_50pct_per_layer.pt",
        "wick_kernel_merge_plan.pt",
        "wick_pseudo_protect_50pct_per_layer.pt",
        "wick_pseudo_protect_merge_50pct_per_layer.pt",
        "wick_pseudo_protect_merge_plan.pt",
    ):
        (profile_root / name).write_bytes(name.encode("ascii"))
    channel_cache = tmp_path / "channel.pt"
    channel_cache.write_bytes(b"channel-cache")
    return profile_root, channel_cache


def test_downstream_matrix_assigns_ordered_methods_to_gpu_workers(tmp_path: Path) -> None:
    profile_root, channel_cache = _create_artifacts(tmp_path)
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--model-path",
            "/models/qwen3",
            "--model-id",
            "qwen3",
            "--model-family",
            "qwen3",
            "--pruning-ratio",
            "50pct",
            "--gpus",
            "4,5",
            "--datasets",
            "arc,gsm8k,ifeval",
            "--methods",
            "official_reap,route_tail_per_layer,tail_risk_per_layer",
            "--profile-root",
            str(profile_root),
            "--channel-cache",
            str(channel_cache),
            "--results-root",
            str(tmp_path / "results"),
            "--limit",
            "0.1",
            "--dataset-limits",
            '{"arc":400,"hellaswag":1000,"gsm8k":256,"math_500":40,"ifeval":541,"mmlu_pro":100}',
            "--sandbox",
            '{"enabled":true,"engine":"docker"}',
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "GPU 4 methods: official_reap tail_risk_per_layer" in result.stdout
    assert "GPU 5 methods: route_tail_per_layer" in result.stdout
    assert "--datasets arc gsm8k ifeval" in result.stdout
    assert "--limit 0.1" in result.stdout
    assert "--dataset-limits" in result.stdout
    assert "--sandbox" in result.stdout
    assert "enabled" in result.stdout
    assert "docker" in result.stdout
    for item in ("arc", "400", "hellaswag", "1000", "gsm8k", "256", "math_500", "40", "ifeval", "541", "mmlu_pro", "100"):
        assert item in result.stdout


def test_downstream_matrix_accepts_gpu_zero_and_rejects_gpu_six(tmp_path: Path) -> None:
    profile_root, channel_cache = _create_artifacts(tmp_path)
    accepted = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--model-path",
            "/models/qwen3",
            "--model-id",
            "qwen3",
            "--pruning-ratio",
            "50pct",
            "--gpus",
            "0",
            "--datasets",
            "arc",
            "--methods",
            "official_reap",
            "--profile-root",
            str(profile_root),
            "--channel-cache",
            str(channel_cache),
            "--results-root",
            str(tmp_path / "results"),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )

    rejected = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--model-path",
            "/models/qwen3",
            "--model-id",
            "qwen3",
            "--pruning-ratio",
            "50pct",
            "--gpus",
            "6",
            "--datasets",
            "arc",
            "--methods",
            "official_reap",
            "--profile-root",
            str(profile_root),
            "--channel-cache",
            str(channel_cache),
            "--results-root",
            str(tmp_path / "results"),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )

    assert accepted.returncode == 0, accepted.stderr
    assert "GPU 0 methods: official_reap" in accepted.stdout
    assert rejected.returncode == 2
    assert "physical GPU 0-5" in rejected.stderr


def test_downstream_matrix_resolves_aimer_profile(tmp_path: Path) -> None:
    profile_root, channel_cache = _create_artifacts(tmp_path)
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--model-path",
            "/models/qwen3",
            "--model-id",
            "qwen3",
            "--pruning-ratio",
            "50pct",
            "--gpus",
            "0",
            "--datasets",
            "math_500",
            "--methods",
            "aimer",
            "--profile-root",
            str(profile_root),
            "--channel-cache",
            str(channel_cache),
            "--results-root",
            str(tmp_path / "results"),
            "--dataset-limits",
            '{"math_500":20}',
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "aimer_50pct_per_layer.pt" in result.stdout
    assert "--dataset-limits" in result.stdout
    assert "math_500" in result.stdout
    assert "20" in result.stdout


def test_downstream_matrix_resolves_pure_pseudo_profile(tmp_path: Path) -> None:
    profile_root, channel_cache = _create_artifacts(tmp_path)
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--model-path",
            "/models/qwen3",
            "--model-id",
            "qwen3-pure-pseudo",
            "--pruning-ratio",
            "50pct",
            "--gpus",
            "0",
            "--datasets",
            "arc",
            "--methods",
            "pure_pseudo",
            "--profile-root",
            str(profile_root),
            "--channel-cache",
            str(channel_cache),
            "--results-root",
            str(tmp_path / "results"),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "GPU 0 methods: pure_pseudo" in result.stdout
    assert "pure_pseudo_50pct_per_layer.pt" in result.stdout


def test_downstream_matrix_resolves_enp_tenp_with_4096_generation(tmp_path: Path) -> None:
    profile_root, channel_cache = _create_artifacts(tmp_path)
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--model-path",
            "/models/qwen3",
            "--model-id",
            "qwen3-tenp",
            "--pruning-ratio",
            "50pct",
            "--gpus",
            "4,5",
            "--datasets",
            "math_500",
            "--methods",
            "enp,tenp",
            "--profile-root",
            str(profile_root),
            "--channel-cache",
            str(channel_cache),
            "--results-root",
            str(tmp_path / "results"),
            "--dataset-limits",
            '{"math_500":20}',
            "--generation-config",
            '{"max_tokens":4096}',
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "GPU 4 methods: enp" in result.stdout
    assert "GPU 5 methods: tenp" in result.stdout
    assert "enp_50pct_per_layer.pt" in result.stdout
    assert "tenp_50pct_trapezoid.pt" in result.stdout
    assert "max_tokens" in result.stdout
    assert "4096" in result.stdout
    assert "math_500" in result.stdout
    assert "20" in result.stdout


def test_downstream_matrix_resolves_wick_merge_plan(tmp_path: Path) -> None:
    profile_root, channel_cache = _create_artifacts(tmp_path)
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--model-path",
            "/models/qwen3",
            "--model-id",
            "qwen3-wick",
            "--pruning-ratio",
            "50pct",
            "--gpus",
            "0",
            "--datasets",
            "arc",
            "--methods",
            "wick_kernel,wick_kernel_merge",
            "--profile-root",
            str(profile_root),
            "--channel-cache",
            str(channel_cache),
            "--results-root",
            str(tmp_path / "results"),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "wick_kernel_50pct_per_layer.pt" in result.stdout
    assert "wick_kernel_merge_50pct_per_layer.pt" in result.stdout
    assert "--merge-plan" in result.stdout
    assert "wick_kernel_merge_plan.pt" in result.stdout
    assert "--expected-merge-plan-file-sha256" in result.stdout