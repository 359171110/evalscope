from __future__ import annotations

import os
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_aimer_quick9_parallel.sh"


def test_aimer_quick9_parallel_round_robins_on_allowed_gpus() -> None:
    environment = dict(os.environ, GPUS_CSV="4,5", DRY_RUN="true")
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert "GPU 4 dataset: arc" in result.stdout
    assert "GPU 5 dataset: hellaswag" in result.stdout
    assert "GPU 4 dataset: mmlu" in result.stdout
    assert "GPU 5 dataset: winogrande" in result.stdout
    assert "GPU 4 dataset: gsm8k" in result.stdout
    assert "GPU 5 dataset: math_500" in result.stdout


def test_aimer_quick9_parallel_rejects_disallowed_gpu() -> None:
    environment = dict(os.environ, GPUS_CSV="3,4", DRY_RUN="true")
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 2
    assert "physical GPUs 4-7" in result.stderr