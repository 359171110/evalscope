from __future__ import annotations

import os
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_aimer_channel_recovery.sh"


def _environment(tmp_path: Path, gpus: str) -> dict[str, str]:
    profile = tmp_path / "profile.pt"
    channel = tmp_path / "channel.pt"
    profile.write_bytes(b"profile")
    channel.write_bytes(b"channel")
    return dict(
        os.environ,
        GPUS_CSV=gpus,
        DRY_RUN="true",
        PROFILE=str(profile),
        CHANNEL_CACHE=str(channel),
        RESULTS_ROOT=str(tmp_path / "results"),
    )


def test_recovery_launcher_assigns_resilient_protocol_queues(tmp_path: Path) -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        env=_environment(tmp_path, "4,5"),
    )

    assert result.returncode == 0, result.stderr
    assert "GPU 4 mmlu subject: abstract_algebra limit: 10 max_tokens: 4096" in result.stdout
    assert "GPU 5 gsm8k limit: 128 max_tokens: 4096" in result.stdout
    assert "GPU 5 math_500 subset: Level 1 limit: 20 max_tokens: 4096" in result.stdout
    assert "GPU 5 mmlu subject: logical_fallacies limit: 10 max_tokens: 4096" in result.stdout


def test_recovery_launcher_rejects_disallowed_physical_gpu(tmp_path: Path) -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        env=_environment(tmp_path, "3,4"),
    )

    assert result.returncode == 2
    assert "one of 4,5,6,7" in result.stderr