from __future__ import annotations

import os
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_aimer_channel_quick9_parallel.sh"


def test_aimer_channel_launcher_uses_quick9_protocol_and_4096_tokens(tmp_path: Path) -> None:
    profile = tmp_path / "profile.pt"
    channel = tmp_path / "channel.pt"
    profile.write_bytes(b"profile")
    channel.write_bytes(b"channel")
    environment = dict(
        os.environ,
        GPUS_CSV="4,5",
        DRY_RUN="true",
        PROFILE=str(profile),
        CHANNEL_CACHE=str(channel),
    )

    result = subprocess.run(["bash", str(SCRIPT)], capture_output=True, text=True, env=environment)

    assert result.returncode == 0, result.stderr
    assert "GPU 4 dataset: arc limit: 300 max_tokens: 4096" in result.stdout
    assert "GPU 5 dataset: math_500 limit: 20 max_tokens: 4096" in result.stdout


def test_aimer_channel_launcher_rejects_disallowed_physical_gpu(tmp_path: Path) -> None:
    profile = tmp_path / "profile.pt"
    channel = tmp_path / "channel.pt"
    profile.write_bytes(b"profile")
    channel.write_bytes(b"channel")
    environment = dict(
        os.environ,
        GPUS_CSV="0,4",
        DRY_RUN="true",
        PROFILE=str(profile),
        CHANNEL_CACHE=str(channel),
    )

    result = subprocess.run(["bash", str(SCRIPT)], capture_output=True, text=True, env=environment)

    assert result.returncode == 2
    assert "one of 4,5,6,7" in result.stderr


def test_aimer_channel_launcher_rejects_duplicate_gpu(tmp_path: Path) -> None:
    profile = tmp_path / "profile.pt"
    channel = tmp_path / "channel.pt"
    profile.write_bytes(b"profile")
    channel.write_bytes(b"channel")
    environment = dict(
        os.environ,
        GPUS_CSV="4,4",
        DRY_RUN="true",
        PROFILE=str(profile),
        CHANNEL_CACHE=str(channel),
    )

    result = subprocess.run(["bash", str(SCRIPT)], capture_output=True, text=True, env=environment)

    assert result.returncode == 2
    assert "duplicate" in result.stderr