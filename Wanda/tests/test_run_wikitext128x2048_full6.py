from __future__ import annotations

import os
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "run_wikitext128x2048_full6.sh"


def test_wikitext_full6_launcher_freezes_three_model_50pct_protocol(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["RESULT_ROOT"] = str(tmp_path)
    environment["TIMESTAMP"] = "202608181430"
    result = subprocess.run(
        ["bash", str(SCRIPT), "all", "dry-run"],
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert "calibration=WikiText128x2048" in result.stdout
    assert "protocol=full6_v1" in result.stdout
    assert "method=Wanda" in result.stdout
    assert "ratio=50" in result.stdout
    assert "retained_channels=384" in result.stdout
    assert "retained_channels=352" in result.stdout
    assert "retained_channels=256" in result.stdout
    assert "Qwen330BA3BInstruct_50_vllm_WikiText128x2048_full6_v1_Wanda_202608181430_42" in result.stdout
    assert "Gemma4-26B-A4B_50_vllm_WikiText128x2048_full6_v1_Wanda_202608181430_42" in result.stdout
    assert "Qwen3.6-35B-A3B_50_vllm_WikiText128x2048_full6_v1_Wanda_202608181430_42" in result.stdout


def test_wikitext_full6_launcher_freezes_three_model_25pct_protocol(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["RESULT_ROOT"] = str(tmp_path)
    environment["TIMESTAMP"] = "202608181730"
    environment["RATIO"] = "25"
    result = subprocess.run(
        ["bash", str(SCRIPT), "all", "dry-run"],
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert "ratio=25" in result.stdout
    assert "retained_channels=576" in result.stdout
    assert "retained_channels=512" in result.stdout
    assert "retained_channels=384" in result.stdout
    assert "Qwen330BA3BInstruct_25_vllm_WikiText128x2048_full6_v1_Wanda_202608181730_42" in result.stdout
    assert "Gemma4-26B-A4B_25_vllm_WikiText128x2048_full6_v1_Wanda_202608181730_42" in result.stdout
    assert "Qwen3.6-35B-A3B_25_vllm_WikiText128x2048_full6_v1_Wanda_202608181730_42" in result.stdout
