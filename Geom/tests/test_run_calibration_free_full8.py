from __future__ import annotations

import os
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "run_calibration_free_full8.sh"


def test_full8_launcher_freezes_four_model_50pct_protocol(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["RESULT_ROOT"] = str(tmp_path)
    environment["TIMESTAMP"] = "202608211430"
    result = subprocess.run(
        ["bash", str(SCRIPT), "all", "dry-run"],
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert "calibration=CalibrationFree" in result.stdout
    assert "protocol=full8_v1" in result.stdout
    assert "method=Geom" in result.stdout
    assert "ratio=50" in result.stdout
    assert "retained_channels=384" in result.stdout
    assert "retained_channels=352" in result.stdout
    assert "retained_channels=256" in result.stdout
    assert "retained_channels=704" in result.stdout
    assert "Qwen330BA3BInstruct_50_vllm_CalibrationFree_full8_v1_Geom_202608211430_42" in result.stdout
    assert "Gemma4-26B-A4B_50_vllm_CalibrationFree_full8_v1_Geom_202608211430_42" in result.stdout
    assert "Qwen3.6-35B-A3B_50_vllm_CalibrationFree_full8_v1_Geom_202608211430_42" in result.stdout
    assert "DeepSeek-V2-Lite-Chat_50_vllm_CalibrationFree_full8_v1_Geom_202608211430_42" in result.stdout


def test_full8_launcher_freezes_four_model_25pct_protocol(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["RESULT_ROOT"] = str(tmp_path)
    environment["TIMESTAMP"] = "202608211730"
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
    assert "retained_channels=1056" in result.stdout
    assert "Qwen330BA3BInstruct_25_vllm_CalibrationFree_full8_v1_Geom_202608211730_42" in result.stdout
    assert "Gemma4-26B-A4B_25_vllm_CalibrationFree_full8_v1_Geom_202608211730_42" in result.stdout
    assert "Qwen3.6-35B-A3B_25_vllm_CalibrationFree_full8_v1_Geom_202608211730_42" in result.stdout
    assert "DeepSeek-V2-Lite-Chat_25_vllm_CalibrationFree_full8_v1_Geom_202608211730_42" in result.stdout
