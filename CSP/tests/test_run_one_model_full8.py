from __future__ import annotations

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "run_one_model_full8.sh"


def _dry_run(tmp_path: Path, model: str, extra: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["RESULT_ROOT"] = str(tmp_path)
    environment["TIMESTAMP"] = "202608281730"
    environment["DRY_RUN"] = "1"
    if extra:
        environment.update(extra)
    return subprocess.run(
        ["bash", str(SCRIPT), model, "3", "19732"],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )


def test_full8_launcher_serializes_scoring_with_flock() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "csp.score.lock" in text
    assert "flock 8" in text
    assert "WAIT score lock" in text


def test_full8_launcher_freezes_qwen3_50_then_25_protocol(tmp_path: Path) -> None:
    result = _dry_run(tmp_path, "qwen3")
    assert result.returncode == 0, result.stderr
    assert "calibration=CalibrationFree" in result.stdout
    assert "protocol=full8_v1" in result.stdout
    assert "method=CSP" in result.stdout
    assert "ratio=50" in result.stdout
    assert "retained_channels=384" in result.stdout
    assert "ratio=25" in result.stdout
    assert "retained_channels=576" in result.stdout
    assert "Qwen330BA3BInstruct_50_vllm_CalibrationFree_full8_v1_CSP_202608281730_42" in result.stdout
    assert "Qwen330BA3BInstruct_25_vllm_CalibrationFree_full8_v1_CSP_202608281730_42" in result.stdout


def test_full8_launcher_freezes_remaining_model_widths(tmp_path: Path) -> None:
    gemma = _dry_run(tmp_path, "gemma4")
    qwen36 = _dry_run(tmp_path, "qwen36")
    deepseek = _dry_run(tmp_path, "deepseek")
    assert gemma.returncode == 0, gemma.stderr
    assert qwen36.returncode == 0, qwen36.stderr
    assert deepseek.returncode == 0, deepseek.stderr
    assert "retained_channels=352" in gemma.stdout
    assert "retained_channels=512" in gemma.stdout
    assert "Gemma4-26B-A4B_50_vllm_CalibrationFree_full8_v1_CSP_202608281730_42" in gemma.stdout
    assert "retained_channels=256" in qwen36.stdout
    assert "retained_channels=384" in qwen36.stdout
    assert "Qwen3.6-35B-A3B_50_vllm_CalibrationFree_full8_v1_CSP_202608281730_42" in qwen36.stdout
    assert "retained_channels=704" in deepseek.stdout
    assert "retained_channels=1056" in deepseek.stdout
    assert "DeepSeek-V2-Lite-Chat_50_vllm_CalibrationFree_full8_v1_CSP_202608281730_42" in deepseek.stdout
    olmoe = _dry_run(tmp_path, "olmoe")
    assert olmoe.returncode == 0, olmoe.stderr
    assert "retained_channels=512" in olmoe.stdout
    assert "retained_channels=768" in olmoe.stdout
    assert "OLMoE-1B-7B-Instruct_50_vllm_CalibrationFree_full8_v1_CSP_202608281730_42" in olmoe.stdout
    mixtral = _dry_run(tmp_path, "mixtral")
    assert mixtral.returncode == 0, mixtral.stderr
    assert "retained_channels=7168" in mixtral.stdout
    assert "retained_channels=10752" in mixtral.stdout
    assert "Mixtral-8x7B-Instruct_50_vllm_CalibrationFree_full8_v1_CSP_202608281730_42" in mixtral.stdout
