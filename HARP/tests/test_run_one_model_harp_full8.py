from __future__ import annotations

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "run_one_model_harp_full8.sh"


def _dry_run(tmp_path: Path, model: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["RESULT_ROOT"] = str(tmp_path)
    environment["TIMESTAMP"] = "202609031000"
    environment["DRY_RUN"] = "1"
    environment.pop("METHOD_TOKEN", None)
    environment.pop("HARP_METHOD_TOKEN", None)
    return subprocess.run(
        ["bash", str(SCRIPT), model, "3", "20010"],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )


def test_harp_full8_launcher_serializes_scoring_with_flock() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "harp.score.lock" in text
    assert "flock 8" in text
    assert "HARP.build_harp_artifacts" in text
    assert "--low-width" in text
    assert "--high-width" in text


def test_harp_full8_launcher_freezes_qwen3_50_then_25_protocol(tmp_path: Path) -> None:
    result = _dry_run(tmp_path, "qwen3")
    assert result.returncode == 0, result.stderr
    assert "calibration=CalibrationFree" in result.stdout
    assert "protocol=full8_v1" in result.stdout
    assert "method=HARP" in result.stdout
    assert "low_width=320" in result.stdout
    assert "budget_width=384" in result.stdout
    assert "high_width=448" in result.stdout
    assert "low_width=512" in result.stdout
    assert "budget_width=576" in result.stdout
    assert "high_width=640" in result.stdout
    assert "Qwen330BA3BInstruct_50_vllm_CalibrationFree_full8_v1_HARP_202609031000_42" in result.stdout
    assert "Qwen330BA3BInstruct_25_vllm_CalibrationFree_full8_v1_HARP_202609031000_42" in result.stdout


def test_harp_full8_launcher_freezes_remaining_model_widths(tmp_path: Path) -> None:
    gemma = _dry_run(tmp_path, "gemma4")
    qwen36 = _dry_run(tmp_path, "qwen36")
    deepseek = _dry_run(tmp_path, "deepseek")
    olmoe = _dry_run(tmp_path, "olmoe")
    assert gemma.returncode == 0, gemma.stderr
    assert qwen36.returncode == 0, qwen36.stderr
    assert deepseek.returncode == 0, deepseek.stderr
    assert olmoe.returncode == 0, olmoe.stderr
    assert "low_width=288" in gemma.stdout
    assert "high_width=416" in gemma.stdout
    assert "Gemma4-26B-A4B_50_vllm_CalibrationFree_full8_v1_HARP_202609031000_42" in gemma.stdout
    assert "low_width=192" in qwen36.stdout
    assert "high_width=320" in qwen36.stdout
    assert "Qwen3.6-35B-A3B_50_vllm_CalibrationFree_full8_v1_HARP_202609031000_42" in qwen36.stdout
    assert "low_width=640" in deepseek.stdout
    assert "high_width=768" in deepseek.stdout
    assert "DeepSeek-V2-Lite-Chat_50_vllm_CalibrationFree_full8_v1_HARP_202609031000_42" in deepseek.stdout
    assert "low_width=448" in olmoe.stdout
    assert "high_width=576" in olmoe.stdout
    assert "OLMoE-1B-7B-Instruct_50_vllm_CalibrationFree_full8_v1_HARP_202609031000_42" in olmoe.stdout
    mixtral = _dry_run(tmp_path, "mixtral")
    assert mixtral.returncode != 0
    assert "Unknown model 'mixtral'" in mixtral.stderr
