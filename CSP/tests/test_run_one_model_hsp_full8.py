from __future__ import annotations

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "run_one_model_hsp_full8.sh"


def _dry_run(tmp_path: Path, model: str, extra: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["RESULT_ROOT"] = str(tmp_path)
    environment["TIMESTAMP"] = "202608290037"
    environment["DRY_RUN"] = "1"
    environment.pop("METHOD_TOKEN", None)
    environment.pop("HSP_METHOD_TOKEN", None)
    if extra:
        environment.update(extra)
    return subprocess.run(
        ["bash", str(SCRIPT), model, "3", "19832"],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )


def test_hsp_full8_launcher_serializes_scoring_with_flock() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "hsp.score.lock" in text
    assert "flock 8" in text
    assert "--apply-input-scale never" in text
    assert "--heterogeneous-widths" in text


def test_hsp_full8_launcher_freezes_qwen3_50_then_25_protocol(tmp_path: Path) -> None:
    result = _dry_run(tmp_path, "qwen3")
    assert result.returncode == 0, result.stderr
    assert "calibration=CalibrationFree" in result.stdout
    assert "protocol=full8_v1" in result.stdout
    assert "method=HSP" in result.stdout
    assert "apply_input_scale=never" in result.stdout
    assert "heterogeneous_widths=320 384 448" in result.stdout
    assert "budget_width=384" in result.stdout
    assert "heterogeneous_widths=512 576 640" in result.stdout
    assert "budget_width=576" in result.stdout
    assert "Qwen330BA3BInstruct_50_vllm_CalibrationFree_full8_v1_HSP_202608290037_42" in result.stdout
    assert "Qwen330BA3BInstruct_25_vllm_CalibrationFree_full8_v1_HSP_202608290037_42" in result.stdout


def test_hsp_full8_launcher_freezes_remaining_model_widths(tmp_path: Path) -> None:
    gemma = _dry_run(tmp_path, "gemma4")
    qwen36 = _dry_run(tmp_path, "qwen36")
    deepseek = _dry_run(tmp_path, "deepseek")
    assert gemma.returncode == 0, gemma.stderr
    assert qwen36.returncode == 0, qwen36.stderr
    assert deepseek.returncode == 0, deepseek.stderr
    assert "heterogeneous_widths=288 352 416" in gemma.stdout
    assert "heterogeneous_widths=448 512 576" in gemma.stdout
    assert "Gemma4-26B-A4B_50_vllm_CalibrationFree_full8_v1_HSP_202608290037_42" in gemma.stdout
    assert "heterogeneous_widths=192 256 320" in qwen36.stdout
    assert "heterogeneous_widths=320 384 448" in qwen36.stdout
    assert "Qwen3.6-35B-A3B_50_vllm_CalibrationFree_full8_v1_HSP_202608290037_42" in qwen36.stdout
    assert "heterogeneous_widths=640 704 768" in deepseek.stdout
    assert "heterogeneous_widths=992 1056 1120" in deepseek.stdout
    assert "DeepSeek-V2-Lite-Chat_50_vllm_CalibrationFree_full8_v1_HSP_202608290037_42" in deepseek.stdout
    olmoe = _dry_run(tmp_path, "olmoe")
    assert olmoe.returncode == 0, olmoe.stderr
    assert "heterogeneous_widths=448 512 576" in olmoe.stdout
    assert "heterogeneous_widths=704 768 832" in olmoe.stdout
    assert "OLMoE-1B-7B-Instruct_50_vllm_CalibrationFree_full8_v1_HSP_202608290037_42" in olmoe.stdout
    mixtral = _dry_run(tmp_path, "mixtral")
    assert mixtral.returncode == 0, mixtral.stderr
    assert "heterogeneous_widths=7104 7168 7232" in mixtral.stdout
    assert "heterogeneous_widths=10688 10752 10816" in mixtral.stdout
    assert "Mixtral-8x7B-Instruct_50_vllm_CalibrationFree_full8_v1_HSP_202608290037_42" in mixtral.stdout
