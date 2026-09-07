from __future__ import annotations

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "run_one_model_harp_v2_full8.sh"


def _dry_run(tmp_path: Path, model: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["RESULT_ROOT"] = str(tmp_path)
    environment["TIMESTAMP"] = "202609031600"
    environment["DRY_RUN"] = "1"
    environment.pop("METHOD_TOKEN", None)
    environment.pop("HARPV2_METHOD_TOKEN", None)
    environment.pop("HARP_ALLOCATOR", None)
    return subprocess.run(
        ["bash", str(SCRIPT), model, "3", "20110"],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )


def test_harp_v2_full8_launcher_uses_combo_and_flock() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "harp_v2.score.lock" in text
    assert "flock 8" in text
    assert "--allocator" in text
    assert "harp_combo" in text
    assert "HARPv2" in text


def test_harp_v2_full8_launcher_freezes_protocol(tmp_path: Path) -> None:
    result = _dry_run(tmp_path, "qwen3")
    assert result.returncode == 0, result.stderr
    assert "method=HARPv2" in result.stdout
    assert "allocator=combo" in result.stdout
    assert "protocol=full8_v1" in result.stdout
    assert "Qwen330BA3BInstruct_50_vllm_CalibrationFree_full8_v1_HARPv2_202609031600_42" in result.stdout
    assert "Qwen330BA3BInstruct_25_vllm_CalibrationFree_full8_v1_HARPv2_202609031600_42" in result.stdout
    gemma = _dry_run(tmp_path, "gemma4")
    assert "Gemma4-26B-A4B_50_vllm_CalibrationFree_full8_v1_HARPv2_202609031600_42" in gemma.stdout
    mixtral = _dry_run(tmp_path, "mixtral")
    assert mixtral.returncode != 0
