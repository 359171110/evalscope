from __future__ import annotations

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "run_one_model_harp_v2_two_full8.sh"


def _dry_run(tmp_path: Path, model: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["RESULT_ROOT"] = str(tmp_path)
    environment["TIMESTAMP"] = "202609032220"
    environment["DRY_RUN"] = "1"
    environment.pop("METHOD_TOKEN", None)
    environment.pop("HARPV2TWO_METHOD_TOKEN", None)
    environment.pop("HARP_ALLOCATOR", None)
    return subprocess.run(
        ["bash", str(SCRIPT), model, "3", "20210"],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )


def test_harp_v2_two_full8_launcher_uses_combo_two() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "harp_v2_two.score.lock" in text
    assert "combo_two" in text
    assert "harp_combo_two" in text
    assert "HARPv2two" in text


def test_harp_v2_two_full8_launcher_freezes_protocol(tmp_path: Path) -> None:
    result = _dry_run(tmp_path, "qwen3")
    assert result.returncode == 0, result.stderr
    assert "method=HARPv2two" in result.stdout
    assert "allocator=combo_two" in result.stdout
    assert "protocol=full8_v1" in result.stdout
    assert "Qwen330BA3BInstruct_50_vllm_CalibrationFree_full8_v1_HARPv2two_202609032220_42" in result.stdout
    assert "Qwen330BA3BInstruct_25_vllm_CalibrationFree_full8_v1_HARPv2two_202609032220_42" in result.stdout
    mixtral = _dry_run(tmp_path, "mixtral")
    assert mixtral.returncode != 0
