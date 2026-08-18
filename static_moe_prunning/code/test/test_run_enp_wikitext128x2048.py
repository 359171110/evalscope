from __future__ import annotations

import os
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[3] / "TENP" / "run_enp_wikitext128x2048.sh"


def test_enp_wikitext_launcher_freezes_both_ratios_and_quick9_protocol(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["RESULT_ROOT"] = str(tmp_path)
    environment["TIMESTAMP"] = "202608081200"
    result = subprocess.run(
        ["bash", str(SCRIPT), "dry-run"],
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert "enp_25pct_per_layer.pt; retained channels: 576" in result.stdout
    assert "enp_50pct_per_layer.pt; retained channels: 384" in result.stdout
    assert "Qwen330BA3BInstruct_25_vllm_WikiText128x2048_quick9_ENP_202608081200_42" in result.stdout
    assert "Qwen330BA3BInstruct_50_vllm_WikiText128x2048_quick9_ENP_202608081200_42" in result.stdout
    assert "MATH-500 5x20" in result.stdout
    assert "MATH-500 max_tokens: 4096" in result.stdout
    assert "Dense: skipped" in result.stdout