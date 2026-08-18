from __future__ import annotations

import os
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[3] / "TENP" / "run_gemma4_enp_wikitext128x2048.sh"


def test_gemma4_enp_launcher_freezes_full6_and_aligned_widths(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["RESULT_ROOT"] = str(tmp_path)
    environment["TIMESTAMP"] = "202608182200"
    result = subprocess.run(
        ["bash", str(SCRIPT), "dry-run"],
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert "target_model=Gemma4-26B-A4B" in result.stdout
    assert "calibration=WikiText128x2048" in result.stdout
    assert "protocol=full6_v1" in result.stdout
    assert "method=ENP" in result.stdout
    assert "enp_25pct_per_layer.pt; retained channels: 512" in result.stdout
    assert "enp_50pct_per_layer.pt; retained channels: 384" in result.stdout
    assert "source_expert_width: 704" in result.stdout
    assert "channel_block_size: 64" in result.stdout
    assert "Gemma4-26B-A4B_25_vllm_WikiText128x2048_full6_v1_ENP_202608182200_42" in result.stdout
    assert "Gemma4-26B-A4B_50_vllm_WikiText128x2048_full6_v1_ENP_202608182200_42" in result.stdout
    assert "Dense: skipped" in result.stdout
