from __future__ import annotations

import subprocess
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "create_result_dir.sh"


def test_create_result_dir_accepts_enp_25_and_50(tmp_path: Path) -> None:
    for pruning_ratio in ("25", "50"):
        result = subprocess.run(
            [
                "bash",
                str(SCRIPT),
                "--inference",
                "vllm",
                "--calibration",
                "WikiText128x2048",
                "--method",
                "ENP",
                "--pruning-ratio",
                pruning_ratio,
                "--timestamp",
                "202608081200",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            env={"RESULT_ROOT": str(tmp_path)},
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip().endswith(
            f"Qwen330BA3BInstruct_{pruning_ratio}_vllm_WikiText128x2048_quick9_ENP_202608081200_42"
        )


def test_create_result_dir_rejects_unregistered_pruning_ratio(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--inference",
            "vllm",
            "--calibration",
            "WikiText128x2048",
            "--method",
            "ENP",
            "--pruning-ratio",
            "40",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        env={"RESULT_ROOT": str(tmp_path)},
    )

    assert result.returncode == 2
    assert "25 or 50" in result.stderr