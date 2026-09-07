"""Build HARP combo-search profiles for the five eval families (no export/eval)."""

from __future__ import annotations

from pathlib import Path

from HARP.build_harp_artifacts import build_profile

ROOT = Path("/data/xinpeigao/evalscope_results/_artifacts")
CACHE_ROOT = ROOT / "harp"
OUT_ROOT = ROOT / "harp_combo"

MODELS: dict[str, dict[str, object]] = {
    "qwen3": {
        "path": Path("/data/xinpeigao/models/Qwen3-30B-A3B-Instruct-2507"),
        50: (320, 384, 448),
        25: (512, 576, 640),
    },
    "gemma4": {
        "path": Path("/data/xinpeigao/models/gemma-4-26B-A4B-it"),
        50: (288, 352, 416),
        25: (448, 512, 576),
    },
    "qwen36": {
        "path": Path("/data/xinpeigao/models/Qwen3.6-35B-A3B"),
        50: (192, 256, 320),
        25: (320, 384, 448),
    },
    "deepseek": {
        "path": Path("/data/xinpeigao/models/DeepSeek-V2-Lite-Chat"),
        50: (640, 704, 768),
        25: (992, 1056, 1120),
    },
    "olmoe": {
        "path": Path(
            "/data1/xinpeigao/caches/huggingface/hub/"
            "models--allenai--OLMoE-1B-7B-0125-Instruct/snapshots/b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e"
        ),
        50: (448, 512, 576),
        25: (704, 768, 832),
    },
}


def main() -> int:
    """Reuse v1 ranking caches and write combo profiles next to them."""

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for name, spec in MODELS.items():
        model_path = Path(str(spec["path"]))
        cache = CACHE_ROOT / name / "harp_rankings.pt"
        if not cache.is_file():
            raise FileNotFoundError(f"missing HARP ranking cache: {cache}")
        dest = OUT_ROOT / name
        dest.mkdir(parents=True, exist_ok=True)
        for ratio in (50, 25):
            low_width, budget_width, high_width = spec[ratio]  # type: ignore[misc]
            profile = dest / f"harp_combo_{ratio}pct.pt"
            print(f"BUILD combo {name} {ratio}% -> {profile}")
            build_profile(
                model_path,
                cache,
                profile,
                int(budget_width),
                int(low_width),
                int(high_width),
                allocator="combo",
                gamma=2.0,
                min_fraction=0.15,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
