"""Build HARP combo_two (low/high only) profiles for the five eval families."""

from __future__ import annotations

from pathlib import Path

from HARP.build_harp_artifacts import build_profile
from HARP.search_combo_five import MODELS

ROOT = Path("/data/xinpeigao/evalscope_results/_artifacts")
CACHE_ROOT = ROOT / "harp"
OUT_ROOT = ROOT / "harp_combo_two"


def main() -> int:
    """Reuse v1 ranking caches; do not overwrite HARP-v2 combo profiles."""

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
            if int(high_width) - int(low_width) != 128:
                raise ValueError(f"{name} {ratio}% high-low is not 128.")
            profile = dest / f"harp_combo_two_{ratio}pct.pt"
            print(f"BUILD combo_two {name} {ratio}% -> {profile}")
            build_profile(
                model_path,
                cache,
                profile,
                int(budget_width),
                int(low_width),
                int(high_width),
                allocator="combo_two",
                gamma=2.0,
                min_fraction=0.15,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
