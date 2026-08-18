from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a uniformly channel-pruned Qwen3 MoE HF checkpoint.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--channel-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--retained-channels", type=int, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expert_indices(name: str) -> tuple[int, int] | None:
    parts = name.split(".")
    if len(parts) < 7 or parts[0] != "model" or parts[1] != "layers" or parts[3:5] != ["mlp", "experts"]:
        return None
    return int(parts[2]), int(parts[5])


def prune_tensor(name: str, tensor: torch.Tensor, retained: torch.Tensor) -> torch.Tensor:
    if name.endswith(("gate_proj.weight", "up_proj.weight")):
        return tensor.index_select(0, retained)
    if name.endswith("down_proj.weight"):
        return tensor.index_select(1, retained)
    return tensor


def main() -> int:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_path = args.channel_cache.expanduser().resolve()
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    retained_by_expert = {
        (int(layer_id), expert_id): row[: args.retained_channels].to(torch.long)
        for layer_id, table in cache["table"].items()
        for expert_id, row in enumerate(table["ranked_indices"])
    }
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    source_moe_intermediate_size = int(config["moe_intermediate_size"])
    if source_moe_intermediate_size <= args.retained_channels:
        raise ValueError("retained channels must be smaller than the source MoE intermediate size.")

    index_path = model_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    total_size = 0
    for shard_name in sorted(set(weight_map.values())):
        tensors = {}
        with safe_open(model_path / shard_name, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                tensor = handle.get_tensor(name)
                ids = expert_indices(name)
                if ids is not None:
                    tensor = prune_tensor(name, tensor, retained_by_expert[ids])
                tensors[name] = tensor.contiguous()
                total_size += tensor.numel() * tensor.element_size()
        save_file(tensors, output_dir / shard_name, metadata={"format": "pt"})
        print(f"Exported {shard_name}", flush=True)

    for source in model_path.iterdir():
        if source.name.startswith("model-") and source.suffix == ".safetensors":
            continue
        if source.name in {"config.json", "model.safetensors.index.json"}:
            continue
        target = output_dir / source.name
        if source.is_file():
            shutil.copy2(source, target)
        elif source.is_dir():
            shutil.copytree(source, target)

    config["moe_intermediate_size"] = args.retained_channels
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    index.setdefault("metadata", {})["total_size"] = total_size
    (output_dir / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    manifest = {
        "source_model": str(model_path),
        "source_config_sha256": file_sha256(model_path / "config.json"),
        "source_weight_index_sha256": file_sha256(index_path),
        "channel_cache": str(cache_path),
        "channel_cache_sha256": file_sha256(cache_path),
        "retained_channels": args.retained_channels,
        "source_moe_intermediate_size": source_moe_intermediate_size,
        "exported_weight_bytes": total_size,
    }
    (output_dir / "pruning_export_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())