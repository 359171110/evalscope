from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from NAPS_v2.build_naps_v2_artifacts import file_sha256, load_weight_map
from NAPS_v2.model_adapter import PurePseudoModelAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a small materialized CHANNEL-Puzzle validation checkpoint.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--retained-width", type=int, default=384)
    parser.add_argument("--effective-width", type=int, default=416)
    return parser.parse_args()


def _pad_rows(tensor: torch.Tensor, width: int) -> torch.Tensor:
    output = torch.zeros((width, tensor.shape[1]), dtype=tensor.dtype)
    output[: tensor.shape[0]] = tensor
    return output


def _pad_columns(tensor: torch.Tensor, width: int) -> torch.Tensor:
    output = torch.zeros((tensor.shape[0], width), dtype=tensor.dtype)
    output[:, : tensor.shape[1]] = tensor
    return output


def _layer_id(name: str) -> int:
    return int(name.split(".layers.", 1)[1].split(".", 1)[0])


def _table(rankings: dict[str, Any], layer_id: int) -> dict[str, Any]:
    table = rankings["table"].get(layer_id, rankings["table"].get(str(layer_id)))
    if table is None:
        raise KeyError(f"Missing CHANNEL ranking layer {layer_id}")
    return table


def main() -> int:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    rankings_path = args.rankings.expanduser().resolve()
    validation_dir = args.validation_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rankings = torch.load(rankings_path, map_location="cpu", weights_only=True)
    diagnostics = json.loads((validation_dir / "diagnostics.json").read_text(encoding="utf-8"))
    materialized = torch.load(
        validation_dir / "materialized_width416_pairs.pt", map_location="cpu", weights_only=True
    )
    if Path(rankings["model_path"]).resolve() != model_path or diagnostics["model_path"] != str(model_path):
        raise ValueError("Model, rankings, and validation paths do not match")
    if diagnostics["rankings_sha256"] != file_sha256(rankings_path):
        raise ValueError("Validation and rankings provenance do not match")
    if diagnostics["retained_width"] != args.retained_width or diagnostics["effective_width"] != args.effective_width:
        raise ValueError("Requested export widths do not match the validation artifact")
    accepted: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
    accepted_pairs = []
    for layer_id, pairs in materialized.items():
        for pair_name, pair in pairs.items():
            if not pair["accepted"]:
                continue
            left_id = int(pair["left_expert_id"])
            right_id = int(pair["right_expert_id"])
            accepted[(int(layer_id), left_id)] = pair["left"]
            accepted[(int(layer_id), right_id)] = pair["right"]
            accepted_pairs.append({"layer_id": int(layer_id), "pair": pair_name, "experts": [left_id, right_id]})
    weight_map = load_weight_map(model_path)
    adapter = PurePseudoModelAdapter.from_checkpoint(model_path, weight_map)
    if adapter.model_family != "gemma4" or adapter.expert_gate_template is not None:
        raise ValueError("This validation exporter currently supports packed Gemma4 experts only")
    total_size = 0
    changed_tensors = 0
    for shard_name in sorted(set(weight_map.values())):
        tensors = {}
        with safe_open(model_path / shard_name, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                source = handle.get_tensor(name)
                changed = source
                if ".layers." in name and name == adapter.expert_gate_up_name(_layer_id(name)):
                    layer_id = _layer_id(name)
                    gate, up = source.split(adapter.intermediate_size, dim=1)
                    outputs = []
                    ranking_table = _table(rankings, layer_id)
                    for expert_id in range(adapter.num_experts):
                        puzzle = accepted.get((layer_id, expert_id))
                        if puzzle is not None:
                            selected_gate = puzzle["gate"].to(source.dtype)
                            selected_up = puzzle["up"].to(source.dtype)
                        else:
                            ids = ranking_table["ranked_indices"][expert_id, :args.retained_width].to(torch.long)
                            selected_gate = _pad_rows(gate[expert_id].index_select(0, ids), args.effective_width)
                            selected_up = _pad_rows(up[expert_id].index_select(0, ids), args.effective_width)
                        outputs.append(torch.cat((selected_gate, selected_up), dim=0))
                    changed = torch.stack(outputs).contiguous()
                    changed_tensors += 1
                elif ".layers." in name and name == adapter.expert_down_name(_layer_id(name)):
                    layer_id = _layer_id(name)
                    outputs = []
                    ranking_table = _table(rankings, layer_id)
                    for expert_id in range(adapter.num_experts):
                        puzzle = accepted.get((layer_id, expert_id))
                        if puzzle is not None:
                            selected_down = puzzle["down"].to(source.dtype)
                        else:
                            ids = ranking_table["ranked_indices"][expert_id, :args.retained_width].to(torch.long)
                            selected_down = _pad_columns(
                                source[expert_id].index_select(1, ids), args.effective_width
                            )
                        outputs.append(selected_down)
                    changed = torch.stack(outputs).contiguous()
                    changed_tensors += 1
                tensors[name] = changed.contiguous()
                total_size += changed.numel() * changed.element_size()
        save_file(tensors, output_dir / shard_name, metadata={"format": "pt"})
    expected_changed = 2 * adapter.num_layers
    if changed_tensors != expected_changed:
        raise RuntimeError(f"Changed {changed_tensors} expert tensors, expected {expected_changed}")
    for source in model_path.iterdir():
        if source.suffix == ".safetensors" or source.name in {
            ".git", "config.json", "model.safetensors.index.json"
        } or source.name.startswith("."):
            continue
        target = output_dir / source.name
        shutil.copytree(source, target) if source.is_dir() else shutil.copy2(source, target)
    source_config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    source_config.get("text_config", source_config)["moe_intermediate_size"] = args.effective_width
    (output_dir / "config.json").write_text(
        json.dumps(source_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    index = json.loads((model_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    index.setdefault("metadata", {})["total_size"] = total_size
    (output_dir / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "method": "channel_puzzle_materialized_small_validation",
        "source_model": str(model_path),
        "rankings_path": str(rankings_path),
        "rankings_sha256": file_sha256(rankings_path),
        "validation_dir": str(validation_dir),
        "diagnostics_sha256": file_sha256(validation_dir / "diagnostics.json"),
        "materialized_pairs_sha256": file_sha256(validation_dir / "materialized_width416_pairs.pt"),
        "retained_width": args.retained_width,
        "physical_width": args.effective_width,
        "default_expert": "channel_384_zero_padded_to_416",
        "accepted_expert_count": len(accepted),
        "accepted_pairs": accepted_pairs,
        "pair_storage_equivalent_width": 2 * args.retained_width,
        "benchmark_metrics_used": False,
    }
    (output_dir / "pruning_export_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)
    print(output_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())