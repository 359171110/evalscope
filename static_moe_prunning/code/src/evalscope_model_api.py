from __future__ import annotations

import hashlib
import json
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import Any, Mapping

import torch

from .channel_runtime import ChannelTable, channel_table_from_payload
from .model_families import resolve_model_family
from .model_loading import load_supported_moe
from .model_structure import iter_moe_layer_bindings
from .static_expert_pruning import (
    StaticExpertRuntimeStats,
    apply_static_down_projection_merge_plan,
    patch_qwen3_moe_blocks_static_expert,
    profile_widths_by_layer,
    validate_static_profile_payload,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _profile_width_sha256(widths: torch.Tensor) -> str:
    return hashlib.sha256(
        widths.detach().cpu().contiguous().numpy().tobytes(order="C")
    ).hexdigest()


@lru_cache(maxsize=8)
def _checkpoint_identity(model_path: str) -> dict[str, object]:
    root = Path(model_path).expanduser().resolve()
    config_path = root / "config.json"
    index_path = root / "model.safetensors.index.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"checkpoint config does not exist: {config_path}")
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, Mapping):
            raise ValueError("checkpoint safetensors index is missing weight_map.")
        shard_names = sorted({str(value) for value in weight_map.values()})
        index_sha256 = file_sha256(index_path)
    else:
        shard_names = ["model.safetensors"]
        index_sha256 = None
    shard_sha256 = {}
    for shard_name in shard_names:
        shard_path = root / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(f"checkpoint shard does not exist: {shard_path}")
        shard_sha256[shard_name] = file_sha256(shard_path)
    return {
        "model_path": str(root),
        "config_sha256": file_sha256(config_path),
        "index_sha256": index_sha256,
        "shard_sha256": shard_sha256,
    }


def _validate_optional_checkpoint_identity(
    expected: object,
    *,
    model_path: str,
    artifact_name: str,
) -> dict[str, object] | None:
    if expected is None:
        return None
    if not isinstance(expected, Mapping):
        raise ValueError(f"{artifact_name} checkpoint_identity must be a mapping.")
    normalized = dict(expected)
    if normalized != _checkpoint_identity(model_path):
        raise ValueError(f"{artifact_name} checkpoint identity does not match the evaluation checkpoint.")
    return normalized


def _expert_count(experts) -> int:
    if hasattr(experts, "gate_up_proj"):
        return int(experts.gate_up_proj.shape[0])
    return len(experts)


def _expert_intermediate_size(experts) -> int:
    if hasattr(experts, "gate_up_proj"):
        return int(experts.gate_up_proj.shape[1] // 2)
    if len(experts) == 0:
        raise ValueError("MoE expert collection is empty.")
    return int(experts[0].gate_proj.weight.shape[0])


def _validate_channel_table(
    table_payload: Mapping[Any, Mapping[str, object]],
    *,
    expected_layers: set[int],
    expected_experts: int,
    expected_blocks: int,
) -> ChannelTable:
    actual_layers = {int(layer_id) for layer_id in table_payload}
    if actual_layers != expected_layers:
        raise ValueError(
            f"channel cache layers {sorted(actual_layers)} do not match profile layers "
            f"{sorted(expected_layers)}."
        )
    for raw_layer_id, values in table_payload.items():
        layer_id = int(raw_layer_id)
        ranked = values.get("ranked_indices")
        relative = values.get("block_relative_scores")
        coverage = values.get("block_coverage_scores")
        block_sizes = values.get("block_sizes")
        intermediate_size = int(values.get("intermediate_size", -1))
        if not all(isinstance(item, torch.Tensor) for item in (ranked, relative, coverage, block_sizes)):
            raise ValueError(f"channel layer {layer_id} contains non-tensor values.")
        if ranked.ndim != 2 or int(ranked.shape[0]) != expected_experts:
            raise ValueError(f"channel layer {layer_id} ranked_indices shape is invalid.")
        if ranked.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError(f"channel layer {layer_id} ranked_indices must be integer.")
        if intermediate_size != int(ranked.shape[1]):
            raise ValueError(f"channel layer {layer_id} intermediate size is inconsistent.")
        expected_indices = torch.arange(intermediate_size, dtype=ranked.dtype).view(1, -1)
        if not bool(torch.all(torch.sort(ranked.cpu(), dim=1).values == expected_indices)):
            raise ValueError(f"channel layer {layer_id} ranked_indices must be row permutations.")
        if block_sizes.ndim != 1 or block_sizes.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError(f"channel layer {layer_id} block_sizes are invalid.")
        if bool((block_sizes <= 0).any()) or int(block_sizes.sum().item()) != intermediate_size:
            raise ValueError(f"channel layer {layer_id} block_sizes do not cover the intermediate dimension.")
        num_blocks = int(block_sizes.numel())
        if num_blocks != expected_blocks:
            raise ValueError(
                f"channel layer {layer_id} block count {num_blocks} does not match profile num_blocks "
                f"{expected_blocks}."
            )
        for name, scores in (("relative", relative), ("coverage", coverage)):
            if scores.ndim != 2 or tuple(scores.shape) != (expected_experts, num_blocks):
                raise ValueError(f"channel layer {layer_id} {name} score shape is invalid.")
            if not bool(torch.isfinite(scores).all()):
                raise ValueError(f"channel layer {layer_id} {name} scores must be finite.")
    return channel_table_from_payload(table_payload)


def _validate_per_layer_budget(profile: Mapping[str, object], widths: torch.Tensor) -> None:
    if profile.get("allocation_scope") != "per_layer":
        return
    expected = widths.to(torch.long).sum(dim=1).tolist()
    target = profile.get("target_blocks_by_layer")
    actual = profile.get("actual_blocks_by_layer")
    if target != expected or actual != expected:
        raise ValueError(
            "per-layer block budget metadata must match profile_widths exactly."
        )


def validate_static_profile_artifacts(
    *,
    model_path: str,
    profile_path: str | Path,
    channel_cache_path: str | Path,
    expected_profile_file_sha256: str | None = None,
    expected_channel_file_sha256: str | None = None,
) -> tuple[dict, dict, torch.Tensor, ChannelTable]:
    """Validate frozen profile/channel artifacts before loading a large model."""

    model = str(Path(model_path).expanduser().resolve())
    profile_file = Path(profile_path).expanduser().resolve()
    channel_file = Path(channel_cache_path).expanduser().resolve()
    if not profile_file.is_file() or not channel_file.is_file():
        raise FileNotFoundError("static profile and channel cache files must exist.")
    profile_file_hash = file_sha256(profile_file)
    channel_file_hash = file_sha256(channel_file)
    if expected_profile_file_sha256 and profile_file_hash != expected_profile_file_sha256:
        raise ValueError("profile file SHA256 does not match expected_profile_file_sha256.")
    if expected_channel_file_sha256 and channel_file_hash != expected_channel_file_sha256:
        raise ValueError("channel file SHA256 does not match expected_channel_file_sha256.")
    profile = torch.load(profile_file, map_location="cpu", weights_only=True)
    channel = torch.load(channel_file, map_location="cpu", weights_only=True)
    raw_widths = profile.get("profile_widths")
    if not isinstance(raw_widths, torch.Tensor) or raw_widths.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise ValueError("profile_widths must use an integer dtype.")
    widths = validate_static_profile_payload(profile)
    if _profile_width_sha256(widths) != profile.get("profile_sha256"):
        raise ValueError("profile_sha256 does not match profile_widths.")
    _validate_per_layer_budget(profile, widths)
    expected_channel_hash = profile.get("cache_provenance", {}).get("channel", {}).get("sha256")
    if expected_channel_hash != channel_file_hash:
        raise ValueError("channel cache SHA256 does not match profile provenance.")
    if str(Path(str(profile.get("model_path", ""))).expanduser().resolve()) != model:
        raise ValueError("profile model_path does not match the evaluation checkpoint.")
    if str(Path(str(channel.get("model_path", ""))).expanduser().resolve()) != model:
        raise ValueError("channel cache model_path does not match the evaluation checkpoint.")
    channel_checkpoint = _validate_optional_checkpoint_identity(
        channel.get("checkpoint_identity"),
        model_path=model,
        artifact_name="channel cache",
    )
    profile_checkpoint = profile.get("wick", {}).get("checkpoint_identity")
    if bool(channel_checkpoint) != bool(profile_checkpoint):
        raise ValueError("WICK profile and channel checkpoint identities must either both be present or both absent.")
    if profile_checkpoint is not None and dict(profile_checkpoint) != channel_checkpoint:
        raise ValueError("WICK profile checkpoint identity does not match the channel artifact.")
    construction = profile.get("profile_construction", "calibrated")
    channel_sequence_length = int(channel.get("sequence_length", -1))
    if construction == "calibration_free":
        approved_purposes = {
            "runtime_topology_only",
            "aimer_weight_only_channel_ranking",
            "pure_pseudo_channel_ranking",
            "wick_weight_kernel_channel_ranking",
        }
        if (
            channel.get("split") != "not_applicable"
            or channel_sequence_length != 0
            or channel.get("purpose") not in approved_purposes
        ):
            raise ValueError(
                "calibration-free profiles require an approved calibration-free channel artifact."
            )
    elif channel.get("split") != "train" or channel_sequence_length <= 0:
        raise ValueError("channel cache must be a train-only artifact with a positive sequence_length.")
    calibration_provenance = profile.get("cache_provenance", {}).get("calibration", {})
    if calibration_provenance:
        if calibration_provenance.get("split") != "train":
            raise ValueError("profile calibration provenance must be train-only.")
        if int(calibration_provenance.get("sequence_length", -1)) != channel_sequence_length:
            raise ValueError("profile and channel cache sequence lengths do not match.")
    table_payload = channel.get("table")
    if not isinstance(table_payload, Mapping):
        raise ValueError("channel cache table is missing.")
    expected_layers = {int(layer_id) for layer_id in profile["layer_ids"]}
    table = _validate_channel_table(
        table_payload,
        expected_layers=expected_layers,
        expected_experts=int(widths.shape[1]),
        expected_blocks=int(profile["num_blocks"]),
    )
    for layer_id, channel_layer in table.items():
        if bool(((widths[profile["layer_ids"].index(layer_id)] < 0) | (widths[profile["layer_ids"].index(layer_id)] > channel_layer.block_sizes.numel())).any()):
            raise ValueError(f"profile widths exceed channel blocks at layer {layer_id}.")
    profile["profile_file_sha256"] = profile_file_hash
    channel["channel_file_sha256"] = channel_file_hash
    return profile, channel, widths, table


def validate_static_merge_plan_artifact(
    *,
    model_path: str,
    merge_plan_path: str | Path,
    profile: Mapping[str, object],
    channel: Mapping[str, object],
    widths: torch.Tensor,
    table: ChannelTable,
    expected_merge_plan_file_sha256: str | None = None,
) -> tuple[dict, str]:
    """Validate a frozen down-projection merge plan before loading a large model."""

    model = str(Path(model_path).expanduser().resolve())
    merge_file = Path(merge_plan_path).expanduser().resolve()
    if not merge_file.is_file():
        raise FileNotFoundError("static merge plan file must exist.")
    merge_hash = file_sha256(merge_file)
    if expected_merge_plan_file_sha256 and merge_hash != expected_merge_plan_file_sha256:
        raise ValueError("merge plan file SHA256 does not match expected_merge_plan_file_sha256.")
    expected_merge_hash = profile.get("cache_provenance", {}).get("merge_plan", {}).get("sha256")
    if expected_merge_hash != merge_hash:
        raise ValueError("merge plan SHA256 does not match profile provenance.")
    merge_plan = torch.load(merge_file, map_location="cpu", weights_only=True)
    if merge_plan.get("purpose") != "wick_down_projection_merge":
        raise ValueError("merge plan purpose is not approved.")
    if str(Path(str(merge_plan.get("model_path", ""))).expanduser().resolve()) != model:
        raise ValueError("merge plan model_path does not match the evaluation checkpoint.")
    channel_hash = channel.get("channel_file_sha256")
    if merge_plan.get("channel_file_sha256") != channel_hash:
        raise ValueError("merge plan channel SHA256 does not match the channel artifact.")
    channel_checkpoint = channel.get("checkpoint_identity")
    merge_checkpoint = merge_plan.get("checkpoint_identity")
    if bool(channel_checkpoint) != bool(merge_checkpoint):
        raise ValueError("merge plan and channel checkpoint identities must either both be present or both absent.")
    if merge_checkpoint is not None and dict(merge_checkpoint) != dict(channel_checkpoint):
        raise ValueError("merge plan checkpoint identity does not match the channel artifact.")
    raw_layers = merge_plan.get("layers")
    if not isinstance(raw_layers, Mapping):
        raise ValueError("merge plan layers are missing.")
    layer_plans = {int(layer_id): values for layer_id, values in raw_layers.items()}
    layer_ids = [int(layer_id) for layer_id in profile["layer_ids"]]
    if set(layer_plans) != set(layer_ids):
        raise ValueError("merge plan layers do not match profile layers.")
    for row, layer_id in enumerate(layer_ids):
        values = layer_plans[layer_id]
        if not isinstance(values, Mapping):
            raise ValueError(f"merge plan layer {layer_id} is invalid.")
        retained = values.get("retained_indices")
        pruned = values.get("pruned_indices")
        representative = values.get("representative_indices")
        beta = values.get("beta")
        rejection_codes = values.get("rejection_codes")
        cumulative_relative = values.get("cumulative_relative_delta_norm")
        if not all(
            isinstance(item, torch.Tensor)
            for item in (retained, pruned, representative, beta, rejection_codes, cumulative_relative)
        ):
            raise ValueError(f"merge plan layer {layer_id} contains non-tensor values.")
        num_experts = int(widths.shape[1])
        if retained.ndim != 2 or int(retained.shape[0]) != num_experts:
            raise ValueError(f"merge plan layer {layer_id} retained_indices shape is invalid.")
        if pruned.ndim != 2 or int(pruned.shape[0]) != num_experts:
            raise ValueError(f"merge plan layer {layer_id} pruned_indices shape is invalid.")
        if representative.shape != pruned.shape or beta.shape != pruned.shape:
            raise ValueError(f"merge plan layer {layer_id} pruned tensors must align.")
        if rejection_codes.shape != pruned.shape:
            raise ValueError(f"merge plan layer {layer_id} rejection_codes must align with pruned tensors.")
        if cumulative_relative.shape != retained.shape:
            raise ValueError(
                f"merge plan layer {layer_id} cumulative relative delta norms must align with retained tensors."
            )
        channel_layer = table[layer_id]
        config = merge_plan.get("config", {})
        beta_max = float(config.get("beta_max", float("inf")))
        relative_max = float(config.get("representative_relative_delta_norm_max", float("inf")))
        for expert_idx in range(num_experts):
            channel_count = int(
                channel_layer.block_sizes[: int(widths[row, expert_idx].item())].sum().item()
            )
            expert_retained = retained[expert_idx].to(torch.long)
            expert_pruned = pruned[expert_idx].to(torch.long)
            expert_representative = representative[expert_idx].to(torch.long)
            expert_beta = beta[expert_idx].float()
            expert_rejection = rejection_codes[expert_idx].to(torch.long)
            expert_cumulative_relative = cumulative_relative[expert_idx].float()
            expected_retained = channel_layer.ranked_indices[expert_idx, :channel_count].to(torch.long)
            if not torch.equal(expert_retained, expected_retained):
                raise ValueError(
                    f"merge plan layer {layer_id} expert {expert_idx} retained indices do not match channel prefix."
                )
            partition = torch.cat((expert_retained, expert_pruned))
            if int(partition.numel()) != int(channel_layer.intermediate_size) or not torch.equal(
                torch.sort(partition).values,
                torch.arange(channel_layer.intermediate_size, dtype=torch.long),
            ):
                raise ValueError(f"merge plan layer {layer_id} expert {expert_idx} channel partition is invalid.")
            if not bool(torch.isfinite(expert_beta).all()):
                raise ValueError(f"merge plan layer {layer_id} expert {expert_idx} beta contains non-finite values.")
            if bool(expert_beta.abs().gt(beta_max).any()):
                raise ValueError(f"merge plan layer {layer_id} expert {expert_idx} beta exceeds beta_max.")
            if not bool(torch.isfinite(expert_cumulative_relative).all()) or bool(
                (expert_cumulative_relative < 0).any()
            ):
                raise ValueError(
                    f"merge plan layer {layer_id} expert {expert_idx} cumulative relative norms are invalid."
                )
            if bool(expert_cumulative_relative.gt(relative_max + 1.0e-6).any()):
                raise ValueError(
                    f"merge plan layer {layer_id} expert {expert_idx} cumulative relative norm exceeds the limit."
                )
            if bool(((expert_rejection < 0) | (expert_rejection > 3)).any()):
                raise ValueError(f"merge plan layer {layer_id} expert {expert_idx} rejection code is invalid.")
            accepted = expert_representative >= 0
            if bool((expert_rejection[accepted] != 0).any()) or bool((expert_rejection[~accepted] == 0).any()):
                raise ValueError(
                    f"merge plan layer {layer_id} expert {expert_idx} rejection codes do not match acceptance."
                )
            if bool((expert_beta[~accepted] != 0).any()):
                raise ValueError(
                    f"merge plan layer {layer_id} expert {expert_idx} rejected pairs must have zero beta."
                )
            retained_lookup = torch.zeros(channel_layer.intermediate_size, dtype=torch.bool)
            retained_lookup[expert_retained] = True
            if bool(accepted.any()) and not bool(retained_lookup[expert_representative[accepted]].all()):
                raise ValueError(
                    f"merge plan layer {layer_id} expert {expert_idx} representative is not retained."
                )
    merge_plan["merge_plan_file_sha256"] = merge_hash
    return merge_plan, merge_hash


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def build_static_expert_profile_api():
    """Create the EvalScope ModelAPI class lazily to keep this module optional."""

    from evalscope.api.model import ModelAPI
    from evalscope.api.registry import register_model_api
    from evalscope.models.modelscope import ModelScopeAPI

    @register_model_api(name="static_expert_profile")
    class StaticExpertProfileAPI(ModelScopeAPI):
        def __init__(
            self,
            model_name: str,
            base_url: str | None = None,
            api_key: str | None = None,
            config=None,
            **model_args: Any,
        ) -> None:
            profile_path = model_args.pop("profile_path", None)
            channel_path = model_args.pop("channel_cache_path", None)
            if profile_path is None or channel_path is None:
                raise ValueError("static_expert_profile requires profile_path and channel_cache_path.")
            model_path = str(model_args.pop("model_path", model_name))
            model_family = model_args.pop("model_family", None)
            device_map = model_args.pop("device_map", None)
            correction_mode = str(model_args.pop("correction_mode", "none"))
            max_correction_ratio = float(model_args.pop("max_correction_ratio", 0.20))
            moe_backend = str(model_args.pop("moe_backend", "torch_index_add"))
            stats_path = Path(model_args.pop("stats_path", Path(profile_path).with_suffix(".evalscope.json")))
            expected_profile_hash = model_args.pop("expected_profile_file_sha256", None)
            expected_channel_hash = model_args.pop("expected_channel_file_sha256", None)
            merge_plan_path = model_args.pop("merge_plan_path", None)
            expected_merge_plan_hash = model_args.pop("expected_merge_plan_file_sha256", None)
            chat_template = model_args.pop("chat_template", None)
            tokenizer_call_args = model_args.pop("tokenizer_call_args", {})
            enable_thinking = model_args.pop("enable_thinking", False)
            tokenizer_path = model_args.pop("tokenizer_path", None)
            if model_args:
                raise ValueError(f"Unknown static_expert_profile model args: {sorted(model_args)}")

            profile, channel, widths, channels = validate_static_profile_artifacts(
                model_path=model_path,
                profile_path=profile_path,
                channel_cache_path=channel_path,
                expected_profile_file_sha256=expected_profile_hash,
                expected_channel_file_sha256=expected_channel_hash,
            )
            expected_profile_merge_hash = profile.get("cache_provenance", {}).get("merge_plan", {}).get("sha256")
            if bool(merge_plan_path) != bool(expected_profile_merge_hash):
                raise ValueError("profile merge provenance and merge_plan_path must either both be present or both absent.")
            merge_plan = None
            merge_plan_hash = None
            if merge_plan_path is not None:
                merge_plan, merge_plan_hash = validate_static_merge_plan_artifact(
                    model_path=model_path,
                    merge_plan_path=merge_plan_path,
                    profile=profile,
                    channel=channel,
                    widths=widths,
                    table=channels,
                    expected_merge_plan_file_sha256=expected_merge_plan_hash,
                )
            family = resolve_model_family(model_path=model_path, model_family=model_family)
            ModelAPI.__init__(self, model_name=model_name, base_url=base_url, api_key=api_key, config=config)
            self._model_path = model_path
            self._profile_path = str(Path(profile_path).expanduser().resolve())
            self._channel_cache_path = str(Path(channel_path).expanduser().resolve())
            self._merge_plan_path = (
                None if merge_plan_path is None else str(Path(merge_plan_path).expanduser().resolve())
            )
            self._profile = profile
            self._channel = channel
            self._stats_path = stats_path
            self._stats_lock = RLock()
            self._runtime_stats = StaticExpertRuntimeStats(
                profile_widths=widths,
                num_blocks=int(profile["num_blocks"]),
            )
            self.chat_template = chat_template
            self.tokenizer_call_args = tokenizer_call_args or {}
            self.enable_thinking = enable_thinking
            self.device = device_map
            self.torch_dtype = "auto"
            self.model, self.tokenizer = load_supported_moe(
                model_path,
                device_map=device_map,
                model_family=family,
            )
            if tokenizer_path is not None:
                from transformers import AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token or "[PAD]"
            self.tokenizer.padding_side = "left"
            if self.chat_template:
                self.tokenizer.chat_template = self.chat_template
            self._validate_loaded_topology(channels)
            self._family = family
            self._correction_mode = correction_mode
            self._moe_backend = moe_backend
            self._merge_audit = None
            if merge_plan is not None:
                self._merge_audit = apply_static_down_projection_merge_plan(
                    self.model,
                    merge_plan,
                    channels,
                    profile_widths_by_layer(widths, layer_ids=[int(x) for x in profile["layer_ids"]]),
                    merge_plan_file_sha256=str(merge_plan_hash),
                )
            retained_experts_by_layer = None
            retained_expert_mask = profile.get("retained_expert_mask")
            if isinstance(retained_expert_mask, torch.Tensor):
                retained_experts_by_layer = {
                    int(layer_id): retained_expert_mask[row].detach().cpu().to(torch.bool)
                    for row, layer_id in enumerate(profile["layer_ids"])
                }
            self._patch_context = patch_qwen3_moe_blocks_static_expert(
                self.model,
                profile_widths_by_layer(widths, layer_ids=[int(x) for x in profile["layer_ids"]]),
                channels,
                retained_experts_by_layer=retained_experts_by_layer,
                correction_mode=correction_mode,
                max_correction_ratio=max_correction_ratio,
                runtime_stats=self._runtime_stats,
                moe_backend=moe_backend,
            )
            self._patch_context.__enter__()
            self._write_runtime_stats()

        def _validate_loaded_topology(self, channels: ChannelTable) -> None:
            bindings = {int(binding.layer_idx): binding for binding in iter_moe_layer_bindings(self.model)}
            expected = {int(layer_id) for layer_id in self._profile["layer_ids"]}
            if set(bindings) != expected:
                raise ValueError(f"loaded model MoE layers {sorted(bindings)} do not match profile layers {sorted(expected)}.")
            for layer_id, channel_layer in channels.items():
                binding = bindings[layer_id]
                if _expert_count(binding.experts) != int(channel_layer.ranked_indices.shape[0]):
                    raise ValueError(f"layer {layer_id} expert count does not match channel cache.")
                if _expert_intermediate_size(binding.experts) != int(channel_layer.intermediate_size):
                    raise ValueError(f"layer {layer_id} intermediate size does not match channel cache.")

        def _write_runtime_stats(self) -> None:
            with self._stats_lock:
                payload = {
                    "evaluation_backend": "evalscope",
                    "evalscope_model_api": "static_expert_profile",
                    "model_path": self._model_path,
                    "model_family": self._family,
                    "profile_path": self._profile_path,
                    "profile_file_sha256": self._profile.get("profile_file_sha256"),
                    "channel_cache_path": self._channel_cache_path,
                    "channel_file_sha256": self._channel.get("channel_file_sha256"),
                    "merge_plan_path": self._merge_plan_path,
                    "merge_plan_file_sha256": self._profile.get("cache_provenance", {})
                    .get("merge_plan", {})
                    .get("sha256"),
                    "merge_audit": self._merge_audit,
                    "method": self._profile.get("method"),
                    "mode": self._profile.get("mode"),
                    "correction_mode": self._correction_mode,
                    "moe_backend": self._moe_backend,
                    "structural_pruning_ratio": self._runtime_stats.structural_pruning_ratio(),
                    "routed_pruning_ratio": self._runtime_stats.routed_pruning_ratio(),
                    "width_histogram": self._runtime_stats.aggregate_width_histogram(),
                    "routed_pruning_by_layer": self._runtime_stats.routed_pruning_by_layer(),
                }
                _write_json_atomic(self._stats_path, payload)

        def generate(self, *args, **kwargs):
            output = super().generate(*args, **kwargs)
            self._write_runtime_stats()
            return output

    return StaticExpertProfileAPI


def register_static_expert_profile_api() -> None:
    build_static_expert_profile_api()