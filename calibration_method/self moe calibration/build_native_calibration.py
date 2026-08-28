"""Build checkpoint-native, naturally terminated MoE calibration blocks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_CODE_ROOT = REPO_ROOT / "static_moe_prunning" / "code"
if str(CALIBRATION_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CALIBRATION_CODE_ROOT))

from src.calibration_data import build_model_cache_identity


SENTINEL = "__CN_MOE_SC_USER_SENTINEL_7F3A__"
PROTOCOL_VERSION = "cn_moe_sc_native_dialogue_v2_minimal_intervention"


@dataclass(frozen=True)
class NativeScaffold:
    """Tokenized native user/assistant scaffold for one checkpoint."""

    user_prefix: list[int]
    user_bridge: list[int]
    assistant_suffix: list[int]
    user_stop_token_id: int
    assistant_stop_token_id: int


@dataclass(frozen=True)
class Episode:
    """One independently generated native self-dialogue episode."""

    token_ids: list[int]
    user_tokens: int
    assistant_tokens: int
    user_terminated: bool
    assistant_terminated: bool
    seed: int
    user_token_ids: tuple[int, ...] = ()
    assistant_token_ids: tuple[int, ...] = ()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=128)
    parser.add_argument(
        "--discovery-blocks",
        type=int,
        default=None,
        help="Natural discovery blocks; the remainder is the coverage reserve.",
    )
    parser.add_argument("--block-length", type=int, default=2048)
    parser.add_argument("--episode-batch-size", type=int, default=16)
    parser.add_argument("--pilot-episodes", type=int, default=16)
    parser.add_argument("--max-attempts", type=int, default=512)
    parser.add_argument("--max-user-tokens", type=int, default=512)
    parser.add_argument("--max-assistant-tokens", type=int, default=1024)
    parser.add_argument(
        "--user-generation-mode",
        choices=("assistant_bootstrap", "user_role_continuation"),
        default="assistant_bootstrap",
        help="Generate semantic content from the trained assistant role by default.",
    )
    parser.add_argument("--max-pilot-rejection-rate", type=float, default=0.20)
    parser.add_argument("--warmup-temperature", type=float, default=1.5)
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _tokenize(tokenizer: Any, text: str) -> list[int]:
    """Tokenize rendered template text without adding an extra special token."""

    encoded = tokenizer(text, add_special_tokens=False)
    return [int(token_id) for token_id in encoded["input_ids"]]


def _render(tokenizer: Any, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> str:
    """Render a native template while tolerating tokenizer-specific options."""

    try:
        return str(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=False,
            )
        )
    except TypeError:
        return str(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        )


def build_native_scaffold(tokenizer: Any) -> NativeScaffold:
    """Derive role, turn, and stop tokens from the checkpoint chat template."""

    user_messages = [{"role": "user", "content": SENTINEL}]
    generation_rendered = _render(tokenizer, user_messages, add_generation_prompt=True)
    generation_parts = generation_rendered.split(SENTINEL)
    if len(generation_parts) != 2:
        raise ValueError("checkpoint generation template did not preserve the user calibration sentinel")
    user_prefix_text, user_bridge_text = generation_parts

    completed_messages = [
        {"role": "user", "content": SENTINEL},
        {"role": "assistant", "content": SENTINEL},
    ]
    completed_rendered = _render(tokenizer, completed_messages, add_generation_prompt=False)
    parts = completed_rendered.split(SENTINEL)
    if len(parts) != 3:
        raise ValueError("checkpoint chat template did not preserve the calibration sentinel twice")
    assistant_suffix_text = parts[2]
    user_prefix = _tokenize(tokenizer, user_prefix_text)
    user_bridge = _tokenize(tokenizer, user_bridge_text)
    assistant_suffix = _tokenize(tokenizer, assistant_suffix_text)
    if not user_prefix or not user_bridge or not assistant_suffix:
        raise ValueError("checkpoint native chat template produced an empty scaffold component")
    return NativeScaffold(
        user_prefix=user_prefix,
        user_bridge=user_bridge,
        assistant_suffix=assistant_suffix,
        user_stop_token_id=user_bridge[0],
        assistant_stop_token_id=assistant_suffix[0],
    )


def _sampling_params(
    sampling_params_type: Any,
    *,
    max_tokens: int,
    stop_token_id: int,
    seed: int,
    temperature: float = 1.0,
) -> Any:
    """Build minimally intervening sampling parameters with native termination."""

    return sampling_params_type(
        temperature=temperature,
        top_p=1.0,
        top_k=0,
        repetition_penalty=1.0,
        max_tokens=max_tokens,
        min_tokens=0,
        ignore_eos=False,
        stop_token_ids=[int(stop_token_id)],
        detokenize=False,
        seed=seed,
    )


def _output_tokens(output: Any, max_tokens: int) -> tuple[list[int], bool]:
    """Return generated token IDs and whether generation stopped before its cap."""

    if not output.outputs:
        return [], False
    candidate = output.outputs[0]
    tokens = [int(token_id) for token_id in candidate.token_ids]
    finish_reason = str(getattr(candidate, "finish_reason", ""))
    naturally_terminated = finish_reason in {"stop", "eos"} or len(tokens) < max_tokens
    return tokens, naturally_terminated


def _without_stop(tokens: list[int], stop_token_id: int, naturally_terminated: bool) -> tuple[list[int], bool]:
    """Remove a stop token if vLLM includes it and report natural termination."""

    if tokens and tokens[-1] == stop_token_id:
        return tokens[:-1], True
    return tokens, naturally_terminated


def _repeat_metrics(token_ids: Iterable[int]) -> dict[str, float]:
    """Compute mechanical degeneration metrics for one episode."""

    tokens = list(token_ids)
    if not tokens:
        return {
            "distinct_token_ratio": 0.0,
            "dominant_token_ratio": 1.0,
            "max_run_ratio": 1.0,
            "repeated_4gram_ratio": 1.0,
            "periodic_loop_ratio": 1.0,
        }
    counts: dict[int, int] = {}
    max_run = 1
    run = 1
    for index, token_id in enumerate(tokens):
        counts[token_id] = counts.get(token_id, 0) + 1
        if index and token_id == tokens[index - 1]:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 1
    repeated_positions: set[int] = set()
    seen_4grams: dict[tuple[int, ...], int] = {}
    for index in range(max(0, len(tokens) - 3)):
        ngram = tuple(tokens[index:index + 4])
        if ngram in seen_4grams:
            repeated_positions.update(range(index, index + 4))
            first = seen_4grams[ngram]
            repeated_positions.update(range(first, first + 4))
        else:
            seen_4grams[ngram] = index
    periodic_tokens = 0
    for period in range(1, min(32, len(tokens) // 4) + 1):
        suffix = tokens[-period * 4:]
        pattern = suffix[:period]
        matched = 0
        for offset, token_id in enumerate(reversed(tokens)):
            if token_id != pattern[-1 - (offset % period)]:
                break
            matched += 1
        periodic_tokens = max(periodic_tokens, matched)
    return {
        "distinct_token_ratio": len(counts) / len(tokens),
        "dominant_token_ratio": max(counts.values()) / len(tokens),
        "max_run_ratio": max_run / len(tokens),
        "repeated_4gram_ratio": len(repeated_positions) / len(tokens),
        "periodic_loop_ratio": periodic_tokens / len(tokens),
    }


def _is_valid_episode(episode: Episode, tokenizer: Any | None = None) -> tuple[bool, dict[str, Any]]:
    """Require both turns to pass role-specific mechanical validity gates."""

    user_tokens = list(episode.user_token_ids)
    assistant_tokens = list(episode.assistant_token_ids)
    if not user_tokens and not assistant_tokens:
        # Unit-test/backward-compatible fallback for manually constructed episodes.
        user_tokens = episode.token_ids[:episode.user_tokens]
        assistant_tokens = episode.token_ids[-episode.assistant_tokens:] if episode.assistant_tokens else []
    user_valid, user_metrics = _is_valid_turn(user_tokens, role="user", tokenizer=tokenizer)
    assistant_valid, assistant_metrics = _is_valid_turn(assistant_tokens, role="assistant", tokenizer=tokenizer)
    episode_metrics = _repeat_metrics(episode.token_ids)
    metrics = {
        "user": user_metrics,
        "assistant": assistant_metrics,
        "episode": episode_metrics,
    }
    valid = user_valid and assistant_valid
    return valid, metrics


def _text_metrics(text: str) -> dict[str, float]:
    """Detect repeated lexical fragments and excessive cross-script switching."""

    compact = re.sub(r"\s+", " ", text).strip()
    words = re.findall(r"\w+", compact.casefold(), flags=re.UNICODE)
    repeated_word_positions: set[int] = set()
    for size in (1, 2, 3):
        seen: set[tuple[str, ...]] = set()
        for index in range(max(0, len(words) - size + 1)):
            phrase = tuple(words[index:index + size])
            if phrase in seen:
                repeated_word_positions.update(range(index, index + size))
            seen.add(phrase)

    script_sequence = []
    for char in compact:
        if "a" <= char.casefold() <= "z":
            script = "latin"
        elif "\u3400" <= char <= "\u9fff":
            script = "cjk"
        elif "\u0400" <= char <= "\u04ff":
            script = "cyrillic"
        elif "\u0600" <= char <= "\u06ff":
            script = "arabic"
        elif "\u1200" <= char <= "\u137f":
            script = "ethiopic"
        elif "\u0900" <= char <= "\u097f":
            script = "devanagari"
        elif "\u0e00" <= char <= "\u0e7f":
            script = "thai"
        elif char.isalpha():
            script = "other"
        else:
            continue
        script_sequence.append(script)
    switches = sum(first != second for first, second in zip(script_sequence, script_sequence[1:]))
    script_counts: dict[str, int] = {}
    for script in script_sequence:
        script_counts[script] = script_counts.get(script, 0) + 1
    minority_share = 0.0
    if script_sequence:
        minority_share = 1.0 - max(script_counts.values()) / len(script_sequence)

    return {
        "repeated_word_ratio": len(repeated_word_positions) / max(len(words), 1),
        "script_switch_ratio": switches / max(len(script_sequence) - 1, 1),
        "minority_script_share": minority_share,
        "characters": float(len(compact)),
        "words": float(len(words)),
    }


def _is_valid_turn(
    token_ids: Iterable[int],
    *,
    role: str,
    tokenizer: Any | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Reject only empty or mechanically degenerate turns.

    Text style, semantic mode, language, and lexical repetition are deliberately
    diagnostic-only.  This keeps clarification, refusal, list, boilerplate,
    and multilingual behavior in the measured model distribution.
    """

    tokens = list(token_ids)
    metrics = _repeat_metrics(tokens)
    text = "" if tokenizer is None else tokenizer.decode(tokens, skip_special_tokens=True)
    text_metrics = _text_metrics(text) if tokenizer is not None else {
        "repeated_word_ratio": 0.0,
        "script_switch_ratio": 0.0,
        "minority_script_share": 0.0,
        "characters": 0.0,
        "words": 0.0,
    }
    metrics = {**metrics, **text_metrics}
    valid = (
        bool(tokens)
        and (tokenizer is None or bool(text.strip()))
        and metrics["distinct_token_ratio"] >= 0.02
        and metrics["dominant_token_ratio"] <= 0.85
        and metrics["max_run_ratio"] <= 0.60
        and metrics["repeated_4gram_ratio"] <= 0.90
        and metrics["periodic_loop_ratio"] <= 0.60
    )
    return valid, metrics


def _generate_batch(
    llm: Any,
    sampling_params_type: Any,
    prompts: list[list[int]],
    *,
    max_tokens: int,
    stop_token_id: int,
    seeds: list[int],
    warmup_temperature: float = 1.0,
    warmup_tokens: int = 0,
) -> list[tuple[list[int], bool]]:
    """Generate one independently seeded batch from token-ID prompts."""

    if warmup_tokens > 0 and warmup_temperature != 1.0:
        prefix_tokens = min(warmup_tokens, max_tokens)
        prefix_params = [
            _sampling_params(
                sampling_params_type,
                max_tokens=prefix_tokens,
                stop_token_id=stop_token_id,
                seed=seed,
                temperature=warmup_temperature,
            )
            for seed in seeds
        ]
        prefix_outputs = llm.generate(
            [{"prompt_token_ids": prompt} for prompt in prompts],
            prefix_params,
            use_tqdm=False,
        )
        prefixes = [_output_tokens(output, prefix_tokens) for output in prefix_outputs]
        results: list[tuple[list[int], bool] | None] = [None] * len(prompts)
        pending_indexes = []
        pending_prompts = []
        pending_seeds = []
        for index, ((tokens, terminated), prompt, seed) in enumerate(zip(prefixes, prompts, seeds)):
            if terminated or len(tokens) >= max_tokens:
                results[index] = (tokens, terminated)
                continue
            pending_indexes.append(index)
            pending_prompts.append(prompt + tokens)
            pending_seeds.append(seed + 10_000_000)
        if pending_indexes:
            suffixes = _generate_batch(
                llm,
                sampling_params_type,
                pending_prompts,
                max_tokens=max_tokens - prefix_tokens,
                stop_token_id=stop_token_id,
                seeds=pending_seeds,
            )
            for index, (suffix_tokens, terminated) in zip(pending_indexes, suffixes):
                results[index] = (prefixes[index][0] + suffix_tokens, terminated)
        return [result for result in results if result is not None]

    params = [
        _sampling_params(
            sampling_params_type,
            max_tokens=max_tokens,
            stop_token_id=stop_token_id,
            seed=seed,
        )
        for seed in seeds
    ]
    outputs = llm.generate([{"prompt_token_ids": prompt} for prompt in prompts], params, use_tqdm=False)
    return [_output_tokens(output, max_tokens) for output in outputs]


def _build_episodes(
    llm: Any,
    sampling_params_type: Any,
    scaffold: NativeScaffold,
    *,
    count: int,
    seed_start: int,
    batch_size: int,
    max_user_tokens: int,
    max_assistant_tokens: int,
    user_generation_mode: str = "assistant_bootstrap",
    special_token_ids: set[int] | None = None,
    tokenizer: Any | None = None,
    warmup_temperature: float = 1.0,
    warmup_tokens: int = 0,
) -> list[Episode]:
    """Generate independent user and assistant turns using native token scaffolds."""

    episodes: list[Episode] = []
    for begin in range(0, count, batch_size):
        current = min(batch_size, count - begin)
        seeds = [seed_start + begin + offset for offset in range(current)]
        if user_generation_mode == "assistant_bootstrap":
            user_prompts = [scaffold.user_prefix + scaffold.user_bridge for _ in range(current)]
            user_stop_token_id = scaffold.assistant_stop_token_id
        else:
            user_prompts = [scaffold.user_prefix for _ in range(current)]
            user_stop_token_id = scaffold.user_stop_token_id
        user_outputs = _generate_batch(
            llm,
            sampling_params_type,
            user_prompts,
            max_tokens=max_user_tokens,
            stop_token_id=user_stop_token_id,
            seeds=seeds,
            warmup_temperature=warmup_temperature,
            warmup_tokens=warmup_tokens,
        )
        user_parts = [
            _without_stop(tokens, user_stop_token_id, naturally_terminated)
            for tokens, naturally_terminated in user_outputs
        ]
        if user_generation_mode == "assistant_bootstrap" and special_token_ids:
            user_parts = [
                ([token_id for token_id in tokens if token_id not in special_token_ids], terminated)
                for tokens, terminated in user_parts
            ]
        valid_user_indexes = [
            index for index, (tokens, _) in enumerate(user_parts)
            if _is_valid_turn(tokens, role="user", tokenizer=tokenizer)[0]
        ]
        assistant_outputs_by_index: dict[int, tuple[list[int], bool]] = {}
        if valid_user_indexes:
            assistant_prompts = [
                scaffold.user_prefix + user_parts[index][0] + scaffold.user_bridge
                for index in valid_user_indexes
            ]
            assistant_outputs = _generate_batch(
                llm,
                sampling_params_type,
                assistant_prompts,
                max_tokens=max_assistant_tokens,
                stop_token_id=scaffold.assistant_stop_token_id,
                seeds=[seeds[index] + 1_000_000 for index in valid_user_indexes],
                warmup_temperature=warmup_temperature,
                warmup_tokens=warmup_tokens,
            )
            assistant_outputs_by_index = dict(zip(valid_user_indexes, assistant_outputs))
        for index, user_part in enumerate(user_parts):
            user_tokens, user_terminated = user_part
            assistant_output_tokens, assistant_stopped = assistant_outputs_by_index.get(index, ([], False))
            assistant_tokens, assistant_terminated = _without_stop(
                assistant_output_tokens,
                scaffold.assistant_stop_token_id,
                assistant_stopped,
            )
            token_ids = (
                scaffold.user_prefix
                + user_tokens
                + scaffold.user_bridge
                + assistant_tokens
                + scaffold.assistant_suffix
            )
            episode = Episode(
                token_ids=token_ids,
                user_tokens=len(user_tokens),
                assistant_tokens=len(assistant_tokens),
                user_terminated=user_terminated,
                assistant_terminated=assistant_terminated,
                seed=seeds[index],
                user_token_ids=tuple(user_tokens),
                assistant_token_ids=tuple(assistant_tokens),
            )
            episodes.append(episode)
    return episodes


def _pack_blocks(
    episodes: list[Episode],
    *,
    blocks: int,
    block_length: int,
) -> tuple[torch.Tensor, list[dict[str, Any]], list[dict[str, Any]]]:
    """Pack valid episodes into fixed-size blocks and preserve episode boundaries."""

    flat: list[int] = []
    episode_records: list[dict[str, Any]] = []
    block_records: list[dict[str, Any]] = []
    block_start = 0
    block_id = 0
    for episode_id, episode in enumerate(episodes):
        remaining = len(episode.token_ids)
        source_offset = 0
        while remaining and block_id < blocks:
            block_offset = len(flat) - block_start
            take = min(remaining, block_length - block_offset)
            start = len(flat)
            flat.extend(episode.token_ids[source_offset:source_offset + take])
            episode_records.append({
                "episode_id": episode_id,
                "block_id": block_id,
                "start": start,
                "end": start + take,
                "complete": take == remaining,
                "user_tokens": episode.user_tokens,
                "assistant_tokens": episode.assistant_tokens,
                "user_terminated": episode.user_terminated,
                "assistant_terminated": episode.assistant_terminated,
                "seed": episode.seed,
                "source_start": source_offset,
                "source_end": source_offset + take,
            })
            remaining -= take
            source_offset += take
            if len(flat) - block_start == block_length:
                block_records.append({"block_id": block_id, "start": block_start, "end": len(flat)})
                block_id += 1
                block_start = len(flat)
        if block_id == blocks:
            break
    if block_id < blocks:
        raise ValueError(f"only packed {block_id}/{blocks} calibration blocks")
    token_ids = torch.tensor(flat[:blocks * block_length], dtype=torch.long).view(1, -1)
    return token_ids, episode_records, block_records


def _sha256_tokens(token_ids: torch.Tensor) -> str:
    """Hash canonical token shape and contents."""

    canonical = token_ids.to(dtype=torch.int64, device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(json.dumps(list(canonical.shape)).encode("ascii"))
    digest.update(b"int64")
    digest.update(canonical.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _build_payload(
    token_ids: torch.Tensor,
    episodes: list[Episode],
    episode_records: list[dict[str, Any]],
    block_records: list[dict[str, Any]],
    *,
    model_path: Path,
    blocks: int,
    block_length: int,
    seed: int,
    attempted: int,
    rejected: int,
    scaffold: NativeScaffold,
    warmup_enabled: bool,
    warmup_temperature: float,
    warmup_tokens: int,
    user_generation_mode: str,
    rejection_diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create an auditable shared calibration payload."""

    terminated_users = sum(episode.user_terminated for episode in episodes)
    terminated_assistants = sum(episode.assistant_terminated for episode in episodes)
    source = {
        "dataset": "model_generated_native_dialogue",
        "split": "model_generated",
        "source_type": "model_generated_natural",
        "arrow_files": [],
        "generation": {
            "backend": "vllm",
            "sampling": "independent_per_sequence",
            "episode_sampling": "independent_native_self_dialogue",
            "prompt": "checkpoint_native_chat_template",
            "user_generation_mode": user_generation_mode,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "repetition_penalty": 1.0,
            "ignore_eos": False,
            "min_tokens": 0,
            "seed": seed,
            "request_seed_rule": "seed_plus_episode_index",
            "natural_termination": True,
            "prefix_temperature_warmup": {
                "enabled": warmup_enabled,
                "temperature": warmup_temperature if warmup_enabled else 1.0,
                "tokens": warmup_tokens if warmup_enabled else 0,
            },
        },
    }
    return {
        "schema_version": 1,
        "purpose": "shared_moe_pruning_calibration",
        "protocol_name": PROTOCOL_VERSION,
        "model_path": str(model_path),
        "model_identity": build_model_cache_identity(model_path),
        "dataset": "model_generated_native_dialogue",
        "dataset_config": None,
        "split": "model_generated",
        "text_field": None,
        "sequence_length": block_length,
        "calibration_sequences": blocks,
        "calibration_tokens": int(token_ids.numel()),
        "calibration_token_offset": 0,
        "calibration_token_end": int(token_ids.numel()),
        "source": source,
        "token_stream": {
            "tokenization_strategy": "checkpoint_native_independent_episodes_packed_into_blocks",
            "add_special_tokens": False,
            "sequence_boundary": "fixed_length_blocks_with_episode_boundaries",
            "source_shape": [blocks, block_length],
            "selected_tokens": int(token_ids.numel()),
            "episode_boundaries": episode_records,
            "block_boundaries": block_records,
        },
        "input_ids": token_ids,
        "input_ids_sha256": _sha256_tokens(token_ids),
        "attention_mask_semantics": "all_ones_no_padding",
        "frozen_before_profile": True,
        "test_metrics_used": False,
        "generation_health": {
            "attempted_episodes": attempted,
            "accepted_episodes": len(episodes),
            "rejected_episodes": rejected,
            "rejection_rate": rejected / max(attempted, 1),
            "naturally_terminated_user_turns": terminated_users,
            "naturally_terminated_assistant_turns": terminated_assistants,
            "user_termination_rate": terminated_users / max(len(episodes), 1),
            "assistant_termination_rate": terminated_assistants / max(len(episodes), 1),
            "mean_episode_tokens": sum(len(episode.token_ids) for episode in episodes) / max(len(episodes), 1),
            "median_episode_tokens": sorted(len(episode.token_ids) for episode in episodes)[len(episodes) // 2]
            if episodes
            else 0,
            "mean_user_tokens": sum(episode.user_tokens for episode in episodes) / max(len(episodes), 1),
            "mean_assistant_tokens": sum(episode.assistant_tokens for episode in episodes)
            / max(len(episodes), 1),
            "native_scaffold": {
                "user_prefix_tokens": len(scaffold.user_prefix),
                "user_bridge_tokens": len(scaffold.user_bridge),
                "assistant_suffix_tokens": len(scaffold.assistant_suffix),
                "user_stop_token_id": scaffold.user_stop_token_id,
                "assistant_stop_token_id": scaffold.assistant_stop_token_id,
            },
        },
        "rejection_diagnostics": rejection_diagnostics,
    }


def build_calibration(args: argparse.Namespace) -> dict[str, Any]:
    """Generate, validate, and save one native calibration cache."""

    if args.blocks <= 0 or args.block_length <= 0 or args.max_attempts <= 0:
        raise ValueError("blocks, block-length, and max-attempts must be positive")
    discovery_blocks = args.discovery_blocks
    if discovery_blocks is None:
        discovery_blocks = args.blocks * 3 // 4
    if not 0 < discovery_blocks <= args.blocks:
        raise ValueError("discovery-blocks must be in [1, blocks]")
    if not 0.0 <= args.max_pilot_rejection_rate <= 1.0:
        raise ValueError("max-pilot-rejection-rate must be in [0, 1]")
    if args.warmup_temperature <= 0.0 or args.warmup_tokens < 0:
        raise ValueError("warmup-temperature must be positive and warmup-tokens cannot be negative")
    if args.output.exists() and not args.force:
        raise FileExistsError(f"calibration cache already exists: {args.output}")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    environment_bin = str(Path(sys.executable).resolve().parent)
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if environment_bin not in path_parts:
        os.environ["PATH"] = os.pathsep.join((environment_bin, *path_parts))
    print(f"runtime_python={sys.executable} runtime_ninja={shutil.which('ninja')}", flush=True)

    model_path = args.model_path.expanduser().resolve()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    scaffold = build_native_scaffold(tokenizer)
    special_token_ids = {int(token_id) for token_id in tokenizer.all_special_ids}
    print(
        f"native_scaffold user_prefix={len(scaffold.user_prefix)} "
        f"user_bridge={len(scaffold.user_bridge)} assistant_suffix={len(scaffold.assistant_suffix)}",
        flush=True,
    )
    llm = LLM(
        model=str(model_path),
        runner="generate",
        tokenizer=str(model_path),
        trust_remote_code=True,
        tensor_parallel_size=1,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        seed=args.seed,
        max_model_len=max(args.block_length, args.max_user_tokens + args.max_assistant_tokens + 256),
    )

    pilot_count = min(args.pilot_episodes, args.max_attempts)
    pilot = _build_episodes(
        llm,
        SamplingParams,
        scaffold,
        count=pilot_count,
        seed_start=args.seed,
        batch_size=args.episode_batch_size,
        max_user_tokens=args.max_user_tokens,
        max_assistant_tokens=args.max_assistant_tokens,
        user_generation_mode=args.user_generation_mode,
        special_token_ids=special_token_ids,
        tokenizer=tokenizer,
    )
    pilot_valid = [_is_valid_episode(episode, tokenizer)[0] for episode in pilot]
    rejection_diagnostics: list[dict[str, Any]] = []
    for index, (episode, valid) in enumerate(zip(pilot, pilot_valid)):
        if not valid:
            _, metrics = _is_valid_episode(episode, tokenizer)
            rejection_diagnostics.append({"stage": "pilot", "index": index, "seed": episode.seed, "metrics": metrics})
    pilot_rejection_rate = sum(not valid for valid in pilot_valid) / max(len(pilot_valid), 1)
    print(
        f"pilot accepted={sum(pilot_valid)}/{len(pilot_valid)} "
        f"rejection_rate={pilot_rejection_rate:.4f}",
        flush=True,
    )
    warmup_enabled = pilot_rejection_rate > args.max_pilot_rejection_rate
    warmup_pilot: list[Episode] = []
    warmup_valid: list[bool] = []
    if warmup_enabled:
        warmup_pilot = _build_episodes(
            llm,
            SamplingParams,
            scaffold,
            count=pilot_count,
            seed_start=args.seed + pilot_count,
            batch_size=args.episode_batch_size,
            max_user_tokens=args.max_user_tokens,
            max_assistant_tokens=args.max_assistant_tokens,
            user_generation_mode=args.user_generation_mode,
            special_token_ids=special_token_ids,
            tokenizer=tokenizer,
            warmup_temperature=args.warmup_temperature,
            warmup_tokens=args.warmup_tokens,
        )
        warmup_valid = [_is_valid_episode(episode, tokenizer)[0] for episode in warmup_pilot]
        for index, (episode, valid) in enumerate(zip(warmup_pilot, warmup_valid)):
            if not valid:
                _, metrics = _is_valid_episode(episode, tokenizer)
                rejection_diagnostics.append(
                    {"stage": "warmup_pilot", "index": index, "seed": episode.seed, "metrics": metrics}
                )
        warmup_rejection_rate = sum(not valid for valid in warmup_valid) / max(len(warmup_valid), 1)
        print(
            f"warmup_pilot accepted={sum(warmup_valid)}/{len(warmup_valid)} "
            f"rejection_rate={warmup_rejection_rate:.4f}",
            flush=True,
        )
        if warmup_rejection_rate > args.max_pilot_rejection_rate:
            raise RuntimeError(
                "checkpoint-native generation remains unhealthy after prefix temperature warm-up: "
                f"rejection_rate={warmup_rejection_rate:.4f}"
            )

    accepted: list[Episode] = []
    target_tokens = args.blocks * args.block_length
    accepted_tokens = 0
    attempted = 0
    rejected = 0
    seed_cursor = args.seed + pilot_count * (2 if warmup_enabled else 1)
    while attempted < args.max_attempts and accepted_tokens < target_tokens:
        request_count = min(args.episode_batch_size, args.max_attempts - attempted)
        generated = _build_episodes(
            llm,
            SamplingParams,
            scaffold,
            count=request_count,
            seed_start=seed_cursor,
            batch_size=request_count,
            max_user_tokens=args.max_user_tokens,
            max_assistant_tokens=args.max_assistant_tokens,
            user_generation_mode=args.user_generation_mode,
            special_token_ids=special_token_ids,
            tokenizer=tokenizer,
            warmup_temperature=args.warmup_temperature if warmup_enabled else 1.0,
            warmup_tokens=args.warmup_tokens if warmup_enabled else 0,
        )
        seed_cursor += request_count
        attempted += request_count
        for generated_index, episode in enumerate(generated):
            valid, metrics = _is_valid_episode(episode, tokenizer)
            if valid:
                accepted.append(episode)
                accepted_tokens += len(episode.token_ids)
            else:
                rejected += 1
                rejection_diagnostics.append(
                    {
                        "stage": "formal",
                        "index": attempted - request_count + generated_index,
                        "seed": episode.seed,
                        "metrics": metrics,
                    }
                )
            if len(accepted) <= 3 or len(accepted) % 16 == 0:
                print(
                    f"episode accepted={len(accepted)} tokens={accepted_tokens}/{target_tokens} "
                    f"length={len(episode.token_ids)} valid={valid} metrics={metrics}",
                    flush=True,
                )
            if accepted_tokens >= target_tokens:
                break

    token_ids, episode_records, block_records = _pack_blocks(
        accepted,
        blocks=args.blocks,
        block_length=args.block_length,
    )
    payload = _build_payload(
        token_ids,
        accepted,
        episode_records,
        block_records,
        model_path=model_path,
        blocks=args.blocks,
        block_length=args.block_length,
        seed=args.seed,
        attempted=attempted + pilot_count + len(warmup_pilot),
        rejected=(
            rejected
            + sum(not valid for valid in pilot_valid)
            + sum(not valid for valid in warmup_valid)
        ),
        scaffold=scaffold,
        warmup_enabled=warmup_enabled,
        warmup_temperature=args.warmup_temperature,
        warmup_tokens=args.warmup_tokens,
        user_generation_mode=args.user_generation_mode,
        rejection_diagnostics=rejection_diagnostics,
    )
    payload["calibration_pools"] = {
        "natural_discovery": {
            "block_start": 0,
            "block_end": discovery_blocks,
            "blocks": discovery_blocks,
            "tokens": discovery_blocks * args.block_length,
            "role": "estimate_natural_routing_and_conditioned_statistics",
        },
        "coverage_reserve": {
            "block_start": discovery_blocks,
            "block_end": args.blocks,
            "blocks": args.blocks - discovery_blocks,
            "tokens": (args.blocks - discovery_blocks) * args.block_length,
            "role": "additional_conditioned_samples_only",
            "exclude_from_natural_prevalence": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(
        f"saved={args.output.resolve()} shape={tuple(token_ids.shape)} "
        f"accepted_episodes={len(accepted)} rejection_rate={payload['generation_health']['rejection_rate']:.4f}",
        flush=True,
    )
    return payload


def main() -> int:
    """Run the native calibration builder."""

    args = parse_args()
    build_calibration(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())