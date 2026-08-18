from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import torch


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_TOKENIZER_FILE_NAMES = (
    "added_tokens.json",
    "merges.txt",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
    "vocab.txt",
)

_TOKENIZATION_CONFIG_FIELDS = (
    "model_type",
    "vocab_size",
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
    "decoder_start_token_id",
)


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def token_tensor_sha256(token_ids: torch.Tensor) -> str:
    """Hash the canonical CPU int64 shape and contents of a token tensor."""

    if not isinstance(token_ids, torch.Tensor) or token_ids.ndim != 2:
        raise ValueError("token_ids must be a two-dimensional tensor.")
    canonical = token_ids.detach().to(dtype=torch.int64, device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(json.dumps(list(canonical.shape)).encode("ascii"))
    digest.update(b"int64")
    digest.update(canonical.numpy().tobytes(order="C"))
    return digest.hexdigest()


def build_model_cache_identity(model_path: str | Path) -> dict:
    """Build a checkpoint-path-independent tokenizer/config identity."""

    checkpoint = Path(model_path).expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {checkpoint}")
    tokenizer_files = []
    tokenizer_digest = hashlib.sha256()
    for name in _TOKENIZER_FILE_NAMES:
        path = checkpoint / name
        if not path.is_file():
            continue
        file_digest = _file_sha256(path)
        tokenizer_files.append(
            {
                "name": name,
                "size_bytes": int(path.stat().st_size),
                "sha256": file_digest,
            }
        )
        tokenizer_digest.update(name.encode("utf-8"))
        tokenizer_digest.update(bytes.fromhex(file_digest))
    if not tokenizer_files:
        raise FileNotFoundError(f"checkpoint has no recognized tokenizer files: {checkpoint}")

    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"checkpoint config does not exist: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    text_config = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}
    tokenization_config = {
        field: text_config.get(field, config.get(field))
        for field in _TOKENIZATION_CONFIG_FIELDS
    }
    return {
        "schema_version": 1,
        "checkpoint_path": str(checkpoint),
        "tokenizer_files": tokenizer_files,
        "tokenizer_sha256": tokenizer_digest.hexdigest(),
        "tokenization_config": tokenization_config,
        "tokenization_config_sha256": _canonical_json_sha256(tokenization_config),
    }


def validate_model_cache_compatibility(identity: dict, model_path: str | Path) -> dict:
    """Require tokenizer/config identity while allowing structural MoE changes."""

    current = build_model_cache_identity(model_path)
    if current["tokenizer_sha256"] != identity.get("tokenizer_sha256"):
        raise ValueError("checkpoint tokenizer is incompatible with the frozen token cache.")
    if current["tokenization_config_sha256"] != identity.get("tokenization_config_sha256"):
        raise ValueError("checkpoint tokenization config is incompatible with the frozen token cache.")
    return current


def validate_calibration_token_cache_payload(
    payload: dict,
    *,
    required_sequence_length: int = 2048,
    model_path: str | Path | None = None,
    require_identity: bool = False,
) -> torch.Tensor:
    """Validate one frozen train-only token stream shared by pruning methods."""

    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("Unsupported calibration token cache schema_version.")
    if payload.get("purpose") != "shared_moe_pruning_calibration":
        raise ValueError("calibration token cache has an unsupported purpose.")
    if payload.get("split") != "train":
        raise ValueError("calibration token cache must use the train split.")
    if payload.get("frozen_before_profile") is not True:
        raise ValueError("calibration token cache must be frozen before profile construction.")
    if payload.get("test_metrics_used") is not False:
        raise ValueError("calibration token cache must not use test metrics.")
    sequence_length = int(payload.get("sequence_length", -1))
    if sequence_length != int(required_sequence_length):
        raise ValueError(
            f"calibration token cache must use sequence_length={required_sequence_length}."
        )
    tokens = payload.get("input_ids")
    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2 or int(tokens.shape[0]) != 1:
        raise ValueError("calibration input_ids must have shape [1, tokens].")
    if tokens.dtype not in (torch.int32, torch.int64):
        raise ValueError("calibration input_ids must use an integer dtype.")
    expected_sequences = int(payload.get("calibration_sequences", -1))
    expected_tokens = int(payload.get("calibration_tokens", -1))
    if expected_sequences <= 0 or expected_tokens != expected_sequences * sequence_length:
        raise ValueError("calibration sequence and token counts are inconsistent.")
    if expected_tokens != int(tokens.shape[1]):
        raise ValueError("calibration token count does not match input_ids.")
    if payload.get("input_ids_sha256") != token_tensor_sha256(tokens):
        raise ValueError("calibration input_ids_sha256 does not match input_ids.")
    if payload.get("attention_mask_semantics") != "all_ones_no_padding":
        raise ValueError("calibration cache must use all_ones_no_padding attention masks.")
    source = payload.get("source")
    if not isinstance(source, dict) or source.get("split") != "train":
        raise ValueError("calibration cache must include train source provenance.")
    arrow_files = source.get("arrow_files")
    if arrow_files:
        if not isinstance(arrow_files, list) or not all(
            isinstance(item, dict) and len(str(item.get("sha256", ""))) == 64
            for item in arrow_files
        ):
            raise ValueError("calibration Arrow files must include SHA256 provenance.")
    identity = payload.get("model_identity")
    if require_identity and not isinstance(identity, dict):
        raise ValueError("calibration token cache must include model_identity.")
    if model_path is not None:
        if not isinstance(identity, dict):
            raise ValueError("calibration token cache has no model_identity.")
        validate_model_cache_compatibility(identity, model_path)
    return tokens.detach().to(dtype=torch.long, device="cpu")


def calibration_batches_from_payload(
    payload: dict,
    *,
    required_sequence_length: int = 2048,
    model_path: str | Path | None = None,
    require_identity: bool = False,
) -> list[dict[str, torch.Tensor]]:
    """Split a validated shared calibration stream into exact model batches."""

    tokens = validate_calibration_token_cache_payload(
        payload,
        required_sequence_length=required_sequence_length,
        model_path=model_path,
        require_identity=require_identity,
    )
    sequence_length = int(required_sequence_length)
    batches = []
    for begin in range(0, int(tokens.shape[1]), sequence_length):
        input_ids = tokens[:, begin : begin + sequence_length].contiguous()
        batches.append(
            {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids, dtype=torch.long),
            }
        )
    return batches


def load_shared_calibration_tokens(
    cache_path: str | Path,
    *,
    required_sequence_length: int,
    model_path: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, dict]:
    """Load the shared train-only token artifact without retokenizing source data."""

    path = Path(cache_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"shared calibration token cache does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    tokens = validate_calibration_token_cache_payload(
        payload,
        required_sequence_length=required_sequence_length,
        model_path=model_path,
        require_identity=model_path is not None,
    )
    return tokens.to(device), {
        "source_type": "shared_calibration_token_cache",
        "path": str(path),
        "cache_file_sha256": _file_sha256(path),
        "protocol_name": payload.get("protocol_name"),
        "dataset": payload.get("dataset"),
        "config": payload.get("dataset_config"),
        "dataset_config": payload.get("dataset_config"),
        "split": payload.get("split"),
        "text_field": payload.get("text_field"),
        "sequence_length": payload.get("sequence_length"),
        "calibration_sequences": payload.get("calibration_sequences"),
        "calibration_tokens": payload.get("calibration_tokens"),
        "input_ids_sha256": payload.get("input_ids_sha256"),
        "source": payload.get("source"),
        "token_stream": payload.get("token_stream"),
    }


def load_calibration_text_dataset(
    *,
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    text_field: str,
    arrow_files: Sequence[Path] = (),
    require_train: bool = True,
):
    """Load a train-only text corpus and return auditable source provenance.

    Explicit Arrow inputs are useful for offline experiments: their order is
    preserved and each file is content-hashed.  When no Arrow input is given,
    Hugging Face ``load_dataset`` is used with the explicit dataset/config/split.
    """

    from datasets import Dataset, concatenate_datasets, load_dataset

    name = str(dataset_name).strip()
    config = None if dataset_config is None else str(dataset_config).strip() or None
    split_name = str(split).strip()
    field = str(text_field).strip()
    if not name:
        raise ValueError("dataset_name must be non-empty.")
    if require_train and split_name != "train":
        raise ValueError("calibration must use the train split.")
    if not field:
        raise ValueError("text_field must be non-empty.")

    resolved_files = [Path(path).expanduser().resolve() for path in arrow_files]
    arrow_provenance = []
    if resolved_files:
        shards = []
        for path in resolved_files:
            if not path.is_file():
                raise FileNotFoundError(f"calibration Arrow file does not exist: {path}")
            shards.append(Dataset.from_file(str(path)))
            arrow_provenance.append(
                {
                    "path": str(path),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": _file_sha256(path),
                }
            )
        dataset = shards[0] if len(shards) == 1 else concatenate_datasets(shards)
        source_type = "arrow_files"
    else:
        dataset = load_dataset(name, config, split=split_name)
        source_type = "huggingface_dataset"

    if field not in dataset.column_names:
        raise ValueError(
            f"calibration text field {field!r} is absent; "
            f"available fields: {dataset.column_names}"
        )
    provenance = {
        "dataset": name,
        "config": config,
        "split": split_name,
        "text_field": field,
        "source_type": source_type,
        "num_rows": int(len(dataset)),
        "arrow_files": arrow_provenance,
    }
    return dataset, provenance


def collect_contiguous_text_tokens(
    tokenizer,
    dataset,
    *,
    text_field: str,
    total_tokens: int,
    token_offset: int = 0,
    separator: str = "\n",
    row_batch_size: int = 1024,
) -> tuple[torch.Tensor, dict]:
    """Materialize one deterministic prefix token stream and slice an interval.

    Documents are joined in dataset order.  The accumulated prefix is retokenized
    after fixed-size row batches until it covers ``offset + total``.  Retokenizing
    the whole prefix avoids tokenizer-boundary drift between calibration folds.
    """

    field = str(text_field).strip()
    if field not in dataset.column_names:
        raise ValueError(
            f"calibration text field {field!r} is absent; "
            f"available fields: {dataset.column_names}"
        )
    count = int(total_tokens)
    offset = int(token_offset)
    batch_size = int(row_batch_size)
    if count <= 0:
        raise ValueError("total_tokens must be positive.")
    if offset < 0:
        raise ValueError("token_offset must be non-negative.")
    if batch_size <= 0:
        raise ValueError("row_batch_size must be positive.")
    required = offset + count
    documents: list[str] = []
    token_ids: list[int] = []
    rows_consumed = 0
    tokenizer.model_max_length = 2**31 - 1
    for begin in range(0, len(dataset), batch_size):
        end = min(begin + batch_size, len(dataset))
        documents.extend(str(value) for value in dataset[begin:end][field])
        rows_consumed = end
        encoded = tokenizer(
            separator.join(documents),
            truncation=False,
            add_special_tokens=False,
        )
        token_ids = encoded["input_ids"]
        if token_ids and isinstance(token_ids[0], list):
            if len(token_ids) != 1:
                raise ValueError("tokenizer must return a single token sequence.")
            token_ids = token_ids[0]
        if len(token_ids) >= required:
            break
    if len(token_ids) < required:
        raise ValueError("dataset does not contain enough calibration tokens.")
    selected = torch.tensor(token_ids[offset:required], dtype=torch.long).view(1, -1)
    metadata = {
        "tokenization_strategy": "joined_documents",
        "add_special_tokens": False,
        "separator": separator.encode("unicode_escape").decode("ascii"),
        "row_batch_size": batch_size,
        "rows_consumed": int(rows_consumed),
        "materialized_prefix_tokens": int(len(token_ids)),
        "token_offset": offset,
        "token_end": required,
        "selected_tokens": count,
    }
    return selected, metadata
