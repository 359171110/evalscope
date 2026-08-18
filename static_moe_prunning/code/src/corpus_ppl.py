from __future__ import annotations

import math
from typing import Mapping

import torch

from .calibration_data import token_tensor_sha256, validate_model_cache_compatibility
from .model_adapter import maybe_bf16_autocast


def frozen_protocol_matches(
    metrics: Mapping[str, object],
    payload: Mapping[str, object],
    *,
    max_windows: int | None,
) -> bool:
    """Return whether an evaluation exactly consumed its frozen token protocol."""

    return (
        max_windows is None
        and int(metrics.get("windows", -1))
        == int(payload.get("evaluation_windows", -2))
        and int(metrics.get("tokens", -1))
        == int(payload.get("evaluation_tokens", -2))
    )


def validate_token_cache_payload(
    payload: Mapping[str, object],
    *,
    required_sequence_length: int = 2048,
    model_path: str | None = None,
    require_identity: bool = False,
) -> torch.Tensor:
    """Validate a frozen, metric-independent corpus token cache."""

    if payload.get("frozen_before_evaluation") is not True:
        raise ValueError("evaluation token cache must be frozen before evaluation.")
    if payload.get("test_metrics_used") is not False:
        raise ValueError("evaluation token cache must be independent of test metrics.")
    if int(payload.get("sequence_length", -1)) != int(required_sequence_length):
        raise ValueError(
            f"evaluation token cache must use sequence_length={required_sequence_length}."
        )
    tokens = payload.get("input_ids")
    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2 or tokens.shape[0] != 1:
        raise ValueError("input_ids must be a tensor with shape [1, tokens].")
    if tokens.dtype not in (torch.int32, torch.int64):
        raise ValueError("input_ids must use an integer dtype.")
    expected_tokens = int(payload.get("evaluation_tokens", -1))
    if expected_tokens != int(tokens.shape[1]):
        raise ValueError("evaluation_tokens must equal the frozen token tensor length.")
    if expected_tokens <= 0:
        raise ValueError("evaluation token cache must contain tokens.")
    token_digest = payload.get("input_ids_sha256")
    if token_digest is not None and token_digest != token_tensor_sha256(tokens):
        raise ValueError("input_ids_sha256 does not match the frozen token tensor.")
    if require_identity and token_digest is None:
        raise ValueError("formal token cache must include input_ids_sha256.")
    mask_semantics = payload.get("attention_mask_semantics")
    if mask_semantics is not None and mask_semantics != "all_ones_no_padding":
        raise ValueError("unsupported attention_mask_semantics.")
    if require_identity and mask_semantics is None:
        raise ValueError("formal token cache must include attention_mask_semantics.")
    identity = payload.get("model_identity")
    if require_identity and not isinstance(identity, Mapping):
        raise ValueError("formal token cache must include model_identity.")
    if model_path is not None:
        if not isinstance(identity, Mapping):
            raise ValueError("token cache has no model_identity for compatibility validation.")
        validate_model_cache_compatibility(dict(identity), model_path)
    source = payload.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("evaluation token cache must include source provenance.")
    arrow_files = source.get("arrow_files")
    if arrow_files:
        if not isinstance(arrow_files, list) or not all(
            isinstance(item, Mapping) and len(str(item.get("sha256", ""))) == 64
            for item in arrow_files
        ):
            raise ValueError("evaluation Arrow files must include SHA256 provenance.")
    return tokens.detach().to(dtype=torch.long, device="cpu")


class FrozenTokenCorpusPerplexity:
    """Corpus perplexity over an immutable, pre-tokenized sequence."""

    def __init__(self, model, token_ids: torch.Tensor):
        if token_ids.ndim != 2 or int(token_ids.shape[0]) != 1:
            raise ValueError("token_ids must have shape [1, tokens].")
        if int(token_ids.shape[1]) <= 0:
            raise ValueError("token_ids must not be empty.")
        self.model = model
        self.token_ids = token_ids.detach().to(dtype=torch.long, device="cpu")
        self._tokens_by_device: dict[str, torch.Tensor] = {}

    def _resolve_device(self):
        if hasattr(self.model, "device"):
            return self.model.device
        if hasattr(self.model, "hf_device_map"):
            for mapped in self.model.hf_device_map.values():
                if mapped not in ("cpu", "disk"):
                    return mapped
        return next(self.model.parameters()).device

    def _tokens_for_device(self, device) -> torch.Tensor:
        key = str(device)
        if key not in self._tokens_by_device:
            self._tokens_by_device[key] = self.token_ids.to(device)
        return self._tokens_by_device[key]

    def calculate_corpus_ppl(
        self,
        *,
        n_ctx: int = 2048,
        max_windows: int | None = None,
    ) -> dict[str, float | int]:
        context = int(n_ctx)
        if context <= 1:
            raise ValueError("n_ctx must be greater than one.")
        if max_windows is not None and int(max_windows) <= 0:
            raise ValueError("max_windows must be positive when provided.")
        tokens = self._tokens_for_device(self._resolve_device())
        sequence_tokens = int(tokens.shape[1])
        nll_sum = 0.0
        token_count = 0
        window_count = 0
        for begin in range(0, sequence_tokens, context):
            if max_windows is not None and window_count >= int(max_windows):
                break
            end = min(begin + context, sequence_tokens)
            length = end - begin
            input_ids = tokens[:, begin:end]
            labels = input_ids.clone()
            with torch.inference_mode(), maybe_bf16_autocast():
                outputs = self.model(input_ids, labels=labels, use_cache=False)
            nll_sum += float(outputs.loss.detach().float().item()) * length
            token_count += length
            window_count += 1
        ppl = math.exp(nll_sum / token_count) if token_count else 0.0
        return {
            "ppl": float(ppl),
            "windows": int(window_count),
            "tokens": int(token_count),
        }


class FullWikiTextPerplexity:
    """Evaluate raw WikiText-2 without chat templating or prompt formatting."""

    def __init__(
        self,
        model,
        tokenizer,
        split: str = "test",
        text_column: str = "text",
        min_text_length: int = 512,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.split = split
        self.text_column = text_column
        self.min_text_length = int(min_text_length)
        self._tokens_by_device: dict[str, torch.Tensor] = {}
        self.text = self._prepare_text()

    def _prepare_text(self) -> str:
        from datasets import load_dataset

        data = load_dataset("wikitext", "wikitext-2-raw-v1", split=self.split)
        texts = []
        for sample in data:
            text = sample[self.text_column]
            if len(text) >= self.min_text_length:
                texts.append(" \n" if text == "" else text)
        return "".join(texts)

    def _tokenize_for_device(self, device) -> torch.Tensor:
        key = str(device)
        if key not in self._tokens_by_device:
            self.tokenizer.model_max_length = 2**31 - 1
            self._tokens_by_device[key] = self.tokenizer(
                self.text,
                truncation=False,
                return_tensors="pt",
            ).input_ids.to(device)
        return self._tokens_by_device[key]

    def calculate_corpus_ppl(
        self,
        n_ctx: int = 2048,
        max_windows: int | None = None,
    ) -> dict[str, float | int]:
        if int(n_ctx) != 2048:
            raise ValueError("Repository protocol requires sequence_length=2048.")
        device = self.model.device if hasattr(self.model, "device") else next(self.model.parameters()).device
        tokens = self._tokenize_for_device(device)
        nll_sum = 0.0
        token_count = 0
        window_count = 0
        for begin in range(0, int(tokens.shape[1]), int(n_ctx)):
            if max_windows is not None and window_count >= int(max_windows):
                break
            end = min(begin + int(n_ctx), int(tokens.shape[1]))
            input_ids = tokens[:, begin:end]
            labels = input_ids.clone()
            with torch.inference_mode(), maybe_bf16_autocast():
                outputs = self.model(input_ids, labels=labels, use_cache=False)
            length = int(end - begin)
            nll_sum += float(outputs.loss.detach().float().item()) * length
            token_count += length
            window_count += 1
        return {
            "ppl": float(math.exp(nll_sum / token_count)) if token_count else 0.0,
            "windows": int(window_count),
            "tokens": int(token_count),
        }
