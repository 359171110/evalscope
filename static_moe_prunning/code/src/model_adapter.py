from __future__ import annotations

import os
import sys
from contextlib import contextmanager, nullcontext
from inspect import signature

import torch

from .model_families import resolve_model_family


def clear_hf_proxy_env() -> None:
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(name, None)


def apply_torch_autocast_compat_patch() -> None:
    try:
        parameter_count = len(signature(torch.is_autocast_enabled).parameters)
    except (TypeError, ValueError):
        parameter_count = 0
    if parameter_count >= 1:
        return
    original = torch.is_autocast_enabled

    def compat_is_autocast_enabled(device_type=None):
        del device_type
        return original()

    torch.is_autocast_enabled = compat_is_autocast_enabled


def load_qwen3_moe(
    model_path: str,
    device_map=None,
    model_family: str | None = None,
    device: str | None = None,
):
    clear_hf_proxy_env()
    apply_torch_autocast_compat_patch()
    extra_site_packages = os.environ.get("MOE_EXTRA_SITE_PACKAGES")
    if extra_site_packages and extra_site_packages not in sys.path:
        sys.path.append(extra_site_packages)
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    family = resolve_model_family(model_path=model_path, model_family=model_family)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model_kwargs = {
        "torch_dtype": "auto",
        "trust_remote_code": True,
    }
    if family == "qwen3":
        model_class = AutoModelForCausalLM
    elif family == "qwen3.6":
        model_class = getattr(transformers, "Qwen3_5MoeForConditionalGeneration", None)
    else:
        model_class = getattr(transformers, "Gemma4ForConditionalGeneration", None)
    if model_class is None:
        raise ImportError(f"The installed transformers build has no model class for {family}.")
    if device_map == "none":
        model = model_class.from_pretrained(model_path, **model_kwargs)
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(resolved_device)
    else:
        if device_map is None:
            device_map = {"": "cuda:0"} if torch.cuda.is_available() else {"": "cpu"}
        model_kwargs["device_map"] = device_map
        model = model_class.from_pretrained(model_path, **model_kwargs)
    model.eval()
    return model, tokenizer


@contextmanager
def maybe_bf16_autocast():
    if not torch.cuda.is_available():
        with nullcontext():
            yield
        return
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        yield