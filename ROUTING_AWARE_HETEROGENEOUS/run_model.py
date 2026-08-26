"""Build pruning artifacts from a real Hugging Face MoE model, without evaluation."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from .adapter import adapter_from_model
from .config import MethodConfig
from .core import RoutingAwarePruner


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--natural-sequences", type=int, default=96)
    parser.add_argument("--guided-sequences", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--calibration-batch-size", type=int, default=1)
    parser.add_argument("--generation-batch-size", type=int, default=1)
    parser.add_argument("--guided-batch-size", type=int, default=1)
    parser.add_argument("--retention", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    return parser.parse_args()


def _dtype(name: str) -> torch.dtype:
    """Map a CLI dtype name to a PyTorch dtype."""

    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def _natural_sequences(
    model: torch.nn.Module,
    tokenizer: object,
    count: int,
    length: int,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Generate self-sampled calibration sequences only."""

    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is None:
        bos = getattr(tokenizer, "eos_token_id", None)
    if bos is None:
        raise ValueError("tokenizer needs bos_token_id or eos_token_id")
    batches = []
    total_tokens = 0
    started_at = time.perf_counter()
    previous_elapsed = 0.0
    for start in range(0, count, batch_size):
        current = min(batch_size, count - start)
        prompts = torch.full((current, 1), int(bos), dtype=torch.long, device=device)
        attention_mask = torch.ones_like(prompts)
        print(
            f"[natural-generation-start] batch={start // batch_size + 1} "
            f"sequences={start + 1}-{start + current}/{count} "
            f"target_length={length} batch_size={current}",
            flush=True,
        )
        with torch.inference_mode():
            generated = model.generate(
                input_ids=prompts,
                attention_mask=attention_mask,
                max_new_tokens=max(1, length - 1),
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
                use_cache=True,
                pad_token_id=getattr(tokenizer, "pad_token_id", None) or int(bos),
            )[:, :length]
        batches.append(generated.cpu())
        total_tokens += int(generated.shape[0] * generated.shape[1])
        elapsed = max(time.perf_counter() - started_at, 1.0e-6)
        batch_elapsed = elapsed if start == 0 else elapsed - previous_elapsed
        batch_tokens = int(generated.shape[0] * generated.shape[1])
        batch_rate = batch_tokens / max(batch_elapsed, 1.0e-6)
        overall_rate = total_tokens / elapsed
        print(
            f"[natural-generation] batch={start // batch_size + 1} "
            f"sequences={min(start + current, count)}/{count} "
            f"tokens={total_tokens} batch_tokens_per_sec={batch_rate:.2f} "
            f"overall_tokens_per_sec={overall_rate:.2f} elapsed_sec={elapsed:.1f}",
            flush=True,
        )
        previous_elapsed = elapsed
        del attention_mask, generated, prompts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return torch.cat(batches, dim=0)


def main() -> int:
    """Load a model, build artifacts, and never run benchmark inference."""

    args = _parse_args()
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA is unavailable; pass --device cpu explicitly")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    if device.type == "cuda":
        device_index = torch.cuda.current_device() if device.index is None else device.index
        properties = torch.cuda.get_device_properties(device_index)
        print(
            f"[runtime] torch={torch.__version__} cuda={torch.version.cuda} "
            f"device={properties.name} capability={properties.major}.{properties.minor} "
            f"memory_gib={properties.total_memory / 2**30:.1f}",
            flush=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    load_kwargs = {
        "torch_dtype": _dtype(args.dtype),
        "trust_remote_code": True,
    }
    if device.type != "cuda" or torch.cuda.device_count() <= 1:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **load_kwargs).to(device).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map={"": str(device)},
            **load_kwargs,
        ).eval()
    adapter = adapter_from_model(model)
    config = MethodConfig(
        natural_sequences=args.natural_sequences,
        guided_sequences=args.guided_sequences,
        sequence_length=args.sequence_length,
        calibration_batch_size=args.calibration_batch_size,
        generation_batch_size=args.generation_batch_size,
        guided_batch_size=args.guided_batch_size,
        retention=args.retention,
        device=str(device),
        dtype=args.dtype,
    )
    input_ids = _natural_sequences(
        model,
        tokenizer,
        args.natural_sequences,
        args.sequence_length,
        device,
        args.generation_batch_size,
    )
    result = RoutingAwarePruner(adapter, config).run(input_ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    print(f"Saved pruning artifacts to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())