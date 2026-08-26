"""Build pruning artifacts from a real Hugging Face MoE model, without evaluation."""

from __future__ import annotations

import argparse
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
    for start in range(0, count, batch_size):
        current = min(batch_size, count - start)
        prompts = torch.full((current, 1), int(bos), dtype=torch.long, device=device)
        with torch.inference_mode():
            generated = model.generate(
                input_ids=prompts,
                max_new_tokens=max(1, length - 1),
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
                use_cache=True,
                pad_token_id=getattr(tokenizer, "pad_token_id", None) or int(bos),
            )[:, :length]
        batches.append(generated.cpu())
        del generated, prompts
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
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=_dtype(args.dtype),
        trust_remote_code=True,
        device_map={"": str(device)},
    ).eval()
    adapter = adapter_from_model(model)
    config = MethodConfig(
        natural_sequences=args.natural_sequences,
        guided_sequences=args.guided_sequences,
        sequence_length=args.sequence_length,
        calibration_batch_size=args.calibration_batch_size,
        generation_batch_size=args.generation_batch_size,
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