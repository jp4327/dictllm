#!/usr/bin/env python
"""Evaluate converted identity NVFP4 with an explicitly selected runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .packed_nvfp4 import load_packed_model
from .verify_checkpoint import write_json


@torch.no_grad()
def runtime_smoke(model):
    """Exercise both prompt-like and single-token shapes without dataset access."""
    device = next(model.parameters()).device
    results = []
    for length in (8, 1):
        ids = torch.arange(length, device=device).reshape(1, -1) % model.config.vocab_size
        logits = model(input_ids=ids, use_cache=False).logits
        if not torch.isfinite(logits).all():
            raise RuntimeError("Runtime smoke produced non-finite logits.")
        results.append({"input_length": length, "output_shape": list(logits.shape), "finite": True})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--runtime", choices=("dequantize", "qutlass"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--cache-dequantized", action="store_true",
                        help="Faster control evaluation but caches dense weights; NOT a low-memory benchmark.")
    parser.add_argument("--smoke-only", action="store_true", help="No tokenizer, data, or PPL; test two sequence lengths.")
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=2048)
    parser.add_argument("--dataset-cache-dir", type=Path,
                        default=Path(__file__).resolve().parent / "cache/huggingface")
    parser.add_argument("--output", type=Path, help="New report path; defaults to a runtime-specific file.")
    args = parser.parse_args()
    if args.context_length < 2 or not 1 <= args.stride <= args.context_length:
        parser.error("Need context length >=2 and stride within [1, context length].")
    if args.runtime == "qutlass" and args.cache_dequantized:
        parser.error("--cache-dequantized is only for the dequantize control.")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not (checkpoint / "config.json").exists() and (checkpoint / "wikitext/config.json").exists():
        checkpoint /= "wikitext"
    kind = "runtime_smoke" if args.smoke_only else "final_ppl_wikitext"
    output = args.output or checkpoint / f"{kind}_{args.runtime}_{args.dtype}.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}.")
    dtype = None if args.dtype == "auto" else getattr(torch, args.dtype)
    model = load_packed_model(checkpoint, runtime=args.runtime, device=args.device,
                              dtype=dtype, cache_dequantized=args.cache_dequantized)
    compute_dtype = str(next(model.parameters()).dtype)
    report = {"checkpoint": str(checkpoint), "runtime": args.runtime, "dtype": args.dtype,
              "resolved_model_dtype": compute_dtype,
              "transform_class": "identity", "activation_quantization": args.runtime == "qutlass",
              "activation_format": "nvfp4" if args.runtime == "qutlass" else "floating_point",
              "activation_packing_backend": "torch_reference_no_rotation" if args.runtime == "qutlass" else None,
              "output_kernel_dtype": "torch.bfloat16" if args.runtime == "qutlass" else compute_dtype,
              "cache_dequantized": args.cache_dequantized,
              "context_length": args.context_length, "stride": args.stride,
              "perplexity_aggregation": "legacy_unweighted_mean_of_window_losses"}
    if args.smoke_only:
        report["checks"] = runtime_smoke(model)
    else:
        # Keep the zero-preserving experiment's tokenizer and loss aggregation.
        from .checkpoint_adapter import load_tokenizer
        from .run_quantization import compute_ppl, prepare_data
        config_data = model.config.to_dict()
        tokenizer = load_tokenizer(checkpoint, config_data)
        args.eval_perplexity = True
        _, test = prepare_data(tokenizer, args, need_calibration=False)
        report["perplexity"] = compute_ppl(model, test, args)
        report["dataset_name"] = "wikitext"
        print(f"perplexity={report['perplexity']:.4f}")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    print(f"saved_report={output}")


if __name__ == "__main__":
    main()
