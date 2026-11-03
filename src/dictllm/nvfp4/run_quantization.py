#!/usr/bin/env python
"""Quantize native DictLLM LLaMA factors without rotation."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, Subset

from .checkpoint_adapter import factor_modules, load_tokenizer, model_class
from .quantization import collect_hessian, quantize_weight, reconstruction_errors
from .verify_checkpoint import verify_checkpoint, write_json


HERE = Path(__file__).resolve().parent


def resolve_checkpoint(value):
    path = Path(value).expanduser().resolve()
    if (path / "config.json").is_file():
        return path
    if (path / "wikitext/config.json").is_file():
        return path / "wikitext"
    raise FileNotFoundError(f"No checkpoint config.json under {path} or its wikitext subfolder.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", "--source_model_path", required=True)
    parser.add_argument("--output-path", "--save_path", type=Path, required=True)
    parser.add_argument("--method", choices=("rtn", "masked_gptq"), default="masked_gptq")
    parser.add_argument("--quant-targets", "--quant_targets", default="basis,coefficient")
    parser.add_argument("--w-group-size", type=int, default=16)
    parser.add_argument("--scale-precision", choices=("e4m3", "fp32"), default="e4m3")
    parser.add_argument("--blocksize", type=int, default=128)
    parser.add_argument("--rel-damp", type=float, default=.1)
    parser.add_argument("--calibration-size", type=int, default=16)
    parser.add_argument("--calibration-batch-size", type=int, default=1)
    parser.add_argument("--calibration-selection", choices=("first", "random"), default="first")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=2048)
    parser.add_argument("--dataset-cache-dir", type=Path, default=HERE / "cache/huggingface")
    parser.add_argument("--no-activation-metrics", action="store_true",
                        help="Skip RTN Hessian collection; GPTQ still requires calibration.")
    parser.add_argument("--eval-perplexity", action="store_true")
    parser.add_argument("--max-modules", type=int, help="Debug only: export a partially quantized model.")
    parser.add_argument("--max-shard-size", default="4GB")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if min(args.w_group_size, args.blocksize, args.calibration_size,
           args.calibration_batch_size, args.context_length, args.stride) < 1:
        parser.error("Group, block, calibration, context, and stride sizes must be positive.")
    if args.context_length < 2 or args.stride > args.context_length:
        parser.error("Need context length >=2 and stride <= context length.")
    if not 0 <= args.rel_damp < float("inf"):
        parser.error("Damping must be finite and nonnegative.")
    if args.max_modules is not None and args.max_modules < 1:
        parser.error("--max-modules must be positive.")
    args.quant_targets = [value.strip() for value in args.quant_targets.split(",")]
    if not args.quant_targets or not set(args.quant_targets) <= {"basis", "coefficient"}:
        parser.error("Quantization targets must be basis, coefficient, or basis,coefficient.")
    return args


def load_model_and_tokenizer(source):
    config = json.loads((source / "config.json").read_text())
    cls = model_class(config)
    import transformers
    dtype_key = "dtype" if int(transformers.__version__.split(".", 1)[0]) >= 5 else "torch_dtype"
    model = cls.from_pretrained(source, device_map="cpu", **{dtype_key: "auto"})
    tokenizer = load_tokenizer(source, config)
    model.eval().requires_grad_(False)
    return model, tokenizer


def prepare_data(tokenizer, args, need_calibration):
    from datasets import load_dataset
    from transformers import default_data_collator
    # Reuse the original tokenization/grouping functions, not its incompatible
    # dataset_cache_dir keyword or previously processed Arrow caches.
    from .calibration_data import group_text, tokenize_func
    from functools import partial

    loader, test = None, None
    args.dataset_cache_dir.mkdir(parents=True, exist_ok=True)
    if need_calibration:
        train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train",
                             cache_dir=str(args.dataset_cache_dir))
        train = train.map(partial(tokenize_func, tokenizer=tokenizer, content="text"),
                          num_proc=4, batched=True, remove_columns="text")
        train = train.map(partial(group_text, context_length=args.context_length),
                          num_proc=4, batch_size=1024, batched=True)
        if len(train) < args.calibration_size:
            raise ValueError("Not enough full calibration sequences for the requested size.")
        if args.calibration_selection == "first":
            indices = list(range(args.calibration_size))
        else:
            indices = torch.randperm(len(train), generator=torch.Generator().manual_seed(args.seed))[
                :args.calibration_size].tolist()
        loader = DataLoader(Subset(train, indices), batch_size=args.calibration_batch_size,
                            shuffle=False, collate_fn=default_data_collator, num_workers=0)
    if args.eval_perplexity:
        test_raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="test",
                                cache_dir=str(args.dataset_cache_dir))
        test = tokenizer("\n\n".join(test_raw["text"]), return_tensors="pt")
    return loader, test


@torch.no_grad()
def compute_ppl(model, test, args):
    from tqdm import tqdm

    losses = []
    previous_end = 0
    length = test.input_ids.shape[1]
    for begin in tqdm(range(0, length, args.stride), desc="WikiText-2 PPL"):
        end = min(begin + args.context_length, length)
        ids = test.input_ids[:, begin:end].to(args.device)
        target_length = end - previous_end
        if ids.shape[1] < 2:
            break
        labels = ids.clone()
        labels[:, :-target_length] = -100
        loss = model(input_ids=ids, labels=labels, use_cache=False).loss.float()
        if not torch.isfinite(loss):
            raise RuntimeError("Perplexity evaluation produced a non-finite loss.")
        losses.append(loss.cpu())
        previous_end = end
        if end == length:
            break
    if not losses:
        raise ValueError("No evaluable WikiText-2 windows.")
    # Deliberately match the original 12_* unweighted mean of window losses.
    ppl = float(torch.exp(torch.stack(losses).mean()))
    if not torch.isfinite(torch.tensor(ppl)):
        raise RuntimeError("Perplexity overflowed.")
    return ppl


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def quantize_model_inplace(model, loader, args):
    """Reuse the identity quantizer without checkpoint I/O or PPL evaluation."""
    device = torch.device(args.device)
    model.eval().requires_grad_(False).to(device)
    modules = list(factor_modules(model, set(args.quant_targets)))
    if args.max_modules is not None:
        modules = modules[:args.max_modules]
    if not modules:
        raise ValueError("No matching Basis/Coefficient factors.")
    needs_hessian = args.method == "masked_gptq" or not args.no_activation_metrics
    summary = {
        "quant_method": args.method, "format": "simulated_nvfp4", "w_bits": 4,
        "packed_storage": False, "activation_quantization": False,
        "transform_class": "identity", "preserves_sparse_mask": True,
        "scale_policy": "tensor_global_normalized_block_amax",
        "underflow_policy": "nonzero_block_scale_clamped_to_min_e4m3_subnormal",
        "w_group_size": args.w_group_size, "scale_precision": args.scale_precision,
        "blocksize": args.blocksize, "rel_damp": args.rel_damp,
        "source_checkpoint": str(args.source_checkpoint), "output_path": str(args.output_path),
        "quant_targets": args.quant_targets, "calibration_size": args.calibration_size if needs_hessian else 0,
        "calibration_batch_size": args.calibration_batch_size,
        "calibration_selection": args.calibration_selection, "seed": args.seed,
        "context_length": args.context_length, "stride": args.stride,
        "partial_quantization": args.max_modules is not None,
        "activation_metric_reference": "module inputs in progressively quantized model",
        "perplexity_aggregation": "legacy_unweighted_mean_of_window_losses",
        "torch_version": torch.__version__, "per_tensor": [],
    }
    for index, (name, module, kind, aliases) in enumerate(modules, 1):
        print(f"[{index}/{len(modules)}] {args.method}: {name} {tuple(module.weight.shape)}", flush=True)
        original = module.weight.detach().clone()
        synchronize(device)
        start = time.perf_counter()
        H, tokens = collect_hessian(model, module, loader, device) if needs_hessian else (None, 0)
        synchronize(device)
        calibration_time = time.perf_counter() - start
        start = time.perf_counter()
        quantized, stats = quantize_weight(
            original, method=args.method, hessian=H, group_size=args.w_group_size,
            scale_precision=args.scale_precision, blocksize=args.blocksize, rel_damp=args.rel_damp,
        )
        synchronize(device)
        quantization_time = time.perf_counter() - start
        errors = reconstruction_errors(original, quantized, None if args.no_activation_metrics else H)
        with torch.no_grad():
            module.weight.copy_(quantized)
        summary["per_tensor"].append({"name": name, "aliases": aliases, "tensor_type": kind,
                                      "shape": list(original.shape), "dtype": str(original.dtype),
                                      "calibration_activation_rows": tokens,
                                      "calibration_runtime_sec": calibration_time,
                                      "quantization_runtime_sec": quantization_time, **stats, **errors})
        print(f"  sparsity={stats['sparsity_before']:.6f}->{stats['sparsity_after']:.6f} "
              f"revived_zeros={stats['revived_zeros']} quantization_sec={quantization_time:.3f}", flush=True)
        del original, quantized, H
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    summary["num_tensors"] = len(modules)
    summary["num_parameters"] = sum(record["num_parameters"] for record in summary["per_tensor"])
    summary["zero_counts_by_target"] = {
        kind: {field: sum(record[field] for record in summary["per_tensor"] if record["tensor_type"] == kind)
               for field in ["num_parameters", "zeros_before", "zeros_after", "revived_zeros", "additional_zeros"]}
        for kind in args.quant_targets
    }
    return summary


def main(argv=None):
    args = parse_args(argv)
    source = resolve_checkpoint(args.source_checkpoint)
    output = args.output_path.expanduser().resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Output and source must be separate, non-nested directories.")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}. Choose a new output path.")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    model, tokenizer = load_model_and_tokenizer(source)
    if not list(factor_modules(model, set(args.quant_targets))):
        raise ValueError("No matching Basis/Coefficient factors.")
    needs_hessian = args.method == "masked_gptq" or not args.no_activation_metrics
    loader, test = prepare_data(tokenizer, args, needs_hessian)
    args.source_checkpoint, args.output_path = str(source), output
    summary = quantize_model_inplace(model, loader, args)
    output.mkdir(parents=True, exist_ok=False)
    # Legacy sharing has duplicate state-dict aliases; keep the established bin format.
    model.save_pretrained(output, safe_serialization=False, max_shard_size=args.max_shard_size)
    tokenizer.save_pretrained(output)
    write_json(output / "zero_preserving_summary.json", summary)
    model.cpu()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    report = verify_checkpoint(source, output, summary["per_tensor"])
    write_json(output / "zero_preservation_verification.json", report)
    if not report["passed"]:
        raise RuntimeError(f"Saved checkpoint verification failed: {report['errors']}")
    print(f"saved_checkpoint={output}\nsaved_mask_verification=passed", flush=True)
    if args.eval_perplexity:
        # Evaluate a fresh disk reload, rather than only the in-memory result.
        del model
        gc.collect()
        model, _ = load_model_and_tokenizer(output)
        model.to(device)
        ppl = compute_ppl(model, test, args)
        write_json(output / "final_ppl_wikitext.json", {
            "perplexity": ppl, "dataset_name": "wikitext", "checkpoint_dir": str(output),
            "source_checkpoint": str(source), "quant_method": args.method,
            "format": "simulated_nvfp4", "transform_class": "identity",
            "context_length": args.context_length, "stride": args.stride,
            "perplexity_aggregation": summary["perplexity_aggregation"], "evaluated_after_reload": True,
        })
        print(f"perplexity={ppl:.4f}\nsaved_ppl_json={output / 'final_ppl_wikitext.json'}")


if __name__ == "__main__":
    main()
