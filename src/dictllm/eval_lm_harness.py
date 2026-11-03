"""Zero-shot LM Harness evaluation for native DictLLM and dense HF models."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dictllm.io import write_json
from dictllm.models import _dtype_kwargs, dtype_from_name, model_class


DEFAULT_TASKS = (
    "openbookqa", "winogrande", "hellaswag", "piqa", "mathqa",
    "arc_easy", "arc_challenge",
)
PRIMARY_METRICS = {
    name: "acc_norm" if name in {"hellaswag", "piqa", "arc_easy", "arc_challenge"}
    else "acc" for name in DEFAULT_TASKS
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", help="Native DictLLM checkpoint directory or dense HF model ID.")
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument(
        "--add-bos-token", "--add_bos_token", type=str.lower,
        choices=("true", "false"), default="false",
        help="Prepend BOS during harness tokenization (default: false, matching the original evaluation).",
    )
    parser.add_argument("--max-length", type=int, help="Optional harness context-length override.")
    parser.add_argument("--limit", type=int, help="Examples per task; smoke tests only, not full benchmarks.")
    parser.add_argument("--summary-metric", choices=("mixed", "acc", "acc_norm"), default="acc")
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--packed-runtime", choices=("dequantize",),
                        help="Explicitly load identity-packed NVFP4 with floating-point activations.")
    parser.add_argument("--cache-dequantized", action="store_true",
                        help="Cache unpacked weights for accuracy evaluation, not a memory benchmark.")
    args = parser.parse_args(argv)
    args.add_bos_token = args.add_bos_token == "true"
    args.tasks = list(dict.fromkeys(task.strip() for task in args.tasks.split(",") if task.strip()))
    if not args.tasks or args.num_fewshot < 0 or args.batch_size < 1:
        parser.error("Need at least one task, nonnegative few-shot count, and positive batch size.")
    if args.max_length is not None and args.max_length < 2:
        parser.error("--max-length must be at least two.")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive.")
    if args.cache_dequantized and args.packed_runtime is None:
        parser.error("--cache-dequantized requires --packed-runtime dequantize.")
    return args


def load_model(source, config, dtype, device):
    from transformers import AutoModelForCausalLM

    architectures = getattr(config, "architectures", []) or []
    if any(name.startswith("Share") for name in architectures):
        raise ValueError("Legacy Basis_Sharing checkpoints need the original harness; use native DictLLM outputs here.")
    if getattr(config, "dictllm_quantization", {}).get("packed"):
        raise ValueError("Packed NVFP4 requires its dedicated runtime loader, not this floating-point evaluator.")
    if getattr(config, "dictllm_enabled", False):
        cls = model_class(config.model_type)
    else:
        if any(name.startswith("Dict") for name in architectures):
            raise ValueError("DictLLM checkpoint is missing dictllm_enabled; refusing a dense-model fallback.")
        cls = AutoModelForCausalLM
    resolved_dtype = "auto" if dtype == "auto" else dtype_from_name(dtype)
    # Load on CPU before moving to one device; do not dispatch shared dictionaries.
    model = cls.from_pretrained(
        source, config=config, low_cpu_mem_usage=True, **_dtype_kwargs(resolved_dtype),
    )
    return model.to(device).eval()


def load_tokenizer(source, config):
    from transformers import AutoTokenizer, LlamaTokenizer, PreTrainedTokenizerBase

    try:
        tokenizer = AutoTokenizer.from_pretrained(source)
    except RecursionError:
        if config.model_type != "llama":
            raise
        tokenizer = None
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        if config.model_type != "llama":
            raise TypeError(f"Invalid tokenizer returned for {source}.")
        # Match the original LM Harness fallback, including BOS=1 rather than UNK=0.
        tokenizer = LlamaTokenizer.from_pretrained(
            source, legacy=True, bos_token="<s>", eos_token="</s>", unk_token="<unk>",
        )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither padding nor EOS token.")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@contextmanager
def mathqa_compatibility(tasks):
    """Keep the old harness workaround narrow and restore the dataset loader."""
    import datasets

    original = datasets.load_dataset
    needs_trust = "mathqa" in tasks and int(datasets.__version__.split(".")[0]) < 4
    if needs_trust:
        def load_dataset(path, *args, **kwargs):
            if str(path).lower() in {"mathqa", "math_qa"} or str(path).lower().endswith("/math_qa"):
                kwargs.setdefault("trust_remote_code", True)
            return original(path, *args, **kwargs)
        datasets.load_dataset = load_dataset
    try:
        yield
    finally:
        datasets.load_dataset = original


def print_summary(task_results, mode):
    print(f"{'Task':<18} {'Metric':<10} {'Score':>9} {'Stderr':>9}")
    for task, metrics in task_results.items():
        metric = PRIMARY_METRICS.get(task, "acc") if mode == "mixed" else mode
        key = f"{metric},none"
        score = metrics.get(key)
        # Never silently substitute acc for acc_norm (or another metric).
        score_text = "N/A" if score is None else f"{score:.4f}"
        stderr = metrics.get(f"{metric}_stderr,none")
        stderr_text = f"{stderr:.4f}" if isinstance(stderr, (float, int)) else "N/A"
        print(f"{task:<18} {metric:<10} {score_text:>9} {stderr_text:>9}")


def main(argv=None):
    args = parse_args(argv)
    candidate = Path(args.model_path).expanduser()
    if candidate.is_dir():
        if (candidate / "packed_nvfp4_manifest.json").exists() and args.packed_runtime is None:
            raise ValueError("Packed NVFP4 requires --packed-runtime dequantize.")
        source = str(candidate.resolve())
    elif (candidate.exists() or args.model_path.startswith((".", "/", "~"))
          or args.model_path.count("/") > 1):
        raise FileNotFoundError(f"Checkpoint directory not found: {candidate}")
    else:
        source = args.model_path
    if args.output_path is None:
        if not candidate.is_dir():
            raise ValueError("--output-path is required for a dense Hugging Face model ID.")
        output = candidate.resolve() / "lm_harness_results.json"
    else:
        output = args.output_path.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}; pass --overwrite.")
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError as exc:
        raise ImportError(
            "LM Harness dependencies are unavailable. Install the optional eval extra "
            "in this interpreter: python -m pip install -e '.[eval]'."
        ) from exc
    from transformers import AutoConfig

    runtime_metadata = {}
    if args.packed_runtime is not None:
        from dictllm.nvfp4.evaluation import (
            evaluation_metadata, load_evaluation_model, read_evaluation_config,
        )

        config = read_evaluation_config(source)
        tokenizer = load_tokenizer(source, config)
        model = load_evaluation_model(source, args.dtype, args.device, args.cache_dequantized)
        runtime_metadata = evaluation_metadata(args.cache_dequantized)
    else:
        config = AutoConfig.from_pretrained(source)
        tokenizer = load_tokenizer(source, config)
        model = load_model(source, config, args.dtype, args.device)
    # Pin this explicitly: harness versions differ in their default BOS policy.
    harness_kwargs = dict(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size,
                          device=args.device, add_bos_token=args.add_bos_token)
    if args.max_length is not None:
        harness_kwargs["max_length"] = args.max_length
    harness = HFLM(**harness_kwargs)
    print(f"Model: {source}\nTasks: {args.tasks}\nAdd BOS token: {args.add_bos_token}\nOutput: {output}")
    if args.limit is not None:
        print(f"SMOKE TEST ONLY: limited to {args.limit} examples per task.")
    with mathqa_compatibility(args.tasks):
        results = lm_eval.simple_evaluate(
            model=harness, tasks=args.tasks, num_fewshot=args.num_fewshot,
            limit=args.limit, log_samples=False,
            random_seed=0, numpy_random_seed=1234, torch_random_seed=1234,
            fewshot_random_seed=1234,
        )
    if results is None:
        raise RuntimeError("LM Harness returned no results; use a single-process evaluation.")
    results["dictllm_evaluation"] = {
        "model_path": source, "dtype_requested": args.dtype,
        "dtype_resolved": str(next(model.parameters()).dtype),
        "device": args.device, "batch_size": args.batch_size,
        "num_fewshot": args.num_fewshot, "limit": args.limit,
        "summary_metric": args.summary_metric,
        "add_bos_token": args.add_bos_token,
        "add_bos_token_resolved": getattr(harness, "add_bos_token", None),
        "tokenizer_class": type(tokenizer).__name__,
        "bos_token_id": tokenizer.bos_token_id, "eos_token_id": tokenizer.eos_token_id,
        **runtime_metadata,
    }
    # Harness results can contain dtype objects; retain its full metric structure.
    write_json(output, json.loads(json.dumps(results, default=str)))
    print_summary(results["results"], args.summary_metric)
    print(f"saved_lm_harness_json={output}")


if __name__ == "__main__":
    main()
