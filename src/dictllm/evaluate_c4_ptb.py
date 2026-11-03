"""Evaluate a native DictLLM checkpoint using the research C4/PTB protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import datasets
from datasets import DownloadConfig, DownloadManager, load_dataset

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dictllm.evaluation import compute_perplexity
from dictllm.io import write_json
from dictllm.models import load_checkpoint_model, load_tokenizer


C4_DATASET_ID = "allenai/c4"
C4_VALIDATION_FILE = "en/c4-validation.00000-of-00008.json.gz"
PTB_DATASET_ID = "ptb-text-only/ptb_text_only"
PTB_TEST_URL = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--datasets", nargs="+", choices=("c4", "ptb"),
                        default=["c4", "ptb"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"),
                        default="bfloat16")
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=2048)
    parser.add_argument("--c4-data-file", type=Path,
                        help="Local override; otherwise download the first English C4 validation shard.")
    parser.add_argument("--ptb-data-file", type=Path,
                        help="Local override; otherwise download the PTB test split automatically.")
    parser.add_argument("--c4-revision", default="main", help="C4 Hugging Face dataset revision.")
    parser.add_argument("--ptb-revision", default="main", help="PTB Hugging Face dataset revision.")
    parser.add_argument("--ptb-protocol", choices=("research", "whitespace_preserved"),
                        default="research",
                        help="Research strips sentences; whitespace_preserved keeps them intact and saves a separate JSON.")
    parser.add_argument("--c4-documents", type=int, default=2000)
    parser.add_argument("--dataset-cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path,
                        help="Defaults to the checkpoint directory.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--packed-runtime", choices=("dequantize",),
                        help="Explicitly load identity-packed NVFP4 with floating-point activations.")
    parser.add_argument("--cache-dequantized", action="store_true",
                        help="Cache unpacked weights for faster accuracy evaluation, not a memory benchmark.")
    args = parser.parse_args(argv)
    if args.context_length < 2 or not 1 <= args.stride <= args.context_length:
        parser.error("Need context length >= 2 and stride within [1, context length].")
    if args.c4_documents < 1:
        parser.error("--c4-documents must be positive.")
    if args.cache_dequantized and args.packed_runtime is None:
        parser.error("--cache-dequantized requires --packed-runtime dequantize.")
    args.datasets = list(dict.fromkeys(args.datasets))
    return args


def _ptb_sentences(args):
    if args.ptb_data_file is not None:
        path = args.ptb_data_file.expanduser().resolve()
        return path.read_text(encoding="utf-8").splitlines(), {
            "data_source": "local", "data_file": str(path),
            "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    # The HF script strips whitespace, so the alternate protocol needs raw text.
    if args.ptb_protocol == "research":
        script_options = {"trust_remote_code": False} if int(datasets.__version__.split(".")[0]) < 4 else {}
        try:
            raw = load_dataset(
                PTB_DATASET_ID, "penn_treebank", split="test",
                revision=args.ptb_revision,
                cache_dir=str(args.dataset_cache_dir) if args.dataset_cache_dir else None,
                **script_options,
            )
        except (RuntimeError, ValueError) as exc:
            message = str(exc).lower()
            if not any(marker in message for marker in (
                "dataset scripts are no longer supported", "trust_remote_code", "custom code",
            )):
                raise
            # Do not execute remote Python or change datasets versions to load PTB.
            print("PTB legacy dataset script unavailable; downloading its upstream test text.", flush=True)
            if args.ptb_revision != "main":
                raise ValueError("A pinned PTB revision cannot use the upstream fallback; provide --ptb-data-file.") from exc
        else:
            return list(raw["sentence"]), {
                "data_source": "huggingface", "dataset_id": PTB_DATASET_ID,
                "dataset_config": "penn_treebank", "dataset_split": "test",
                "dataset_revision": args.ptb_revision,
                "dataset_fingerprint": getattr(raw, "_fingerprint", None),
            }
    elif args.ptb_revision != "main":
        raise ValueError("Whitespace-preserving PTB uses upstream text; provide --ptb-data-file for pinned data.")

    manager = DownloadManager(download_config=DownloadConfig(
        cache_dir=str(args.dataset_cache_dir) if args.dataset_cache_dir else None,
    ))
    path = Path(manager.download(PTB_TEST_URL))
    return path.read_text(encoding="utf-8").splitlines(), {
        "data_source": "huggingface_download_manager_upstream",
        "dataset_id": PTB_DATASET_ID, "dataset_split": "test",
        "data_url": PTB_TEST_URL, "data_file": str(path),
        "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def prepare_test_text(name, args):
    if name == "c4":
        cache_dir = str(args.dataset_cache_dir) if args.dataset_cache_dir else None
        if args.c4_data_file is not None:
            path = args.c4_data_file.expanduser().resolve()
            raw = load_dataset("json", data_files=str(path), split="train", cache_dir=cache_dir)
            provenance = {"data_source": "local", "data_file": str(path)}
        else:
            raw = load_dataset(
                C4_DATASET_ID, "en", data_files={"validation": C4_VALIDATION_FILE},
                split="validation", revision=args.c4_revision, cache_dir=cache_dir,
                # The repository's split statistics describe all eight shards.
                verification_mode="no_checks",
            )
            provenance = {
                "data_source": "huggingface", "dataset_id": C4_DATASET_ID,
                "dataset_config": "en", "dataset_split": "validation",
                "dataset_revision": args.c4_revision, "dataset_file": C4_VALIDATION_FILE,
                "dataset_fingerprint": getattr(raw, "_fingerprint", None),
            }
        count = min(args.c4_documents, len(raw))
        text = "\n\n".join(raw[:count]["text"])
    else:
        sentences, provenance = _ptb_sentences(args)
        if args.ptb_protocol == "research":
            # Match the stripped sentences in the original HF penn_treebank split.
            sentences = [line.strip() for line in sentences]
        count = len(sentences)
        text = "\n\n".join(sentences)
    if not text.strip():
        raise ValueError(f"No evaluation text for {name}.")
    provenance.update({"records": count, "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()})
    if name == "ptb":
        provenance.update({
            "protocol": ("preserve_sentence_whitespace_double_newline_join"
                         if args.ptb_protocol == "whitespace_preserved"
                         else "strip_sentence_whitespace_double_newline_join"),
            "strip_sentences": args.ptb_protocol == "research",
            "sentence_separator": "\n\n", "encoding": "utf-8",
        })
    return text, provenance


def main(argv=None):
    args = parse_args(argv)
    checkpoint = args.checkpoint.expanduser().resolve()
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    architecture = config.get("model_type")
    packed = bool(config.get("offline_nvfp4")) or (checkpoint / "packed_nvfp4_manifest.json").exists()
    if packed and args.packed_runtime is None:
        raise ValueError("Packed NVFP4 requires --packed-runtime dequantize.")
    if args.packed_runtime is not None:
        from dictllm.nvfp4.evaluation import read_evaluation_config

        read_evaluation_config(checkpoint)
    elif architecture != "llama" or not config.get("dictllm_enabled"):
        raise ValueError("Expected an unquantized native DictLLM LLaMA checkpoint.")
    if config.get("dictllm_quantization", {}).get("packed"):
        raise ValueError("Packed quantized checkpoints require their dedicated evaluator.")
    output_dir = (args.output_dir or checkpoint).expanduser().resolve()
    outputs = {name: output_dir / f"final_ppl_{name}.json" for name in args.datasets}
    if "ptb" in outputs and args.ptb_protocol == "whitespace_preserved":
        outputs["ptb"] = output_dir / "final_ppl_ptb_whitespace_preserved.json"
    for name, output in outputs.items():
        source = args.c4_data_file if name == "c4" else args.ptb_data_file
        if source is not None and not source.expanduser().is_file():
            raise FileNotFoundError(f"Evaluation data not found: {source}")
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {output}; pass --overwrite.")

    runtime_metadata = {}
    if args.packed_runtime is not None:
        from dictllm.nvfp4.checkpoint_adapter import load_tokenizer as load_packed_tokenizer
        from dictllm.nvfp4.evaluation import evaluation_metadata, load_evaluation_model

        base_config = read_evaluation_config(checkpoint)
        tokenizer = load_packed_tokenizer(checkpoint, base_config.to_dict())
        model = load_evaluation_model(checkpoint, args.dtype, args.device, args.cache_dequantized)
        runtime_metadata = evaluation_metadata(args.cache_dequantized)
        runtime_metadata["dtype_resolved"] = str(next(model.parameters()).dtype)
    else:
        tokenizer = load_tokenizer(checkpoint)
        model = load_checkpoint_model(checkpoint, architecture, args.dtype)
    for name, output in outputs.items():
        text, provenance = prepare_test_text(name, args)
        token_options = {"add_special_tokens": True} if name == "ptb" and args.ptb_protocol == "whitespace_preserved" else {}
        tokens = tokenizer(text, return_tensors="pt", **token_options)
        token_count = int(tokens.input_ids.shape[1])
        if token_count < 2:
            raise ValueError(f"{name} evaluation needs at least two tokens.")
        perplexity = compute_perplexity(
            model, tokens, context_length=args.context_length, stride=args.stride,
            device=args.device, description=f"{name.upper()} PPL",
        )
        if not math.isfinite(perplexity):
            raise ValueError(f"Non-finite {name} perplexity: {perplexity}")
        if token_options:
            provenance.update({
                "add_special_tokens": True,
                "bos_token_id": getattr(tokenizer, "bos_token_id", None),
                "token_sha256": hashlib.sha256(tokens.input_ids.cpu().contiguous().numpy().tobytes()).hexdigest(),
            })
        write_json(output, {
            "perplexity": perplexity,
            "dataset_name": name,
            "checkpoint_dir": str(checkpoint),
            "context_length": args.context_length,
            "stride": args.stride,
            "dtype": args.dtype,
            "device": args.device,
            "token_count": token_count,
            "perplexity_aggregation": "legacy_unweighted_mean_of_window_losses",
            **provenance,
            **runtime_metadata,
        })
        print(f"{name}: perplexity={perplexity:.4f}")
        print(f"saved_ppl_json={output}")


if __name__ == "__main__":
    main()
