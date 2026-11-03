"""WikiText-2 evaluation of a fresh native/legacy floating-point checkpoint load."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from .io import package_versions, write_json
from .nvfp4.run_quantization import compute_ppl, load_model_and_tokenizer, prepare_data, resolve_checkpoint


def evaluate(checkpoint, *, context_length=2048, stride=2048, device="cuda",
             dataset_cache_dir=Path(".cache/huggingface")):
    checkpoint = resolve_checkpoint(checkpoint)
    output = checkpoint / "final_ppl_wikitext.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}.")
    if context_length < 2 or not 1 <= stride <= context_length:
        raise ValueError("Need context length >=2 and stride within [1, context length].")
    # Exactly the standalone identity quantizer's loader, tokenization and PPL;
    # do not cast the model or reconstruct/cast its nonpersistent RoPE buffers.
    model, tokenizer = load_model_and_tokenizer(checkpoint)
    args = argparse.Namespace(context_length=context_length, stride=stride, device=device,
                              dataset_cache_dir=Path(dataset_cache_dir), eval_perplexity=True)
    _, tokens = prepare_data(tokenizer, args, need_calibration=False)
    model.to(device)
    value = compute_ppl(model, tokens, args)
    payload = {
        "perplexity": value, "dataset_name": "wikitext", "checkpoint_dir": str(checkpoint),
        "context_length": context_length, "stride": stride, "evaluated_after_reload": True,
        "perplexity_aggregation": "legacy_unweighted_mean_of_window_losses",
        "loader": "dictllm.nvfp4.run_quantization.load_model_and_tokenizer",
        "rotary_buffer_protocol": "fresh_from_config_like_checkpoint_reload",
        "resolved_model_dtype": str(next(model.parameters()).dtype),
        "tokenizer_class": type(tokenizer).__name__, "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id, "token_count": int(tokens.input_ids.shape[1]),
        "token_sha256": hashlib.sha256(tokens.input_ids.cpu().contiguous().numpy().tobytes()).hexdigest(),
        "versions": package_versions(),
    }
    write_json(output, payload)
    print(f"perplexity={value:.4f}\nsaved_ppl_json={output}")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset-cache-dir", type=Path, default=Path(".cache/huggingface"))
    args = parser.parse_args(argv)
    evaluate(args.checkpoint, context_length=args.context_length, stride=args.stride,
             device=args.device, dataset_cache_dir=args.dataset_cache_dir)


if __name__ == "__main__":
    main()
