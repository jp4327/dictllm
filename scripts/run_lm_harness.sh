#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    printf '%s\n' \
        'Usage: bash scripts/run_lm_harness.sh CHECKPOINT_OR_HF_ID [evaluator options...]' \
        'Defaults: seven tasks, zero-shot, batch size 1, BOS disabled, acc summary.' \
        'Output: CHECKPOINT/lm_harness_results.json (or --output-path PATH).' \
        'Packed checkpoints automatically use the dequantize runtime.' \
        'Use --cache-dequantized for faster packed accuracy evaluation (more memory).' \
        'Dense HF model IDs require --output-path PATH. Existing results require --overwrite.' \
        "Install evaluation dependencies first: python -m pip install -e '.[eval]'" \
        'Set CUDA_VISIBLE_DEVICES and PYTHON_BIN to select the GPU and Python environment.'
    exit 0
fi
if (( $# == 0 )); then
    printf 'A checkpoint directory or Hugging Face model ID is required; see --help.\n' >&2
    exit 2
fi

model="$1"
shift
# Do not change directories: relative checkpoint and output paths belong to the caller.
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
runtime=()
if [[ -f "$model/packed_nvfp4_manifest.json" ]]; then
    runtime=(--packed-runtime dequantize)
fi
exec "${PYTHON_BIN:-python}" -m dictllm.eval_lm_harness "$model" \
    --num-fewshot 0 --batch-size 1 --dtype auto \
    --add-bos-token false --summary-metric acc "${runtime[@]}" "$@"
