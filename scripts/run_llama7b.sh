#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
gain="${1:-2x}"
if (( $# )); then shift; fi
case "$gain" in
    1p25x|1p43x|1p67x|2x) ;;
    *) printf 'Gain must be 1p25x, 1p43x, 1p67x, or 2x.\n' >&2; exit 2 ;;
esac
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
exec "${PYTHON_BIN:-python}" -m dictllm.cli run \
    --config "$ROOT/recipes/llama7b_wikitext_comp_gain_${gain}.yaml" "$@"
