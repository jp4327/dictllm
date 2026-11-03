import hashlib
import json
from pathlib import Path

import pytest

from dictllm.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_numerical_modules_match_collated_reference_hashes():
    provenance = json.loads((ROOT / "PROVENANCE.json").read_text())
    for relative in provenance["numerically_unchanged_modules"]:
        record = provenance["files"][relative]
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert record["identical"]
        assert actual == record["source_sha256"] == record["sha256"]


@pytest.mark.parametrize("gain,ratio,sparsity", [
    ("1p25x", 0.1, 0.1725), ("1p43x", 0.2, 0.1940),
    ("1p67x", 0.3, 0.2217), ("2x", 0.3, 0.4438),
])
def test_release_recipes(gain, ratio, sparsity):
    config = load_config(ROOT / f"recipes/llama7b_wikitext_comp_gain_{gain}.yaml")
    assert config.data["factorization"]["target_reduction"] == ratio
    assert config.data["pruning"]["coefficient_sparsity"] == sparsity
    assert config.data["pruning"]["hessian_damping"] == 1e-4
    assert config.data["refit"]["ridge_damping"] == 1e-4
    assert config.data["quantization"]["method"] == "masked_gptq"
    assert config.data["quantization"]["rel_damp"] == 0.1
    assert config.data["checkpointing"]["policy"] == "final"
    assert config.calibration_cache().is_relative_to(ROOT)
    assert config.output_root.is_relative_to(ROOT)


def test_no_qwen_models_or_compatibility_wrapper_tree():
    assert not (ROOT / "src/dictllm/modeling_qwen3.py").exists()
    assert not (ROOT / "zero_preserving_nvfp4").exists()
    assert not (ROOT / "src/dictllm/nvfp4/legacy_basis_sharing").exists()
