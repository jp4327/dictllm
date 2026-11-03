from pathlib import Path

import pytest

from dictllm.config import load_config


def test_load_config_applies_semantic_defaults(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
schema_version: 1
model:
  name_or_path: local-model
  architecture: llama
calibration:
  cache_dir: cache
output:
  root: output
""",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.data["pruning"]["hessian_damping"] == 1e-4
    assert config.data["refit"]["ridge_damping"] == 1e-4
    assert config.data["factorization"]["default_layer_group_size"] == 2
    assert config.calibration_cache() == (tmp_path / "cache").resolve()
    assert config.stage_output("factorize").name == "01_factorized"


def test_rejects_ambiguous_or_unsupported_rank_alignment(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
model:
  name_or_path: local-model
  architecture: llama
calibration:
  cache_dir: cache
factorization:
  rank_alignment: 128
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="rank_alignment"):
        load_config(path)

