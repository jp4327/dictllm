from pathlib import Path

from dictllm.io import config_hash, read_manifest, stage_workspace, write_json


def test_stage_workspace_publishes_complete_directory_atomically(tmp_path: Path):
    output = tmp_path / "01_factorized"
    with stage_workspace(output) as temporary:
        write_json(temporary / "dictllm_manifest.json", {"stage": "factorize"})
        assert not output.exists()
    assert read_manifest(output)["stage"] == "factorize"


def test_config_hash_is_order_independent():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})

