import json
from pathlib import Path

import pytest
import torch
import yaml

from dictllm.cli import main
from dictllm.config import load_config
from dictllm.models import load_checkpoint_model
from dictllm.pipeline import run_pipeline
from test_pipeline import _save_tiny_source


def make_recipe(tmp_path, policy="final", mode="fake"):
    torch.manual_seed(11)
    source = tmp_path / "source"
    _save_tiny_source(source)
    cache = tmp_path / "calibration/self_attn.k_proj"
    cache.mkdir(parents=True)
    import pickle

    for layer in range(2):
        with (cache / f"{layer}.pkl").open("wb") as handle:
            pickle.dump(torch.eye(16), handle)
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump({
        "model": {"name_or_path": str(source), "architecture": "llama", "dtype": "float32"},
        "calibration": {"cache_dir": str(cache.parent), "cache_format": "legacy_basis_sharing",
                        "samples": 1, "batch_size": 1, "context_length": 16, "device": "cpu",
                        "dataset_cache_dir": str(tmp_path / "data")},
        "factorization": {"targets": ["q"], "target_reduction": 0.3, "device": "cpu"},
        "pruning": {"coefficient_sparsity": 0.25, "device": "cpu", "coefficient_block_size": 4},
        "refit": {"device": "cpu"},
        "quantization": {"device": "cpu", "export_mode": mode, "calibration_size": 1},
        "evaluation": {"enabled": False, "context_length": 16, "stride": 16},
        "output": {"root": str(tmp_path / "outputs")},
        "checkpointing": {"policy": policy},
    }))
    return load_config(path)


@pytest.mark.parametrize("policy", ["final", "all", "none"])
def test_successful_retention_and_reports(tmp_path, policy):
    config = make_recipe(tmp_path, policy)
    final = run_pipeline(config, skip_evaluation=True)
    assert config.stage_output("factorize").exists() is (policy == "all")
    assert config.stage_output("prune").exists() is (policy == "all")
    assert config.stage_output("refit").exists() is (policy != "none")
    assert not list(config.output_root.glob(".pipeline-*"))
    for stage in ["factorize", "prune", "refit"]:
        assert (config.output_root / "reports" / stage / "dictllm_manifest.json").is_file()
    report = json.loads((config.output_root / "pipeline_report.json").read_text())
    assert report["status"] == "complete"
    if policy == "none":
        assert final is None and not list(config.output_root.rglob("*.safetensors"))
    else:
        assert final == config.stage_output("refit")
        loaded = load_checkpoint_model(final, "llama", "bfloat16")
        assert torch.isfinite(loaded(input_ids=torch.tensor([[1, 3, 4]])).logits).all()
    assert run_pipeline(config, skip_evaluation=True) == final


def test_final_retention_is_identical_to_all(tmp_path):
    results = []
    for policy in ["all", "final"]:
        directory = tmp_path / policy
        directory.mkdir()
        config = make_recipe(directory, policy)
        final = run_pipeline(config, skip_evaluation=True)
        results.append(load_checkpoint_model(final, "llama", "bfloat16").state_dict())
    assert results[0].keys() == results[1].keys()
    assert all(torch.equal(results[0][name], results[1][name]) for name in results[0])


def test_failure_retains_owned_workspace(tmp_path, monkeypatch):
    from dictllm import pipeline

    config = make_recipe(tmp_path)
    def fail(*args, **kwargs):
        raise RuntimeError("intentional pruning failure")
    monkeypatch.setattr(pipeline, "prune_stage", fail)
    with pytest.raises(RuntimeError, match="intentional"):
        run_pipeline(config, skip_evaluation=True)
    report = json.loads((config.output_root / "pipeline_report.json").read_text())
    assert report["status"] == "failed"
    work = Path(report["recovery_workspace"])
    assert (work / "01_factorized/config.json").is_file()
    assert not config.stage_output("refit").exists()


def test_dry_run_writes_nothing(tmp_path, capsys):
    config = make_recipe(tmp_path)
    main(["run", "--config", str(config.path), "--quantize", "--dry-run"])
    assert "fresh disk reloads" in capsys.readouterr().out
    assert not config.output_root.exists()


@pytest.mark.parametrize("mode", ["fake", "packed"])
def test_four_stage_identity_quantization_without_network(tmp_path, monkeypatch, mode):
    from dictllm.nvfp4 import run_quantization

    config = make_recipe(tmp_path, mode=mode)
    tokens = {"input_ids": torch.tensor([[1, 3, 4, 3]]), "attention_mask": torch.ones(1, 4, dtype=torch.long)}
    monkeypatch.setattr(run_quantization, "prepare_data", lambda *args: ([tokens], None))
    final = run_pipeline(config, quantize=True, skip_evaluation=True)
    assert not config.stage_output("refit").exists()
    assert not list(config.output_root.glob(".pipeline-*"))
    assert not list(config.output_root.glob(".quantize-*"))
    if mode == "fake":
        summary = json.loads((final / "zero_preserving_summary.json").read_text())
        assert summary["rel_damp"] == 0.1 and summary["transform_class"] == "identity"
        assert all(record["revived_zeros"] == 0 for record in summary["per_tensor"])
        model = load_checkpoint_model(final, "llama", "bfloat16")
    else:
        from dictllm.nvfp4.packed_nvfp4 import load_packed_model
        report = json.loads((final / "conversion_verification.json").read_text())
        assert report["passed"] and report["exact_weight_round_trip"]
        assert report["total_revived_zeros"] == 0
        model = load_packed_model(final, device="cpu", dtype=torch.bfloat16)
    assert torch.isfinite(model(input_ids=tokens["input_ids"]).logits).all()


def test_quantization_config_rejects_old_routes(tmp_path):
    config = make_recipe(tmp_path)
    data = yaml.safe_load(config.path.read_text())
    data["quantization"]["transform_class"] = "hadamard"
    config.path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="identity"):
        load_config(config.path)


def test_keep_all_allows_packed_reverification_after_cleanup(tmp_path, monkeypatch):
    from dictllm.nvfp4 import run_quantization
    from dictllm.nvfp4.convert_fake_to_packed import verify_conversion

    config = make_recipe(tmp_path, policy="all", mode="packed")
    tokens = {"input_ids": torch.tensor([[1, 3, 4, 3]])}
    monkeypatch.setattr(run_quantization, "prepare_data", lambda *args: ([tokens], None))
    final = run_pipeline(config, quantize=True, skip_evaluation=True)
    manifest = json.loads((final / "packed_nvfp4_manifest.json").read_text())
    assert Path(manifest["source_checkpoint"]).is_dir()
    assert Path(manifest["fake_checkpoint"]).is_dir()
    assert not list(config.output_root.glob(".quantize-*"))
    assert verify_conversion(final)["passed"]


def test_quantization_failure_keeps_recovery_refit(tmp_path, monkeypatch):
    from dictllm.nvfp4 import run_pipeline as quantizer

    config = make_recipe(tmp_path)
    def fail(*args, **kwargs):
        raise RuntimeError("intentional quantization failure")
    monkeypatch.setattr(quantizer, "main", fail)
    with pytest.raises(RuntimeError, match="intentional"):
        run_pipeline(config, quantize=True, skip_evaluation=True)
    report = json.loads((config.output_root / "pipeline_report.json").read_text())
    recovery = Path(report["recovery_workspace"])
    assert (recovery / "03_refit/config.json").exists()
    assert not (config.output_root / "04_quantized_fake").exists()
