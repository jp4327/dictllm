import pickle

import torch
import yaml
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from dictllm.config import load_config
from dictllm.io import read_manifest
from dictllm.stages import factorize_stage, prune_stage, refit_stage
from dictllm.models import load_checkpoint_model
from safetensors import safe_open


def _save_tiny_source(path):
    config = LlamaConfig(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=2,
    )
    LlamaForCausalLM(config).save_pretrained(path, safe_serialization=True)
    backend = Tokenizer(
        WordLevel(
            {
                "[UNK]": 0,
                "[BOS]": 1,
                "[EOS]": 2,
                "a": 3,
                "b": 4,
            },
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[EOS]",
    )
    tokenizer.save_pretrained(path)


def test_three_stage_pipeline_with_local_tiny_model(tmp_path):
    source = tmp_path / "source"
    _save_tiny_source(source)
    calibration = tmp_path / "legacy_calibration" / "self_attn.k_proj"
    calibration.mkdir(parents=True)
    for layer in range(2):
        with (calibration / f"{layer}.pkl").open("wb") as handle:
            pickle.dump(torch.eye(16), handle)

    config_path = tmp_path / "recipe.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "name_or_path": str(source),
                    "architecture": "llama",
                    "dtype": "float32",
                },
                "calibration": {
                    "cache_dir": str(calibration.parent),
                    "cache_format": "legacy_basis_sharing",
                    "samples": 1,
                    "batch_size": 1,
                    "context_length": 16,
                    "device": "cpu",
                    "dataset_cache_dir": str(tmp_path / "dataset_cache"),
                },
                "factorization": {
                    "targets": ["q"],
                    "target_reduction": 0.30,
                    "default_layer_group_size": 2,
                    "device": "cpu",
                    "compute_dtype": "float32",
                },
                "pruning": {
                    "coefficient_sparsity": 0.25,
                    "device": "cpu",
                    "coefficient_block_size": 4,
                },
                "refit": {"device": "cpu"},
                "evaluation": {"enabled": False},
                "output": {"root": str(tmp_path / "outputs")},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)
    factorized = factorize_stage(config, skip_evaluation=True)
    pruned = prune_stage(config, checkpoint=factorized, skip_evaluation=True)
    refit = refit_stage(config, checkpoint=pruned, skip_evaluation=True)

    assert read_manifest(factorized)["completed_stages"] == ["factorize"]
    assert read_manifest(pruned)["completed_stages"] == ["factorize", "prune"]
    final = read_manifest(refit)
    assert final["completed_stages"] == ["factorize", "prune", "refit"]
    assert final["summary"]["mean_mse_after"] < final["summary"]["mean_mse_before"]
    assert final["summary"]["checkpoint_dtype"] == "bfloat16"
    for checkpoint, expected_dtype in [
        (factorized, torch.float32),
        (pruned, torch.float32),
        (refit, torch.bfloat16),
    ]:
        for shard in checkpoint.glob("*.safetensors"):
            with safe_open(shard, framework="pt", device="cpu") as tensors:
                for name in tensors.keys():
                    tensor = tensors.get_tensor(name)
                    if tensor.is_floating_point():
                        assert tensor.dtype == expected_dtype
    reloaded = load_checkpoint_model(refit, "llama", "bfloat16").eval()
    assert torch.isfinite(reloaded(input_ids=torch.tensor([[1, 3, 4]])).logits).all()
