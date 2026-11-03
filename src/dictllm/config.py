"""YAML configuration loading and validation."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .constants import PROJECTION_PATHS, STAGE_DIRECTORIES, STAGE_ORDER


DEFAULTS = {
    "schema_version": 1,
    "pipeline": {
        "stages": list(STAGE_ORDER),
        "seed": 2023,
        "resume": True,
        "deterministic_algorithms": False,
    },
    "model": {
        "revision": None,
        "dtype": "float32",
        "attention_implementation": "sdpa",
        "cache_dir": None,
    },
    "calibration": {
        "dataset": "wikitext",
        "split": "train",
        "samples": 256,
        "batch_size": 1,
        "context_length": 2048,
        "seed": 2023,
        "device": "cuda",
        "cache_format": "dictllm",
        "dataset_cache_dir": ".cache/huggingface",
        "covariance_normalization": "sum",
        "num_workers": 4,
        "force": False,
    },
    "factorization": {
        "enabled": True,
        "targets": list(PROJECTION_PATHS),
        "target_reduction": 0.30,
        "default_layer_group_size": 2,
        "per_target_layer_group_size": {},
        "rank_policy": "target_reduction",
        "explicit_ranks": {},
        "rank_alignment": "none",
        "covariance_relative_damping": 0.0,
        "fallback_eigenvalue_shift": 7e-6,
        "cholesky_max_attempts": 6,
        "minimum_rank": 1,
        "compute_dtype": "float32",
        "numerical_method": "stable",
        "device": "cuda",
    },
    "pruning": {
        "enabled": True,
        "method": "hessian",
        "coefficient_sparsity": 0.0,
        "hessian_damping": 1e-4,
        "hessian_inverse_method": "cholesky",
        "coefficient_block_size": 128,
        "device": "cuda",
    },
    "refit": {
        "enabled": True,
        "method": "mod",
        "objective": "weight",
        "ridge_damping": 1e-4,
        "preserve_coefficient_mask": True,
        "device": "cpu",
    },
    "quantization": {
        "export_mode": "fake",
        "targets": ["basis", "coefficient"],
        "format": "nvfp4",
        "w_bits": 4,
        "w_group_size": 16,
        "scale_precision": "e4m3",
        "transform_class": "identity",
        "method": "masked_gptq",
        "blocksize": 128,
        "rel_damp": 0.1,
        "calibration_size": 16,
        "calibration_batch_size": 1,
        "calibration_selection": "first",
        "device": "cuda",
        "max_modules": None,
    },
    "serialization": {
        "format": "safetensors",
        "refit_dtype": "bfloat16",
        "shared_tensor_policy": "canonical_repository",
        "max_shard_size": "4GB",
        "save_tokenizer": True,
    },
    "evaluation": {
        "enabled": True,
        "datasets": ["wikitext"],
        "context_length": 2048,
        "stride": 2048,
        "device": "cuda",
    },
    "output": {"root": "outputs/dictllm_run"},
    "checkpointing": {"policy": "final"},
}


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


@dataclass(frozen=True)
class DictLLMConfig:
    data: dict[str, Any]
    path: Path

    def section(self, name: str) -> dict[str, Any]:
        return self.data[name]

    @property
    def stages(self) -> list[str]:
        return list(self.data["pipeline"]["stages"])

    @property
    def output_root(self) -> Path:
        return self.resolve_path(self.data["output"]["root"])

    def stage_output(self, stage: str) -> Path:
        return self.output_root / STAGE_DIRECTORIES[stage]

    def resolve_path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.path.parent / path).resolve()

    def calibration_cache(self) -> Path:
        return self.resolve_path(self.data["calibration"]["cache_dir"])

    def snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(self.data)


def _validate(config: dict[str, Any]) -> None:
    if config["schema_version"] != 1:
        raise ValueError("Only config schema_version=1 is supported.")

    model = config["model"]
    if not model.get("name_or_path"):
        raise ValueError("model.name_or_path is required.")
    if model.get("architecture") != "llama":
        raise ValueError("This minimal release supports model.architecture='llama' only.")

    stages = config["pipeline"]["stages"]
    if not stages or any(stage not in STAGE_ORDER for stage in stages):
        raise ValueError(f"pipeline.stages must use {list(STAGE_ORDER)}.")
    positions = [STAGE_ORDER.index(stage) for stage in stages]
    if positions != sorted(set(positions)):
        raise ValueError("pipeline.stages must be unique and ordered.")

    factorization = config["factorization"]
    targets = factorization["targets"]
    unknown = sorted(set(targets) - set(PROJECTION_PATHS))
    if unknown:
        raise ValueError(f"Unknown factorization targets: {unknown}")
    reduction = float(factorization["target_reduction"])
    if not 0.0 <= reduction < 1.0:
        raise ValueError("factorization.target_reduction must be in [0, 1).")
    if factorization["rank_alignment"] != "none":
        raise ValueError("The core release supports rank_alignment='none' only.")
    if factorization["rank_policy"] not in {"target_reduction", "explicit"}:
        raise ValueError("Unsupported factorization.rank_policy.")
    if factorization["numerical_method"] not in {"stable", "basis_sharing"}:
        raise ValueError("Unsupported factorization.numerical_method.")
    if (factorization["numerical_method"] == "basis_sharing"
            and factorization["compute_dtype"] != "float64"):
        raise ValueError("basis_sharing factorization requires compute_dtype: float64.")
    if int(factorization["minimum_rank"]) < 1:
        raise ValueError("factorization.minimum_rank must be positive.")
    if int(factorization["cholesky_max_attempts"]) < 1:
        raise ValueError("factorization.cholesky_max_attempts must be positive.")
    for target in targets:
        group_size = factorization["per_target_layer_group_size"].get(
            target, factorization["default_layer_group_size"]
        )
        if int(group_size) < 1:
            raise ValueError(f"Layer group size for {target} must be positive.")

    calibration = config["calibration"]
    if calibration["dataset"] != "wikitext":
        raise ValueError("The initial core release supports WikiText calibration only.")
    if calibration["cache_format"] not in {
        "dictllm",
        "legacy_basis_sharing",
    }:
        raise ValueError("Unsupported calibration.cache_format.")
    if calibration["covariance_normalization"] not in {"sum", "mean"}:
        raise ValueError("calibration.covariance_normalization must be sum or mean.")
    if int(calibration["samples"]) < 1 or int(calibration["batch_size"]) < 1:
        raise ValueError("Calibration sample and batch counts must be positive.")

    pruning = config["pruning"]
    if pruning["method"] != "hessian":
        raise ValueError("The core release currently implements Hessian pruning only.")
    sparsity = float(pruning["coefficient_sparsity"])
    if not 0.0 <= sparsity < 1.0:
        raise ValueError("pruning.coefficient_sparsity must be in [0, 1).")
    if pruning["hessian_inverse_method"] not in {"inverse", "cholesky"}:
        raise ValueError("Unsupported pruning.hessian_inverse_method.")

    refit = config["refit"]
    if refit["method"] != "mod" or refit["objective"] != "weight":
        raise ValueError("The core release implements weight-space MOD refitting only.")
    if float(refit["ridge_damping"]) < 0:
        raise ValueError("refit.ridge_damping must be non-negative.")

    if config["serialization"]["format"] != "safetensors":
        raise ValueError("The canonical repository currently writes Safetensors.")
    if config["serialization"]["refit_dtype"] not in {"bfloat16", "float32"}:
        raise ValueError("serialization.refit_dtype must be bfloat16 or float32.")
    if not config["output"].get("root"):
        raise ValueError("output.root is required.")
    quantization = config["quantization"]
    if quantization["export_mode"] not in {"packed", "fake"}:
        raise ValueError("quantization.export_mode must be packed or fake.")
    if quantization["export_mode"] == "packed" and (
        quantization["w_group_size"] != 16 or quantization["scale_precision"] != "e4m3"
    ):
        raise ValueError("Packed NVFP4 requires group size 16 and E4M3 scales.")
    if quantization["format"] != "nvfp4" or quantization["w_bits"] != 4:
        raise ValueError("Quantization currently supports 4-bit NVFP4 only.")
    if (not quantization["targets"]
            or set(quantization["targets"]) - {"basis", "coefficient"}):
        raise ValueError("quantization.targets must select basis and/or coefficient.")
    for key in ["w_group_size", "blocksize", "calibration_size", "calibration_batch_size"]:
        if int(quantization[key]) < 1:
            raise ValueError(f"quantization.{key} must be positive.")
    if quantization["scale_precision"] not in {"e4m3", "fp32"}:
        raise ValueError("Unsupported quantization.scale_precision.")
    if quantization["transform_class"] != "identity":
        raise ValueError("Only zero-preserving identity quantization is supported.")
    if quantization["method"] not in {"masked_gptq", "rtn"}:
        raise ValueError("quantization.method must be masked_gptq or rtn.")
    if float(quantization["rel_damp"]) < 0:
        raise ValueError("quantization.rel_damp must be non-negative.")
    if quantization["calibration_selection"] not in {"first", "random"}:
        raise ValueError("quantization.calibration_selection must be first or random.")
    if quantization["max_modules"] is not None:
        raise ValueError("Partial quantization is excluded from the minimal pipeline.")

    if config["checkpointing"]["policy"] not in {"final", "all", "none"}:
        raise ValueError("checkpointing.policy must be final, all, or none.")


def load_config(path: str | Path) -> DictLLMConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}.")
    merged = _deep_merge(DEFAULTS, loaded)
    if "cache_dir" not in merged["calibration"]:
        merged["calibration"]["cache_dir"] = ".cache/dictllm/calibration"
    _validate(merged)
    return DictLLMConfig(data=merged, path=config_path)
