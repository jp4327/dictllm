"""Immutable factorize, prune, and refit stages."""

from __future__ import annotations

import gc
import random
import time
from pathlib import Path

import torch
import numpy as np

from .calibration import (
    cache_is_complete,
    collect_grouped_covariances,
    groups_by_family,
)
from .config import DictLLMConfig
from .constants import STAGE_ORDER
from .data import prepare_wikitext_calibration, prepare_wikitext_test
from .evaluation import compute_perplexity
from .factorization import factorize_model
from .io import (
    config_hash,
    make_manifest,
    model_file_bytes,
    read_manifest,
    stage_workspace,
    write_json,
    write_yaml,
)
from .modeling import build_groups, topology_counts
from .models import dtype_from_name, load_checkpoint_model, load_source_model, load_tokenizer
from .pruning import prune_model
from .refit import refit_model


def _seed_everything(config: dict) -> None:
    seed = int(config["pipeline"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(
        bool(config["pipeline"]["deterministic_algorithms"]), warn_only=True
    )


def _groups_for_model(model, config: dict) -> dict[str, list[list[int]]]:
    factorization = config["factorization"]
    num_layers = int(model.config.num_hidden_layers)
    return {
        target: build_groups(
            num_layers,
            int(
                factorization["per_target_layer_group_size"].get(
                    target, factorization["default_layer_group_size"]
                )
            ),
        )
        for target in factorization["targets"]
    }


def _calibration_metadata(config: dict) -> dict:
    calibration = config["calibration"]
    return {
        "source_model": config["model"]["name_or_path"],
        "dataset": calibration["dataset"],
        "split": calibration["split"],
        "samples": int(calibration["samples"]),
        "batch_size": int(calibration["batch_size"]),
        "context_length": int(calibration["context_length"]),
        "seed": int(calibration["seed"]),
    }


def _ensure_calibration(
    model,
    tokenizer,
    config: DictLLMConfig,
    groups: dict[str, list[list[int]]],
    force: bool,
) -> None:
    raw = config.data
    calibration = raw["calibration"]
    cache_dir = config.calibration_cache()
    family_groups = groups_by_family(raw["factorization"]["targets"], groups)
    metadata = _calibration_metadata(raw)
    complete = cache_is_complete(
        cache_dir,
        family_groups,
        metadata,
        calibration["cache_format"],
    )
    if calibration["cache_format"] == "legacy_basis_sharing":
        if not complete:
            raise FileNotFoundError(
                f"Research calibration cache is incomplete: {cache_dir}"
            )
        print(f"Using research calibration cache: {cache_dir}")
        return
    if complete and not (force or calibration["force"]):
        print(f"Using DictLLM calibration cache: {cache_dir}")
        return

    dataset_cache = config.resolve_path(calibration["dataset_cache_dir"])
    dataloader = prepare_wikitext_calibration(tokenizer, calibration, dataset_cache)
    model.to(calibration["device"])
    collect_grouped_covariances(model, dataloader, family_groups, cache_dir, metadata)
    model.to("cpu")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Saved DictLLM calibration cache: {cache_dir}")


def _evaluate(model, tokenizer, config: DictLLMConfig, skip: bool) -> dict:
    evaluation = config.data["evaluation"]
    if skip or not evaluation["enabled"]:
        return {}
    unsupported = set(evaluation["datasets"]) - {"wikitext"}
    if unsupported:
        raise NotImplementedError(f"Unsupported evaluation datasets: {sorted(unsupported)}")
    dataset_cache = config.resolve_path(config.data["calibration"]["dataset_cache_dir"])
    test_tokens = prepare_wikitext_test(tokenizer, dataset_cache)
    perplexity = compute_perplexity(
        model,
        test_tokens,
        int(evaluation["context_length"]),
        int(evaluation["stride"]),
        evaluation["device"],
    )
    model.to("cpu")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"wikitext": {"perplexity": perplexity}}


def _save_stage(
    *,
    model,
    tokenizer,
    config: DictLLMConfig,
    stage: str,
    output: Path,
    parent: Path | None,
    summary: dict,
    evaluation: dict,
    overwrite: bool,
) -> Path:
    completed = list(STAGE_ORDER[: STAGE_ORDER.index(stage) + 1])
    model.config.use_cache = True
    model.config.dictllm_completed_stages = completed
    model.config.dictllm_source_model = config.data["model"]["name_or_path"]
    class_name = type(model).__name__
    model.config.architectures = [class_name]
    with stage_workspace(output, overwrite=overwrite) as temporary:
        model.save_pretrained(
            temporary,
            safe_serialization=True,
            max_shard_size=config.data["serialization"]["max_shard_size"],
        )
        if config.data["serialization"]["save_tokenizer"]:
            tokenizer.save_pretrained(temporary)
        write_yaml(temporary / "dictllm_config.yaml", config.snapshot())
        if "wikitext" in evaluation:
            write_json(
                temporary / "final_ppl_wikitext.json",
                {
                    "perplexity": evaluation["wikitext"]["perplexity"],
                    "dataset_name": "wikitext",
                    "stage": stage,
                    "checkpoint_dir": str(output.resolve()),
                },
            )
        manifest = make_manifest(
            config=config.snapshot(),
            stage=stage,
            completed_stages=completed,
            output=output,
            parent=parent,
            summary=summary,
            evaluation=evaluation,
        )
        write_json(temporary / "dictllm_manifest.json", manifest)
    manifest = read_manifest(output)
    manifest["serialized_model_bytes"] = model_file_bytes(output)
    write_json(output / "dictllm_manifest.json", manifest)
    print(f"saved_checkpoint={output}")
    return output


def factorize_stage(
    config: DictLLMConfig,
    *,
    output: Path | None = None,
    overwrite: bool = False,
    force_calibration: bool = False,
    skip_evaluation: bool = False,
) -> Path:
    raw = config.data
    _seed_everything(raw)
    output = (output or config.stage_output("factorize")).resolve()
    tokenizer = load_tokenizer(
        raw["model"]["name_or_path"],
        raw["model"].get("revision"),
        raw["model"].get("cache_dir"),
    )
    print(f"Loading dense source model: {raw['model']['name_or_path']}")
    model = load_source_model(raw["model"], device_map="cpu")
    model.config.use_cache = False
    groups = _groups_for_model(model, raw)
    _ensure_calibration(model, tokenizer, config, groups, force_calibration)
    model.to("cpu")
    started = time.perf_counter()
    records = factorize_model(model, groups, raw, config.calibration_cache())
    summary = {
        "elapsed_seconds": time.perf_counter() - started,
        "targeted_dense_parameters": sum(item["dense_parameters"] for item in records),
        "targeted_factor_parameters": sum(item["factor_parameters"] for item in records),
        "groups": records,
        "topology_counts": topology_counts(model),
    }
    summary["targeted_retained_ratio"] = (
        summary["targeted_factor_parameters"] / summary["targeted_dense_parameters"]
    )
    evaluation = _evaluate(model, tokenizer, config, skip_evaluation)
    return _save_stage(
        model=model,
        tokenizer=tokenizer,
        config=config,
        stage="factorize",
        output=output,
        parent=None,
        summary=summary,
        evaluation=evaluation,
        overwrite=overwrite,
    )


def _validate_parent(parent: Path, expected_stage: str, config: DictLLMConfig) -> dict:
    manifest = read_manifest(parent)
    if manifest["stage"] != expected_stage:
        raise ValueError(
            f"Expected a {expected_stage} checkpoint, found {manifest['stage']} at {parent}."
        )
    if manifest["config_sha256"] != config_hash(config.snapshot()):
        raise ValueError("Checkpoint and requested YAML have different configuration hashes.")
    return manifest


def prune_stage(
    config: DictLLMConfig,
    checkpoint: Path | None = None,
    *,
    output: Path | None = None,
    overwrite: bool = False,
    skip_evaluation: bool = False,
) -> Path:
    raw = config.data
    _seed_everything(raw)
    parent = (checkpoint or config.stage_output("factorize")).resolve()
    _validate_parent(parent, "factorize", config)
    output = (output or config.stage_output("prune")).resolve()
    tokenizer = load_tokenizer(parent)
    model = load_checkpoint_model(parent, raw["model"]["architecture"], raw["model"]["dtype"])
    started = time.perf_counter()
    records = prune_model(model, raw, config.calibration_cache())
    total_before = sum(item["coefficient_nonzeros_before"] for item in records)
    total_after = sum(item["coefficient_nonzeros_after"] for item in records)
    summary = {
        "elapsed_seconds": time.perf_counter() - started,
        "coefficient_nonzeros_before": total_before,
        "coefficient_nonzeros_after": total_after,
        "measured_coefficient_sparsity": 1.0 - total_after / total_before,
        "groups": records,
        "topology_counts": topology_counts(model),
    }
    evaluation = _evaluate(model, tokenizer, config, skip_evaluation)
    return _save_stage(
        model=model,
        tokenizer=tokenizer,
        config=config,
        stage="prune",
        output=output,
        parent=parent,
        summary=summary,
        evaluation=evaluation,
        overwrite=overwrite,
    )


def refit_stage(
    config: DictLLMConfig,
    checkpoint: Path | None = None,
    *,
    output: Path | None = None,
    overwrite: bool = False,
    skip_evaluation: bool = False,
) -> Path:
    raw = config.data
    _seed_everything(raw)
    parent = (checkpoint or config.stage_output("prune")).resolve()
    _validate_parent(parent, "prune", config)
    output = (output or config.stage_output("refit")).resolve()
    tokenizer = load_tokenizer(parent)
    factorized = load_checkpoint_model(
        parent, raw["model"]["architecture"], raw["model"]["dtype"]
    )
    dense = load_source_model(raw["model"], device_map="cpu").eval()
    started = time.perf_counter()
    records = refit_model(dense, factorized, raw)
    del dense
    gc.collect()
    summary = {
        "elapsed_seconds": time.perf_counter() - started,
        "num_dictionaries": len(records),
        "mean_mse_before": sum(item["mse_before"] for item in records) / len(records),
        "mean_mse_after": sum(item["mse_after"] for item in records) / len(records),
        "groups": records,
        "topology_counts": topology_counts(factorized),
    }
    export_dtype = raw["serialization"]["refit_dtype"]
    factorized.to(dtype=dtype_from_name(export_dtype))
    summary["checkpoint_dtype"] = export_dtype
    evaluation = _evaluate(factorized, tokenizer, config, skip_evaluation)
    return _save_stage(
        model=factorized,
        tokenizer=tokenizer,
        config=config,
        stage="refit",
        output=output,
        parent=parent,
        summary=summary,
        evaluation=evaluation,
        overwrite=overwrite,
    )
