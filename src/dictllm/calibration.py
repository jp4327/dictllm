"""Grouped activation-covariance collection and cache loading."""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm

from .constants import CALIBRATION_FAMILIES, TARGET_TO_FAMILY
from .modeling import get_layers


@dataclass
class CovarianceAccumulator:
    covariance: torch.Tensor | None = None
    rows: int = 0

    def update(self, inputs: torch.Tensor) -> None:
        flattened = inputs.detach().reshape(-1, inputs.shape[-1]).float()
        gram = (flattened.transpose(0, 1) @ flattened).cpu()
        if self.covariance is None:
            self.covariance = gram
        else:
            self.covariance.add_(gram)
        self.rows += flattened.shape[0]


def covariance_path(cache_dir: Path, family: str, group_index: int) -> Path:
    return cache_dir / family / f"group_{group_index}.pt"


def finest_compatible_groups(
    family: str,
    targets: list[str],
    groups_by_target: dict[str, list[list[int]]],
) -> list[list[int]]:
    candidates = [groups_by_target[target] for target in targets]
    finest = max(candidates, key=len)
    finest_sets = [set(group) for group in finest]
    for target in targets:
        for target_group in groups_by_target[target]:
            requested = set(target_group)
            pieces = [piece for piece in finest_sets if piece.issubset(requested)]
            reconstructed = set().union(*pieces) if pieces else set()
            if reconstructed != requested:
                raise ValueError(
                    f"Groups for calibration family {family} are not nested partitions."
                )
    return [list(group) for group in finest]


def groups_by_family(
    targets: list[str], groups_by_target: dict[str, list[list[int]]]
) -> dict[str, list[list[int]]]:
    result = {}
    for family in CALIBRATION_FAMILIES:
        family_targets = [target for target in targets if TARGET_TO_FAMILY[target] == family]
        if family_targets:
            result[family] = finest_compatible_groups(
                family, family_targets, groups_by_target
            )
    return result


def _read_manifest(cache_dir: Path) -> dict | None:
    path = cache_dir / "manifest.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def cache_is_complete(
    cache_dir: Path,
    family_groups: dict[str, list[list[int]]],
    metadata: dict,
    cache_format: str,
) -> bool:
    if cache_format == "legacy_basis_sharing":
        return all(
            (cache_dir / CALIBRATION_FAMILIES[family] / f"{layer}.pkl").is_file()
            for family, groups in family_groups.items()
            for group in groups
            for layer in group
        )
    manifest = _read_manifest(cache_dir)
    if manifest is None or manifest.get("groups_by_family") != family_groups:
        return False
    if (cache_format == "dictllm"
            and manifest.get("accumulation_method") != "per_layer_then_sum"):
        return False
    compatibility_keys = {
        "source_model": "model_name",
        "dataset": "calibration_dataset",
        "samples": "calibration_size",
        "batch_size": "calibration_batch_size",
    }
    for key, value in metadata.items():
        compatibility_key = compatibility_keys.get(key)
        saved = manifest.get(key, manifest.get(compatibility_key or ""))
        if saved != value:
            return False
    return all(
        covariance_path(cache_dir, family, index).is_file()
        for family, groups in family_groups.items()
        for index in range(len(groups))
    )


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def save_covariances(
    cache_dir: Path,
    accumulators: dict[str, list[CovarianceAccumulator]],
    family_groups: dict[str, list[list[int]]],
    metadata: dict,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for family, family_accumulators in accumulators.items():
        (cache_dir / family).mkdir(parents=True, exist_ok=True)
        for index, accumulator in enumerate(family_accumulators):
            if accumulator.covariance is None or accumulator.rows == 0:
                raise RuntimeError(f"No covariance collected for {family} group {index}.")
            output = covariance_path(cache_dir, family, index)
            temporary = output.with_suffix(".pt.tmp")
            torch.save(
                {
                    "covariance": accumulator.covariance,
                    "rows": accumulator.rows,
                    "layers": family_groups[family][index],
                },
                temporary,
            )
            os.replace(temporary, output)
    _write_json(
        cache_dir / "manifest.json",
        {
            **metadata,
            "cache_schema_version": 1,
            "accumulation_method": "per_layer_then_sum",
            "families": CALIBRATION_FAMILIES,
            "groups_by_family": family_groups,
        },
    )


def collect_grouped_covariances(
    model,
    dataloader,
    family_groups: dict[str, list[list[int]]],
    cache_dir: Path,
    metadata: dict,
) -> None:
    layer_accumulators = {
        family: {layer: CovarianceAccumulator() for group in groups for layer in group}
        for family, groups in family_groups.items()
    }
    hooks = []
    layers = get_layers(model)
    for family, projection_path in CALIBRATION_FAMILIES.items():
        if family not in family_groups:
            continue
        for layer_index, layer in enumerate(layers):
            accumulator = layer_accumulators[family][layer_index]
            projection = layer.get_submodule(projection_path)

            def hook_fn(module, inputs, output, accumulator=accumulator):
                del module, output
                accumulator.update(inputs[0])

            hooks.append(projection.register_forward_hook(hook_fn))

    model.config.use_cache = False
    model.eval()
    device = next(model.parameters()).device
    try:
        for batch in tqdm(dataloader, desc="Collect activation covariances"):
            inputs = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
                if key != "labels"
            }
            with torch.no_grad():
                output = model.model(**inputs, use_cache=False, return_dict=False)
            del output, inputs
    finally:
        for hook in hooks:
            hook.remove()
    # Match Basis Sharing: accumulate batches per layer before summing layers.
    accumulators = {}
    for family, groups in family_groups.items():
        accumulators[family] = []
        for group in groups:
            combined = CovarianceAccumulator()
            for layer in group:
                current = layer_accumulators[family].pop(layer)
                if current.covariance is None:
                    raise RuntimeError(f"No covariance collected for {family} layer {layer}.")
                if combined.covariance is None:
                    combined.covariance = current.covariance
                else:
                    combined.covariance.add_(current.covariance)
                combined.rows += current.rows
            accumulators[family].append(combined)
    save_covariances(cache_dir, accumulators, family_groups, metadata)


def _selected_cached_indices(
    cache_dir: Path, family: str, requested_layers: list[int]
) -> list[int]:
    manifest = _read_manifest(cache_dir)
    if manifest is None:
        raise FileNotFoundError(f"Missing calibration manifest in {cache_dir}.")
    cached_groups = manifest["groups_by_family"][family]
    requested = set(requested_layers)
    selected = [
        index for index, group in enumerate(cached_groups) if set(group).issubset(requested)
    ]
    reconstructed = {layer for index in selected for layer in cached_groups[index]}
    if reconstructed != requested:
        raise ValueError(
            f"Cached {family} groups cannot reconstruct layers {requested_layers}."
        )
    return selected


def load_group_covariance(
    cache_dir: Path,
    target: str,
    layers: list[int],
    cache_format: str,
    normalization: str,
) -> tuple[torch.Tensor, int]:
    family = TARGET_TO_FAMILY[target]
    if cache_format == "legacy_basis_sharing":
        covariance = None
        for layer in layers:
            path = cache_dir / CALIBRATION_FAMILIES[family] / f"{layer}.pkl"
            with path.open("rb") as handle:
                current = pickle.load(handle)
            covariance = current.clone() if covariance is None else covariance.add(current)
        rows = 1
    else:
        covariance = None
        rows = 0
        for index in _selected_cached_indices(cache_dir, family, layers):
            payload = torch.load(
                covariance_path(cache_dir, family, index),
                map_location="cpu",
                weights_only=True,
            )
            current = payload["covariance"]
            covariance = current.clone() if covariance is None else covariance.add(current)
            rows += int(payload["rows"])
    if covariance is None:
        raise RuntimeError(f"No covariance found for {target} layers {layers}.")
    if normalization == "mean":
        covariance.div_(max(rows, 1))
    return covariance, rows
