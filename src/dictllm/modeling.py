"""Shared-dictionary projection modules and topology helpers."""

from __future__ import annotations

import weakref
from collections.abc import Iterable, Iterator

import torch
import torch.nn as nn

from .constants import PROJECTION_PATHS


class FactorizedLinear(nn.Module):
    """A coefficient projection backed by one canonical shared dictionary."""

    def __init__(self, dictionary: nn.Linear, out_features: int, bias: bool = False):
        super().__init__()
        if dictionary.bias is not None:
            raise ValueError("DictLLM dictionaries must not have a bias.")
        self.in_features = dictionary.in_features
        self.out_features = int(out_features)
        self.rank = dictionary.out_features
        self.coefficient = nn.Linear(self.rank, self.out_features, bias=bias)
        object.__setattr__(self, "_dictionary_ref", weakref.ref(dictionary))

    @property
    def dictionary(self) -> nn.Linear:
        dictionary = self._dictionary_ref()
        if dictionary is None:
            raise RuntimeError("The canonical dictionary was released unexpectedly.")
        return dictionary

    @property
    def weight(self) -> torch.Tensor:
        return self.coefficient.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.coefficient.bias

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.coefficient(self.dictionary(inputs))


def get_layers(model: nn.Module):
    try:
        return model.model.layers
    except AttributeError as exc:
        raise TypeError("Expected a LLaMA model.model.layers layout.") from exc


def get_projection(model: nn.Module, layer_index: int, target: str) -> nn.Module:
    return get_layers(model)[layer_index].get_submodule(PROJECTION_PATHS[target])


def set_projection(
    model: nn.Module,
    layer_index: int,
    target: str,
    projection: nn.Module,
) -> None:
    parent_path, attribute = PROJECTION_PATHS[target].rsplit(".", 1)
    parent = get_layers(model)[layer_index].get_submodule(parent_path)
    setattr(parent, attribute, projection)


def dictionary_key(target: str, group_index: int) -> str:
    return f"{target}_group_{group_index}"


def ensure_dictionary_repository(model: nn.Module) -> nn.ModuleDict:
    repository = getattr(model.model, "dictllm_dictionaries", None)
    if repository is None:
        repository = nn.ModuleDict()
        model.model.dictllm_dictionaries = repository
    return repository


def build_groups(num_layers: int, group_size: int) -> list[list[int]]:
    if group_size < 1:
        raise ValueError("Layer group size must be positive.")
    return [
        list(range(start, min(start + group_size, num_layers)))
        for start in range(0, num_layers, group_size)
    ]


def replace_projection_group(
    model: nn.Module,
    target: str,
    group_index: int,
    layers: Iterable[int],
    rank: int,
    dictionary_weight: torch.Tensor | None = None,
    coefficient_weights: list[torch.Tensor] | None = None,
) -> None:
    layers = list(layers)
    if not layers:
        raise ValueError("A sharing group cannot be empty.")
    reference = get_projection(model, layers[0], target)
    if not isinstance(reference, nn.Linear):
        raise TypeError(f"Expected a dense nn.Linear for {target} group {group_index}.")
    if reference.bias is not None:
        raise NotImplementedError("Biased projection factorization is not supported.")

    repository = ensure_dictionary_repository(model)
    key = dictionary_key(target, group_index)
    if key in repository:
        raise ValueError(f"Dictionary {key} already exists.")
    dictionary = nn.Linear(reference.in_features, rank, bias=False)
    dictionary.to(device=reference.weight.device, dtype=reference.weight.dtype)
    repository[key] = dictionary

    if dictionary_weight is not None:
        expected = (rank, reference.in_features)
        if tuple(dictionary_weight.shape) != expected:
            raise ValueError(f"Dictionary {key} has {dictionary_weight.shape}, expected {expected}.")
        with torch.no_grad():
            dictionary.weight.copy_(dictionary_weight.to(dictionary.weight))

    if coefficient_weights is not None and len(coefficient_weights) != len(layers):
        raise ValueError("Expected one coefficient matrix per layer.")
    for offset, layer_index in enumerate(layers):
        dense = get_projection(model, layer_index, target)
        if not isinstance(dense, nn.Linear):
            raise TypeError(f"Expected a dense projection at {target} layer {layer_index}.")
        factorized = FactorizedLinear(
            dictionary=dictionary,
            out_features=dense.out_features,
            bias=False,
        ).to(device=dense.weight.device, dtype=dense.weight.dtype)
        if coefficient_weights is not None:
            expected = (dense.out_features, rank)
            coefficient = coefficient_weights[offset]
            if tuple(coefficient.shape) != expected:
                raise ValueError(
                    f"Coefficient at {target} layer {layer_index} has "
                    f"{coefficient.shape}, expected {expected}."
                )
            with torch.no_grad():
                factorized.coefficient.weight.copy_(coefficient.to(factorized.coefficient.weight))
        set_projection(model, layer_index, target, factorized)


def install_empty_dictllm(model: nn.Module) -> None:
    targets = list(getattr(model.config, "dictllm_targets", []))
    groups_by_target = getattr(model.config, "dictllm_groups", {})
    ranks_by_target = getattr(model.config, "dictllm_ranks", {})
    for target in targets:
        groups = groups_by_target[target]
        saved_ranks = ranks_by_target[target]
        ranks = [int(saved_ranks)] * len(groups) if isinstance(saved_ranks, int) else saved_ranks
        if len(ranks) != len(groups):
            raise ValueError(f"Rank metadata does not match groups for {target}.")
        for group_index, (layers, rank) in enumerate(zip(groups, ranks)):
            replace_projection_group(model, target, group_index, layers, int(rank))

    if getattr(model.config, "dictllm_quantization", {}).get("packed", False):
        raise ValueError("Rotation-based legacy packed checkpoints are not supported.")


def iter_dictionaries(model: nn.Module) -> Iterator[tuple[str, int, list[int], nn.Linear]]:
    repository = model.model.dictllm_dictionaries
    for target in model.config.dictllm_targets:
        for group_index, layers in enumerate(model.config.dictllm_groups[target]):
            yield target, group_index, list(layers), repository[dictionary_key(target, group_index)]


def iter_coefficients(
    model: nn.Module,
) -> Iterator[tuple[str, int, int, FactorizedLinear]]:
    for target in model.config.dictllm_targets:
        for group_index, layers in enumerate(model.config.dictllm_groups[target]):
            for layer_index in layers:
                module = get_projection(model, layer_index, target)
                if not isinstance(module, FactorizedLinear):
                    raise TypeError(f"{target} layer {layer_index} is not factorized.")
                yield target, group_index, layer_index, module


def topology_counts(model: nn.Module) -> dict[str, int]:
    return {
        "dictionaries": sum(1 for _ in iter_dictionaries(model)),
        "coefficients": sum(1 for _ in iter_coefficients(model)),
    }
