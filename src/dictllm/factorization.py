"""Activation-aware cross-layer factorization."""

from __future__ import annotations

import gc
import time

import torch
import torch.nn as nn

from .calibration import load_group_covariance
from .modeling import get_projection, replace_projection_group
from .models import dtype_from_name


def compute_rank(
    in_features: int,
    out_features: int,
    group_size: int,
    target_reduction: float,
    minimum_rank: int = 1,
) -> int:
    retained = 1.0 - float(target_reduction)
    dense_parameters = in_features * out_features * group_size
    denominator = in_features + out_features * group_size
    rank = int((dense_parameters * retained) // denominator)
    upper = min(in_features, out_features * group_size)
    return min(max(int(minimum_rank), rank), upper)


def _cholesky_scaling(
    covariance: torch.Tensor,
    relative_damping: float,
    fallback_shift: float,
    max_attempts: int = 6,
) -> tuple[torch.Tensor, float]:
    covariance = covariance.clone()
    diagonal_mean = covariance.diagonal().mean().abs().clamp_min(1e-12)
    base = float(diagonal_mean.item()) * relative_damping
    if base == 0.0:
        try:
            return torch.linalg.cholesky(covariance).transpose(0, 1), 0.0
        except torch.linalg.LinAlgError:
            base = fallback_shift
    applied = 0.0
    for attempt in range(max_attempts):
        requested = base * (10**attempt)
        covariance.diagonal().add_(requested - applied)
        applied = requested
        try:
            return torch.linalg.cholesky(covariance).transpose(0, 1), applied
        except torch.linalg.LinAlgError:
            continue
    smallest = float(torch.linalg.eigvalsh(covariance).min().item())
    shift = max(-smallest + fallback_shift, fallback_shift)
    covariance.diagonal().add_(shift)
    return torch.linalg.cholesky(covariance).transpose(0, 1), applied + shift


def _thin_svd(matrix: torch.Tensor):
    if matrix.device.type != "cuda":
        return torch.linalg.svd(matrix, full_matrices=False)
    try:
        return torch.linalg.svd(matrix, full_matrices=False, driver="gesvdj")
    except torch.linalg.LinAlgError:
        return torch.linalg.svd(matrix, full_matrices=False, driver="gesvd")


def factorize_projection_group(
    model,
    target: str,
    group_index: int,
    layers: list[int],
    rank: int,
    covariance: torch.Tensor,
    factorization: dict,
) -> dict:
    dense = [get_projection(model, layer, target) for layer in layers]
    if any(not isinstance(module, nn.Linear) for module in dense):
        raise TypeError(f"{target} group {group_index} is not entirely dense.")
    in_features = dense[0].in_features
    out_features = dense[0].out_features
    if any(
        module.in_features != in_features or module.out_features != out_features
        for module in dense
    ):
        raise ValueError(f"Projection dimensions differ in {target} group {group_index}.")

    device = torch.device(factorization["device"])
    compute_dtype = dtype_from_name(factorization["compute_dtype"])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    original_numerics = factorization.get("numerical_method", "stable") == "basis_sharing"
    if original_numerics:
        covariance = covariance.cpu().double().clone()
        applied_damping = (
            float(covariance.diagonal().mean().abs().item())
            * float(factorization["covariance_relative_damping"])
        )
        covariance.diagonal().add_(applied_damping)
        try:
            scaling = torch.linalg.cholesky(covariance).T
        except torch.linalg.LinAlgError:
            shift = -float(torch.linalg.eigvalsh(covariance)[0].item()) + float(
                factorization["fallback_eigenvalue_shift"]
            )
            covariance += shift * torch.eye(covariance.shape[0])
            applied_damping += shift
            scaling = torch.linalg.cholesky(covariance).T
        inverse_scaling = torch.linalg.inv(scaling).to(device)
        scaling = scaling.to(device)
    else:
        covariance = covariance.to(device=device, dtype=compute_dtype)
        scaling, applied_damping = _cholesky_scaling(
            covariance,
            float(factorization["covariance_relative_damping"]),
            float(factorization["fallback_eigenvalue_shift"]),
            int(factorization.get("cholesky_max_attempts", 6)),
        )
    concatenated = torch.cat(
        [
            module.weight.detach().transpose(0, 1).to(device=device, dtype=compute_dtype)
            for module in dense
        ],
        dim=1,
    )
    if original_numerics:
        left, singular_values, right = torch.svd(scaling @ concatenated)
        right_t = right.T
    else:
        left, singular_values, right_t = _thin_svd(scaling @ concatenated)
    rank = min(int(rank), singular_values.numel())
    if original_numerics:
        dictionary = ((inverse_scaling @ left) @ torch.diag(singular_values))[:, :rank]
    else:
        scaled_dictionary = left[:, :rank] * singular_values[:rank].unsqueeze(0)
        dictionary = torch.linalg.solve_triangular(
            scaling, scaled_dictionary, upper=True
        )
    coefficients = right_t[:rank, :]

    source_dtype = dense[0].weight.dtype
    dictionary_weight = dictionary.transpose(0, 1).cpu().to(source_dtype)
    coefficient_weights = [
        coefficients[:, offset * out_features:(offset + 1) * out_features]
        .transpose(0, 1)
        .cpu()
        .to(source_dtype)
        for offset in range(len(layers))
    ]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    replace_projection_group(
        model,
        target,
        group_index,
        layers,
        rank,
        dictionary_weight,
        coefficient_weights,
    )
    dense_parameters = in_features * out_features * len(layers)
    factor_parameters = rank * in_features + rank * out_features * len(layers)
    return {
        "target": target,
        "group_index": group_index,
        "layers": layers,
        "in_features": in_features,
        "out_features": out_features,
        "rank": rank,
        "dense_parameters": dense_parameters,
        "factor_parameters": factor_parameters,
        "retained_ratio": factor_parameters / dense_parameters,
        "applied_covariance_damping": applied_damping,
        "elapsed_seconds": elapsed,
    }


def factorize_model(
    model,
    groups_by_target: dict[str, list[list[int]]],
    config: dict,
    cache_dir,
) -> list[dict]:
    factorization = config["factorization"]
    calibration = config["calibration"]
    summaries = []
    ranks_by_target = {}
    for target in factorization["targets"]:
        target_ranks = []
        for group_index, layers in enumerate(groups_by_target[target]):
            reference = get_projection(model, layers[0], target)
            if factorization["rank_policy"] == "explicit":
                explicit = factorization["explicit_ranks"][target]
                rank = int(explicit[group_index] if isinstance(explicit, list) else explicit)
            else:
                rank = compute_rank(
                    reference.in_features,
                    reference.out_features,
                    len(layers),
                    factorization["target_reduction"],
                    factorization["minimum_rank"],
                )
            covariance, _ = load_group_covariance(
                cache_dir,
                target,
                layers,
                calibration["cache_format"],
                calibration["covariance_normalization"],
            )
            summary = factorize_projection_group(
                model, target, group_index, layers, rank, covariance, factorization
            )
            summaries.append(summary)
            target_ranks.append(summary["rank"])
            print(
                f"{target} group={group_index:02d} layers={layers} "
                f"rank={summary['rank']} retained={summary['retained_ratio']:.4f}"
            )
            del covariance
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        ranks_by_target[target] = target_ranks

    model.config.dictllm_enabled = True
    model.config.dictllm_schema_version = 1
    model.config.dictllm_targets = list(factorization["targets"])
    model.config.dictllm_groups = groups_by_target
    model.config.dictllm_ranks = ranks_by_target
    model.config.dictllm_target_reduction = float(factorization["target_reduction"])
    return summaries
