"""Dictionary-aware FasterHessian coefficient pruning."""

from __future__ import annotations

import time

import torch

from .calibration import load_group_covariance
from .modeling import FactorizedLinear, dictionary_key, get_projection


def compute_hessian_inverse(
    hessian: torch.Tensor, damping: float, method: str
) -> torch.Tensor:
    hessian = hessian.double()
    identity = torch.eye(hessian.shape[0], device=hessian.device, dtype=hessian.dtype)
    damped = hessian + float(damping) * identity
    if method == "inverse":
        return torch.linalg.inv(damped)
    if method == "cholesky":
        return torch.cholesky_inverse(torch.linalg.cholesky(damped))
    raise ValueError(f"Unsupported inverse method: {method}")


def faster_hessian_prune_from_inverse(
    weight: torch.Tensor,
    hessian_inverse: torch.Tensor,
    sparsity: float,
    block_size: int,
) -> torch.Tensor:
    if sparsity <= 0:
        return weight.clone()
    pruned = weight.double().clone()
    input_dim, output_dim = pruned.shape
    count = int(input_dim * float(sparsity))
    if count <= 0:
        return weight.clone()
    inverse = hessian_inverse.to(device=pruned.device, dtype=pruned.dtype)
    diagonal = torch.diag(inverse).clamp_min(1e-12)
    scores = pruned.square() / diagonal.unsqueeze(1)
    prune_indices = torch.argsort(scores, dim=0)[:count]
    columns = None
    for start in range(0, output_dim, int(block_size)):
        end = min(start + int(block_size), output_dim)
        current = pruned[:, start:end]
        indices = prune_indices[:, start:end]
        width = end - start
        if columns is None or columns.numel() != width:
            columns = torch.arange(width, device=pruned.device)
        for step in range(count):
            index = indices[step]
            error = current[index, columns] / diagonal[index]
            current -= inverse[:, index] * error.unsqueeze(0)
        current[indices, columns.unsqueeze(0).expand_as(indices)] = 0.0
        pruned[:, start:end] = current
    return pruned.to(weight.dtype)


@torch.no_grad()
def prune_model(model, config: dict, cache_dir) -> list[dict]:
    pruning = config["pruning"]
    calibration = config["calibration"]
    device = torch.device(pruning["device"])
    repository = model.model.dictllm_dictionaries
    records = []
    for target in model.config.dictllm_targets:
        for group_index, layers in enumerate(model.config.dictllm_groups[target]):
            started = time.perf_counter()
            dictionary_module = repository[dictionary_key(target, group_index)]
            dictionary = dictionary_module.weight.detach().transpose(0, 1).to(
                device=device, dtype=torch.float64
            )
            covariance, rows = load_group_covariance(
                cache_dir,
                target,
                list(layers),
                calibration["cache_format"],
                calibration["covariance_normalization"],
            )
            covariance = covariance.to(device=device, dtype=torch.float64)
            coefficient_hessian = dictionary.transpose(0, 1) @ covariance @ dictionary
            inverse = compute_hessian_inverse(
                coefficient_hessian,
                pruning["hessian_damping"],
                pruning["hessian_inverse_method"],
            )
            before = after = 0
            for layer_index in layers:
                projection = get_projection(model, layer_index, target)
                if not isinstance(projection, FactorizedLinear):
                    raise TypeError(f"{target} layer {layer_index} is not factorized.")
                coefficient = projection.coefficient.weight.detach().transpose(0, 1).to(device)
                before += int(torch.count_nonzero(coefficient).item())
                pruned = faster_hessian_prune_from_inverse(
                    coefficient,
                    inverse,
                    pruning["coefficient_sparsity"],
                    pruning["coefficient_block_size"],
                )
                after += int(torch.count_nonzero(pruned).item())
                projection.coefficient.weight.copy_(
                    pruned.transpose(0, 1).to(projection.coefficient.weight)
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            records.append(
                {
                    "target": target,
                    "group_index": group_index,
                    "layers": list(layers),
                    "rank": dictionary.shape[1],
                    "covariance_rows": rows,
                    "coefficient_nonzeros_before": before,
                    "coefficient_nonzeros_after": after,
                    "coefficient_sparsity": 1.0 - after / before,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            print(
                f"{target} group={group_index:02d} layers={list(layers)} "
                f"sparsity={records[-1]['coefficient_sparsity']:.6f}"
            )
    return records

