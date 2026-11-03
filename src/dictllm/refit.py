"""Fixed-support MOD dictionary refitting."""

from __future__ import annotations

import time

import torch

from .modeling import FactorizedLinear, dictionary_key, get_projection


def mod_update_dictionary(
    dense_weight: torch.Tensor,
    coefficients: torch.Tensor,
    damping: float,
) -> torch.Tensor:
    gram = coefficients @ coefficients.transpose(0, 1)
    gram.diagonal().add_(float(damping))
    right = dense_weight @ coefficients.transpose(0, 1)
    return torch.linalg.solve(gram, right.transpose(0, 1)).transpose(0, 1).contiguous()


@torch.no_grad()
def refit_model(dense_model, factorized_model, config: dict) -> list[dict]:
    refit = config["refit"]
    device = torch.device(refit["device"])
    repository = factorized_model.model.dictllm_dictionaries
    records = []
    for target in factorized_model.config.dictllm_targets:
        for group_index, layers in enumerate(factorized_model.config.dictllm_groups[target]):
            started = time.perf_counter()
            dense_weights = []
            coefficients = []
            nonzeros_before = 0
            for layer_index in layers:
                dense_projection = get_projection(dense_model, layer_index, target)
                factorized_projection = get_projection(factorized_model, layer_index, target)
                if not isinstance(factorized_projection, FactorizedLinear):
                    raise TypeError(f"{target} layer {layer_index} is not factorized.")
                dense_weights.append(
                    dense_projection.weight.detach().transpose(0, 1).to(device=device).float()
                )
                current = (
                    factorized_projection.coefficient.weight.detach()
                    .transpose(0, 1)
                    .to(device=device)
                    .float()
                )
                coefficients.append(current)
                nonzeros_before += int(torch.count_nonzero(current).item())
            dense_weight = torch.cat(dense_weights, dim=1)
            coefficient = torch.cat(coefficients, dim=1)
            dictionary_module = repository[dictionary_key(target, group_index)]
            dictionary_before = (
                dictionary_module.weight.detach().transpose(0, 1).to(device=device).float()
            )
            mse_before = torch.mean(
                (dense_weight - dictionary_before @ coefficient).square()
            ).item()
            dictionary_after = mod_update_dictionary(
                dense_weight, coefficient, refit["ridge_damping"]
            )
            mse_after = torch.mean(
                (dense_weight - dictionary_after @ coefficient).square()
            ).item()
            dictionary_module.weight.copy_(
                dictionary_after.transpose(0, 1).to(dictionary_module.weight)
            )
            nonzeros_after = sum(
                int(
                    torch.count_nonzero(
                        get_projection(factorized_model, layer, target).coefficient.weight
                    ).item()
                )
                for layer in layers
            )
            if refit["preserve_coefficient_mask"] and nonzeros_after != nonzeros_before:
                raise RuntimeError(
                    f"MOD unexpectedly changed coefficient support for {target} group {group_index}."
                )
            records.append(
                {
                    "target": target,
                    "group_index": group_index,
                    "layers": list(layers),
                    "rank": coefficient.shape[0],
                    "dictionary_shape": list(dictionary_after.shape),
                    "coefficient_shape": list(coefficient.shape),
                    "coefficient_nonzeros": nonzeros_after,
                    "mse_before": mse_before,
                    "mse_after": mse_after,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            print(
                f"{target} group={group_index:02d} layers={list(layers)} "
                f"mse={mse_before:.6e}->{mse_after:.6e}"
            )
    return records

