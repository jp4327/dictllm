"""Rotation-free simulated FP4 quantization with a fixed original zero mask.

Weights remain floating point for evaluation. No packing or QuTLASS is used.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def validate_weight(weight):
    if weight.ndim != 2 or not weight.is_floating_point() or weight.numel() == 0:
        raise ValueError("Expected a nonempty floating-point matrix.")
    if not torch.isfinite(weight).all():
        raise ValueError("Weights contain NaN or infinity.")


def group_scales(weight, group_size=16, scale_precision="e4m3"):
    """Return effective FP32 scales using tensor-global E4M3 normalization."""
    validate_weight(weight)
    if group_size < 1 or scale_precision not in {"e4m3", "fp32"}:
        raise ValueError("Invalid group size or scale precision.")
    columns = weight.shape[1]
    padding = (-columns) % group_size
    padded = F.pad(weight.float(), (0, padding))
    amax = padded.reshape(weight.shape[0], -1, group_size).abs().amax(-1)
    scales = amax / 6.0
    global_scale = torch.ones((), device=weight.device)
    if scale_precision == "e4m3":
        tensor_amax = amax.max()
        if tensor_amax > 0:
            global_scale = 448.0 * 6.0 / tensor_amax
        if not torch.isfinite(global_scale) or global_scale <= 0:
            raise ValueError("Tensor range cannot be represented by an FP32 global scale.")
        encoded = (scales * global_scale).clamp(max=448).to(torch.float8_e4m3fn).float()
        # Keep tiny nonzero groups representable instead of replacing their scale by 1.
        encoded = torch.where((amax > 0) & (encoded == 0), 2.0 ** -9, encoded)
        scales = encoded / global_scale
    scales = torch.where(amax == 0, torch.ones_like(scales), scales)
    if not torch.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("Effective scales must be finite and positive.")
    return scales, float(global_scale)


def round_fp4(values, scales):
    """E2M1 nearest-even rounding, including exact zero and saturation at 6."""
    absolute = (values / scales).abs()
    rounded = torch.zeros_like(absolute)
    rounded[(absolute > .25) & (absolute < .75)] = .5
    rounded[(absolute >= .75) & (absolute <= 1.25)] = 1
    rounded[(absolute > 1.25) & (absolute < 1.75)] = 1.5
    rounded[(absolute >= 1.75) & (absolute <= 2.5)] = 2
    rounded[(absolute > 2.5) & (absolute < 3.5)] = 3
    rounded[(absolute >= 3.5) & (absolute <= 5)] = 4
    rounded[absolute > 5] = 6
    return rounded * torch.sign(values) * scales


def mask_statistics(before, after):
    if before.shape != after.shape:
        raise ValueError("Source and quantized shapes differ.")
    original_zeros, final_zeros = before == 0, after == 0
    count = before.numel()
    zeros_before = int(original_zeros.sum())
    zeros_after = int(final_zeros.sum())
    revived = int((original_zeros & ~final_zeros).sum())
    return {
        "num_parameters": count,
        "zeros_before": zeros_before,
        "zeros_after": zeros_after,
        "sparsity_before": zeros_before / count,
        "sparsity_after": zeros_after / count,
        "revived_zeros": revived,
        "additional_zeros": int((~original_zeros & final_zeros).sum()),
        "mask_preserved": revived == 0,
        "nonfinite": int((~torch.isfinite(after)).sum()),
    }


def reconstruction_errors(before, after, hessian=None, chunk_rows=128):
    """H=2 X.T X/N gives the relative XW error without retaining X."""
    weight_num = weight_den = act_num = act_den = 0.0
    for start in range(0, before.shape[0], chunk_rows):
        original = before[start:start + chunk_rows].float()
        difference = after[start:start + chunk_rows].float() - original
        weight_num += float(difference.double().square().sum())
        weight_den += float(original.double().square().sum())
        if hessian is not None:
            act_num += float(((difference @ hessian).double() * difference.double()).sum())
            act_den += float(((original @ hessian).double() * original.double()).sum())
    return {
        "weight_relative_frobenius_error": math.sqrt(weight_num / weight_den) if weight_den else None,
        "activation_relative_frobenius_error": (
            math.sqrt(max(0.0, act_num) / act_den) if act_den > 0 else None
        ),
        "zero_weight_reference": weight_den == 0,
        "zero_activation_reference": act_den <= 0 if hessian is not None else None,
    }


@torch.no_grad()
def quantize_weight(weight, *, method="masked_gptq", hessian=None, group_size=16,
                    scale_precision="e4m3", blocksize=128, rel_damp=.01):
    validate_weight(weight)
    if method not in {"rtn", "masked_gptq"}:
        raise ValueError("Method must be rtn or masked_gptq.")
    if blocksize < 1 or not math.isfinite(rel_damp) or rel_damp < 0:
        raise ValueError("Invalid block size or damping.")
    scales, global_scale = group_scales(weight, group_size, scale_precision)
    working = weight.float().clone()
    original_mask = weight == 0
    columns = weight.shape[1]
    if hessian is not None:
        if hessian.shape != (columns, columns) or hessian.device != weight.device:
            raise ValueError("Hessian shape/device does not match the weight input dimension.")
        if not torch.isfinite(hessian).all():
            raise ValueError("Hessian contains NaN or infinity.")
    if method == "rtn":
        for start in range(0, columns, group_size):
            end = min(start + group_size, columns)
            working[:, start:end] = round_fp4(working[:, start:end], scales[:, start // group_size, None])
    else:
        if hessian is None:
            raise ValueError("masked_gptq requires an input Hessian.")
        H = hessian.float().clone()
        if not torch.allclose(H, H.T, rtol=1e-4, atol=1e-6):
            raise ValueError("Hessian must be symmetric.")
        H = (H + H.T) * .5
        diagonal = torch.diag(H)
        if (diagonal < 0).any():
            raise ValueError("Hessian has a negative diagonal.")
        dead = diagonal == 0
        H[dead, dead] = 1
        working[:, dead] = 0
        damping = rel_damp * torch.diag(H).mean()
        H.diagonal().add_(damping)
        try:
            inverse = torch.cholesky_inverse(torch.linalg.cholesky(H))
            inverse = torch.linalg.cholesky(inverse, upper=True)
        except RuntimeError as exc:
            raise RuntimeError("Hessian Cholesky failed; increase --rel-damp. No identity fallback is used.") from exc
        for start in range(0, columns, blocksize):
            end = min(start + blocksize, columns)
            block = working[:, start:end].clone()
            quantized = torch.zeros_like(block)
            errors = torch.zeros_like(block)
            for index in range(end - start):
                column = start + index
                current = block[:, index].clone()
                q = round_fp4(current, scales[:, column // group_size])
                # Future masked entries can drift in scratch space; enforce their
                # constraint when finalized, and compensate that drift as well.
                q.masked_fill_(original_mask[:, column], 0)
                quantized[:, index] = q
                error = (current - q) / inverse[column, column]
                block[:, index:].addmm_(error[:, None], inverse[column, column:end][None, :], alpha=-1)
                errors[:, index] = error
            working[:, start:end] = quantized
            if end < columns:
                working[:, end:].addmm_(errors, inverse[start:end, end:], alpha=-1)
    result = working.to(weight.dtype)
    statistics = mask_statistics(weight, result)
    if not statistics["mask_preserved"] or statistics["nonfinite"]:
        raise RuntimeError("Quantized output is non-finite or revived an original zero.")
    return result, {**statistics, "global_scale": global_scale,
                    "effective_scale_min": float(scales.min()),
                    "effective_scale_max": float(scales.max())}


@torch.no_grad()
def collect_hessian(model, module, dataloader, device):
    hessian = None
    tokens = 0

    def hook(_module, inputs, _output):
        nonlocal hessian, tokens
        x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float()
        if not torch.isfinite(x).all():
            raise ValueError("Calibration activations contain NaN or infinity.")
        if hessian is None:
            hessian = torch.zeros((x.shape[1], x.shape[1]), device=x.device)
        hessian.addmm_(x.T, x, alpha=2.0)
        tokens += x.shape[0]

    handle = module.register_forward_hook(hook)
    try:
        for batch in dataloader:
            inputs = {name: tensor.to(device) for name, tensor in batch.items() if name != "labels"}
            # Decoder only: no vocabulary-sized logits, loss, or KV cache.
            model.model(**inputs, use_cache=False, return_dict=False)
    finally:
        handle.remove()
    if hessian is None or tokens == 0:
        raise RuntimeError("No calibration inputs reached the selected factor.")
    hessian.div_(tokens)
    return hessian, tokens
