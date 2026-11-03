"""Independent CPU reference checks for the simulated NVFP4 scale hierarchy.

Does not import the quantizer or QuTLASS. Scales are reconstructed from the
source checkpoint because historical floating-point exports omit scale arrays.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


DTYPES = {"torch.float32": torch.float32, "torch.float16": torch.float16,
          "torch.bfloat16": torch.bfloat16, "torch.float64": torch.float64}
MAGNITUDES = (0., .5, 1., 1.5, 2., 3., 4., 6.)


def close_scale(actual, expected):
    return math.isfinite(actual) and math.isclose(actual, expected, rel_tol=5e-7, abs_tol=0)


def prepare_reference(store, record, settings, chunk_rows):
    name = record["name"]
    source_dtype = DTYPES.get(record.get("dtype"))
    if source_dtype is None:
        raise ValueError(f"{name}: missing or unsupported recorded weight dtype")
    maximum = torch.tensor(0., dtype=torch.float32)
    for start in range(0, record["shape"][0], chunk_rows):
        weight = store.rows(name, start, start + chunk_rows).to(source_dtype).float()
        if not torch.isfinite(weight).all():
            raise ValueError(f"{name}: source contains non-finite entries")
        maximum = torch.maximum(maximum, weight.abs().max())
    expected = float(448.0 * 6.0 / maximum) if maximum > 0 else 1.0
    actual = float(record.get("global_scale", float("nan")))
    if not close_scale(actual, expected) or actual <= 0:
        raise ValueError(f"{name}: recorded tensor-global scale {actual} differs from expected {expected}")
    return {"source_dtype": source_dtype, "group_size": settings["w_group_size"],
            "global_scale": actual, "expected_global_scale": expected,
            "source_amax": float(maximum)}


def scan_levels(before, after, reference, *, return_details=False):
    """Check the E2M1 grid; optionally expose reconstructed tensors for plots.

    Details are CPU tensors, not serialized scale metadata. The default result
    remains JSON-serializable for the checkpoint verifier.
    """
    group_size = reference["group_size"]
    original = before.to(reference["source_dtype"]).float()
    padded = F.pad(original, (0, (-original.shape[1]) % group_size))
    amax = padded.reshape(before.shape[0], -1, group_size).abs().amax(-1)
    global_scale = torch.tensor(reference["global_scale"], dtype=torch.float32)
    ideal = (amax / 6.0 * global_scale).clamp(max=448)
    encoded = ideal.to(torch.float8_e4m3fn).float()
    clamped = (amax > 0) & (encoded == 0)
    encoded[clamped] = 2.0 ** -9
    scales = encoded / global_scale
    # An all-zero source group must stay zero. Its arbitrary effective scale
    # of 1 in the quantizer need not be encoded; zero/any finite scale is zero.
    scales[amax == 0] = 1
    finite_scales = torch.isfinite(encoded).all() and torch.isfinite(scales).all()
    if not finite_scales or (scales <= 0).any() or (encoded < 0).any() or (encoded > 448).any():
        raise ValueError("Invalid reconstructed E4M3/effective group scales")
    scales_expanded, indices, difference, valid = check_grid(after, scales, group_size)
    # CUDA optimizes division by the CPU scalar 6 into multiplication by its
    # FP32 reciprocal. Check that explicit primitive as a second reference,
    # not a widened grid tolerance. The two can cross an FP8 rounding midpoint.
    alternative = (amax * torch.tensor(1.0 / 6.0) * global_scale).clamp(max=448)
    alternative = alternative.to(torch.float8_e4m3fn).float()
    alternative[(amax > 0) & (alternative == 0)] = 2.0 ** -9
    ambiguous = alternative != encoded
    used_alternative = torch.zeros_like(ambiguous)
    if ambiguous.any():
        alternative_scales = alternative / global_scale
        alternative_scales[amax == 0] = 1
        alt_expanded, alt_indices, alt_difference, alt_valid = check_grid(after, alternative_scales, group_size)
        padding = (-after.shape[1]) % group_size
        group_valid = F.pad(valid, (0, padding), value=True).reshape(before.shape[0], -1, group_size).all(-1)
        alt_group_valid = F.pad(alt_valid, (0, padding), value=True).reshape(before.shape[0], -1, group_size).all(-1)
        used_alternative = ambiguous & ~group_valid & alt_group_valid
        selected = used_alternative.repeat_interleave(group_size, dim=1)[:, :after.shape[1]]
        scales_expanded = torch.where(selected, alt_expanded, scales_expanded)
        indices = torch.where(selected, alt_indices, indices)
        difference = torch.where(selected, alt_difference, difference)
        valid = torch.where(selected, alt_valid, valid)
        scales = torch.where(used_alternative, alternative_scales, scales)
        encoded = torch.where(used_alternative, alternative, encoded)
    zero_groups = (amax == 0).repeat_interleave(group_size, dim=1)[:, :after.shape[1]]
    valid &= ~zero_groups | (after == 0)
    histogram = torch.bincount(indices[valid].flatten(), minlength=8).tolist()
    normalized = difference / scales_expanded.double()
    finite_residuals = normalized[torch.isfinite(normalized)]
    result = {
        "groups_checked": encoded.numel(), "all_zero_groups": int((amax == 0).sum()),
        "cpu_cuda_scale_boundary_groups": int(ambiguous.sum()),
        "cuda_reciprocal_scale_groups": int(used_alternative.sum()),
        "underflow_clamped_groups": int(clamped.sum()), "off_grid_elements": int((~valid).sum()),
        "max_normalized_grid_residual": float(finite_residuals.max()) if finite_residuals.numel() else 0.,
        "effective_scale_min": float(scales.min()), "effective_scale_max": float(scales.max()),
        "encoded_e4m3_scale_min": float(encoded.min()), "encoded_e4m3_scale_max": float(encoded.max()),
        "magnitude_code_histogram": histogram,
    }
    if return_details:
        result["details"] = {
            "encoded_scales": encoded,
            "effective_scales": scales,
            "all_zero_groups": amax == 0,
            "magnitude_indices": indices,
            "valid": valid,
            "cuda_reciprocal_groups": used_alternative,
            "scale_boundary_groups": ambiguous,
        }
    return result


def check_grid(after, scales, group_size):
    scales_expanded = scales.repeat_interleave(group_size, dim=1)[:, :after.shape[1]]
    saved = after.float()
    boundaries = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.])
    indices = torch.bucketize((saved.abs() / scales_expanded).contiguous(), boundaries)
    grid = torch.tensor(MAGNITUDES)
    expected_fp32 = grid[indices] * torch.sign(saved) * scales_expanded
    expected = expected_fp32.to(after.dtype)
    difference = (after.double() - expected.double()).abs()
    if after.dtype == torch.float32:
        tolerance = 8 * torch.finfo(torch.float32).eps * expected.abs().float()
        valid = difference <= tolerance
    else:
        valid = after == expected
    valid &= torch.isfinite(after)
    return scales_expanded, indices, difference, valid


def validate_settings(settings):
    required = {"format": "simulated_nvfp4", "w_bits": 4, "w_group_size": 16,
                "scale_precision": "e4m3", "transform_class": "identity",
                "packed_storage": False, "scale_policy": "tensor_global_normalized_block_amax",
                "underflow_policy": "nonzero_block_scale_clamped_to_min_e4m3_subnormal"}
    for field, expected in required.items():
        if settings.get(field) != expected:
            raise ValueError(f"NVFP4 hierarchy check expects {field}={expected!r}")
