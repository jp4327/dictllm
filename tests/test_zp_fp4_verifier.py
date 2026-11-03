from pathlib import Path
import sys

import pytest
import torch


from dictllm.nvfp4.quantization import quantize_weight
from dictllm.nvfp4.verify_checkpoint import verify_checkpoint


SETTINGS = {
    "format": "simulated_nvfp4", "w_bits": 4, "w_group_size": 16,
    "scale_precision": "e4m3", "transform_class": "identity", "packed_storage": False,
    "scale_policy": "tensor_global_normalized_block_amax",
    "underflow_policy": "nonzero_block_scale_clamped_to_min_e4m3_subnormal",
}


def fixture(tmp_path, dtype=torch.float32, method="rtn"):
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    torch.manual_seed(82)
    weight = torch.randn(3, 35).to(dtype)
    weight[0, :16] = 0
    weight[1, :16] *= 1e-7
    weight[2, ::3] = 0
    result, stats = quantize_weight(weight, method=method, hessian=torch.eye(35))
    torch.save({"coefficient.weight": weight}, source / "pytorch_model.bin")
    torch.save({"coefficient.weight": result}, output / "pytorch_model.bin")
    record = {"name": "coefficient.weight", "tensor_type": "coefficient", "shape": list(weight.shape),
              "dtype": str(dtype), **stats}
    return source, output, record, result


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("method", ["rtn", "masked_gptq"])
def test_levels_scales_partial_groups_and_zero_groups(tmp_path, dtype, method):
    source, output, record, _ = fixture(tmp_path, dtype, method)
    report = verify_checkpoint(source, output, [record], chunk_rows=1, quantization_summary=SETTINGS)
    assert report["passed"], report["errors"]
    check = report["per_tensor"][0]["nvfp4_grid"]
    assert check["groups_checked"] == 9
    assert check["all_zero_groups"] >= 1
    assert check["underflow_clamped_groups"] >= 1
    assert sum(check["magnitude_code_histogram"]) == 105
    assert report["total_off_grid_elements"] == 0
    assert report["totals_by_target"]["coefficient"]["revived_zeros"] == 0


def test_invalid_fp4_value_is_rejected(tmp_path):
    source, output, record, result = fixture(tmp_path)
    result[0, 20] = .1234567
    torch.save({"coefficient.weight": result}, output / "pytorch_model.bin")
    report = verify_checkpoint(source, output, [record], quantization_summary=SETTINGS)
    assert not report["passed"] and report["total_off_grid_elements"] >= 1


def test_plot_details_reconstruct_weights_and_keep_default_json_safe(tmp_path):
    import json
    from dictllm.nvfp4.fp4_reference import MAGNITUDES, scan_levels
    source, _, record, after = fixture(tmp_path)
    before = torch.load(source / "pytorch_model.bin", weights_only=True)["coefficient.weight"]
    reference = {"source_dtype": torch.float32, "group_size": 16,
                 "global_scale": record["global_scale"]}
    checked = scan_levels(before, after, reference, return_details=True)
    details = checked.pop("details")
    assert checked == scan_levels(before, after, reference)
    json.dumps(checked, allow_nan=False)
    assert details["encoded_scales"].shape == (3, 3)
    assert details["valid"].all()
    assert details["all_zero_groups"][0, 0]
    scales = details["effective_scales"].repeat_interleave(16, dim=1)[:, :35]
    reconstructed = torch.tensor(MAGNITUDES)[details["magnitude_indices"]] * after.sign() * scales
    assert torch.equal(reconstructed, after)


def test_global_scale_tampering_is_rejected(tmp_path):
    source, output, record, _ = fixture(tmp_path)
    record["global_scale"] *= .9
    report = verify_checkpoint(source, output, [record], quantization_summary=SETTINGS)
    assert not report["passed"]
    assert any("tensor-global scale" in error for error in report["errors"])


def test_scale_range_tampering_is_rejected(tmp_path):
    source, output, record, _ = fixture(tmp_path)
    record["effective_scale_min"] *= .9
    report = verify_checkpoint(source, output, [record], quantization_summary=SETTINGS)
    assert not report["passed"]
    assert any("effective_scale_min" in error for error in report["errors"])


@pytest.mark.parametrize("override", [{"w_group_size": 32}, {"scale_precision": "fp32"},
                                    {"transform_class": "hadamard"}, {"w_bits": 8}])
def test_rejects_non_nvfp4_settings(tmp_path, override):
    source, output, record, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="expects"):
        verify_checkpoint(source, output, [record], quantization_summary={**SETTINGS, **override})


def test_nonfinite_output_generates_serializable_failure(tmp_path):
    import json
    source, output, record, result = fixture(tmp_path)
    result[0, 20] = float("nan")
    torch.save({"coefficient.weight": result}, output / "pytorch_model.bin")
    report = verify_checkpoint(source, output, [record], quantization_summary=SETTINGS)
    assert not report["passed"]
    json.dumps(report, allow_nan=False)


def test_cuda_scalar_reciprocal_scale_boundary_is_not_a_false_failure():
    from dictllm.nvfp4.fp4_reference import scan_levels
    # Actual boundary encountered in the user's CUDA export: division on CPU
    # gives 22.999998 -> E4M3 22; CUDA reciprocal gives 23 -> nearest-even 24.
    amax = torch.tensor(0.0337638184428215)
    # Choose the exact measured global scale through the midpoint equation.
    G = torch.tensor(23.0) / (amax * torch.tensor(1 / 6.0))
    before = torch.zeros(1, 16)
    before[0, 0] = amax
    encoded_cpu = (amax / 6 * G).to(torch.float8_e4m3fn).float()
    encoded_cuda = (amax * torch.tensor(1 / 6.0) * G).to(torch.float8_e4m3fn).float()
    assert encoded_cpu != encoded_cuda
    after = torch.tensor([[.5, 1., 1.5, 2., 3., 4., 6., 0.] * 2]) * (encoded_cuda / G)
    result = scan_levels(before, after, {"group_size": 16, "source_dtype": torch.float32,
                                         "global_scale": float(G)})
    assert result["off_grid_elements"] == 0 and result["cuda_reciprocal_scale_groups"] == 1
    # Mixing two scales inside one group must still fail.
    after[0, 0] = .5 * (encoded_cpu / G)
    assert scan_levels(before, after, {"group_size": 16, "source_dtype": torch.float32,
                                       "global_scale": float(G)})["off_grid_elements"] > 0
