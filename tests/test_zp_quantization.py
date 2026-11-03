from pathlib import Path

import pytest
import torch

from dictllm import nvfp4

HERE = Path(nvfp4.__file__).resolve().parent

from dictllm.nvfp4.quantization import (collect_hessian, group_scales, mask_statistics,
                          quantize_weight, reconstruction_errors, round_fp4)
from dictllm.nvfp4.verify_checkpoint import verify_checkpoint, write_json


@pytest.mark.parametrize("method", ["rtn", "masked_gptq"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("blocksize", [1, 3, 8])
def test_zero_preservation_across_blocks_and_casts(method, dtype, blocksize):
    generator = torch.Generator().manual_seed(9)
    weight = torch.randn(4, 7, generator=generator).to(dtype)
    weight[torch.rand(4, 7, generator=generator) < .5] = 0
    weight[0] = 0
    x = torch.randn(16, 7, generator=generator)
    hessian = x.T @ x
    before, H_before = weight.clone(), hessian.clone()
    result, stats = quantize_weight(weight, method=method, hessian=hessian,
                                   group_size=4, blocksize=blocksize)
    assert torch.equal(weight, before) and torch.equal(hessian, H_before)
    assert torch.equal(result[weight == 0], torch.zeros_like(result[weight == 0]))
    assert result.shape == weight.shape and result.dtype == dtype
    assert stats["revived_zeros"] == stats["nonfinite"] == 0


def test_fp4_grid_and_nearest_even_ties():
    grid = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
    assert torch.equal(round_fp4(grid, torch.ones_like(grid)), grid)
    boundaries = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.])
    expected = torch.tensor([0., 1., 1., 2., 2., 4., 4.])
    assert torch.equal(round_fp4(boundaries, torch.ones_like(boundaries)), expected)
    assert torch.equal(round_fp4(-boundaries, torch.ones_like(boundaries)), -expected)


def test_global_scale_and_tiny_nonzero_groups():
    weight = torch.tensor([[6., 0., 1e-8, 0., 0., 0.]])
    scales, global_scale = group_scales(weight, 2)
    assert global_scale == 448
    assert scales[0, 1] > 0 and scales[0, 1] < 1
    assert scales[0, 2] == 1
    assert scales[0, 0] == 1


def test_masked_gptq_handles_the_identity_counterexample():
    weight = torch.tensor([[.30, .17, 0.]])
    H = torch.tensor([[1., 0., 0.], [0., 1., .9], [0., .9, 1.]])
    result, _ = quantize_weight(weight, hessian=H, group_size=3, scale_precision="fp32", rel_damp=0)
    assert result[0, 2] == 0
    assert not torch.equal(result, weight)


def test_gptq_with_diagonal_hessian_matches_rtn():
    weight = torch.tensor([[0., 1.17, 1.23, 6.], [-1.19, 0., .12, 3.]])
    rtn, _ = quantize_weight(weight, method="rtn", group_size=4)
    gptq, _ = quantize_weight(weight, hessian=torch.eye(4), group_size=4, blocksize=2)
    assert torch.equal(rtn, gptq)


def test_gptq_blocking_does_not_drop_global_compensation():
    generator = torch.Generator().manual_seed(11)
    weight = torch.randn(3, 9, generator=generator)
    weight[:, [2, 5]] = 0
    x = torch.randn(30, 9, generator=generator)
    H = x.T @ x
    small, _ = quantize_weight(weight, hessian=H, blocksize=1, group_size=3)
    large, _ = quantize_weight(weight, hessian=H, blocksize=9, group_size=3)
    assert torch.allclose(small, large, atol=1e-6)


def test_metrics_equal_explicit_activations():
    x = torch.tensor([[1., 2., 3.], [2., -1., .5]])
    weight = torch.tensor([[0., 1., 2.], [3., 0., -1.]])
    result, _ = quantize_weight(weight, method="rtn", group_size=3)
    metric = reconstruction_errors(weight, result, 2 * x.T @ x / len(x), chunk_rows=1)
    reference = torch.linalg.vector_norm(x @ (weight - result).T) / torch.linalg.vector_norm(x @ weight.T)
    assert metric["activation_relative_frobenius_error"] == pytest.approx(float(reference), abs=1e-6)
    assert reconstruction_errors(torch.zeros_like(weight), torch.zeros_like(weight))["weight_relative_frobenius_error"] is None


def test_new_zeros_are_allowed():
    weight = torch.tensor([[0., .01, 6.]])
    result, stats = quantize_weight(weight, method="rtn", group_size=3)
    assert stats["additional_zeros"] == 1 and stats["mask_preserved"]
    assert stats["sparsity_after"] > stats["sparsity_before"]


def test_validation_and_no_silent_identity_fallback():
    weight = torch.ones(2, 2)
    with pytest.raises(ValueError, match="requires"):
        quantize_weight(weight)
    with pytest.raises(RuntimeError, match="No identity fallback"):
        quantize_weight(weight, hessian=torch.tensor([[1., 2.], [2., 1.]]), rel_damp=0)
    with pytest.raises(ValueError, match="NaN"):
        quantize_weight(weight * float("nan"), method="rtn")
    with pytest.raises(ValueError, match="shape/device"):
        quantize_weight(weight, hessian=torch.eye(3))


def test_hessian_decoder_only_shared_module_and_hook_cleanup():
    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.factor = torch.nn.Linear(2, 2, bias=False)

        def forward(self, input_ids, use_cache, return_dict):
            assert not use_cache and not return_dict
            self.factor(input_ids)
            return self.factor(input_ids * 2)

    model = torch.nn.Module()
    model.model = Decoder()
    x = torch.tensor([[1., 2.], [3., 4.]])
    H, rows = collect_hessian(model, model.model.factor, [{"input_ids": x, "labels": x}], "cpu")
    assert rows == 4 and torch.equal(H, 2 * (x.T @ x + (x * 2).T @ (x * 2)) / 4)
    assert len(model.model.factor._forward_hooks) == 0
    with pytest.raises(TypeError):
        collect_hessian(model, model.model.factor, [{"bad_input": x}], "cpu")
    assert len(model.model.factor._forward_hooks) == 0


@pytest.mark.parametrize("serialization", ["bin", "safe"])
def test_saved_verifier_detects_mask_and_alias_corruption(tmp_path, serialization):
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    weight = torch.tensor([[0., .4, 1.], [0., 0., 6.]])
    result, _ = quantize_weight(weight, method="rtn", group_size=3)
    aliases = ["coefficient.weight", "alias.weight"]
    record = {"name": aliases[0], "tensor_type": "coefficient", "shape": list(weight.shape), "aliases": aliases}

    def save(directory, matrix, corrupt_alias=False):
        tensors = {name: matrix.clone() for name in aliases}
        if corrupt_alias:
            tensors[aliases[1]][0, 0] = 1
        if serialization == "bin":
            torch.save(tensors, directory / "pytorch_model.bin")
        else:
            from safetensors.torch import save_file
            save_file(tensors, directory / "model.safetensors")

    save(source, weight)
    save(output, result)
    assert verify_checkpoint(source, output, [record], chunk_rows=1)["passed"]
    save(output, result, corrupt_alias=True)
    assert not verify_checkpoint(source, output, [record])["passed"]
    bad = result.clone()
    bad[0, 0] = .5
    save(output, bad)
    assert verify_checkpoint(source, output, [record])["total_revived_zeros"] == 1


def test_reports_refuse_overwrite(tmp_path):
    path = tmp_path / "report.json"
    write_json(path, {"passed": True})
    with pytest.raises(FileExistsError):
        write_json(path, {"passed": False})

