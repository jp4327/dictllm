from types import SimpleNamespace

import torch
import torch.nn as nn

from dictllm.factorization import (
    _cholesky_scaling,
    compute_rank,
    factorize_projection_group,
)
from dictllm.modeling import (
    FactorizedLinear,
    build_groups,
    get_projection,
    replace_projection_group,
    topology_counts,
)
from dictllm.pruning import compute_hessian_inverse, faster_hessian_prune_from_inverse
from dictllm.refit import mod_update_dictionary
from dictllm.calibration import (
    cache_is_complete,
    collect_grouped_covariances,
    load_group_covariance,
)


class DummyAttention(nn.Module):
    def __init__(self, size=2):
        super().__init__()
        self.q_proj = nn.Linear(size, size, bias=False)
        self.k_proj = nn.Linear(size, size, bias=False)
        self.v_proj = nn.Linear(size, size, bias=False)
        self.o_proj = nn.Linear(size, size, bias=False)


class DummyMLP(nn.Module):
    def __init__(self, size=2):
        super().__init__()
        self.gate_proj = nn.Linear(size, size, bias=False)
        self.up_proj = nn.Linear(size, size, bias=False)
        self.down_proj = nn.Linear(size, size, bias=False)


class DummyLayer(nn.Module):
    def __init__(self, size=2):
        super().__init__()
        self.self_attn = DummyAttention(size)
        self.mlp = DummyMLP(size)


class DummyBackbone(nn.Module):
    def __init__(self, layers, size=2):
        super().__init__()
        self.layers = nn.ModuleList([DummyLayer(size) for _ in range(layers)])


class DummyModel(nn.Module):
    def __init__(self, layers=1, size=2):
        super().__init__()
        self.model = DummyBackbone(layers, size)
        self.config = SimpleNamespace(num_hidden_layers=layers)


def test_rank_matches_basis_sharing_formula():
    assert compute_rank(4096, 4096, 2, 0.30) == 1911
    assert compute_rank(4096, 11008, 2, 0.30) == 2417


def test_relative_covariance_damping_is_applied_when_requested():
    _, damping = _cholesky_scaling(
        torch.eye(3), relative_damping=1e-6, fallback_shift=7e-6
    )
    assert damping == 1e-6


def test_factorization_reconstructs_full_rank_projection():
    torch.manual_seed(1)
    model = DummyModel(layers=1, size=2)
    original = get_projection(model, 0, "q").weight.detach().clone()
    summary = factorize_projection_group(
        model=model,
        target="q",
        group_index=0,
        layers=[0],
        rank=2,
        covariance=torch.eye(2),
        factorization={
            "device": "cpu",
            "compute_dtype": "float32",
            "covariance_relative_damping": 0.0,
            "fallback_eigenvalue_shift": 7e-6,
        },
    )
    projection = get_projection(model, 0, "q")
    assert isinstance(projection, FactorizedLinear)
    reconstructed = projection.coefficient.weight @ projection.dictionary.weight
    assert summary["rank"] == 2
    assert torch.allclose(reconstructed, original, atol=1e-5, rtol=1e-5)


def test_basis_sharing_factorization_matches_original_factors():
    torch.manual_seed(7)
    model = DummyModel(layers=2, size=4)
    weights = torch.cat([
        get_projection(model, layer, "q").weight.detach().T for layer in [0, 1]
    ], dim=1).double()
    covariance = torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    scaling = torch.linalg.cholesky(covariance.double()).T
    left, singular_values, right = torch.svd(scaling @ weights)
    expected_dictionary = (
        (torch.linalg.inv(scaling) @ left) @ torch.diag(singular_values)
    ).float()[:, :2]
    expected_coefficients = right.T.float()[:2]
    factorize_projection_group(
        model, "q", 0, [0, 1], 2, covariance,
        {
            "device": "cpu", "compute_dtype": "float64",
            "numerical_method": "basis_sharing",
            "covariance_relative_damping": 0.0,
            "fallback_eigenvalue_shift": 7e-6,
        },
    )
    for layer in [0, 1]:
        projection = get_projection(model, layer, "q")
        assert torch.equal(projection.dictionary.weight.T, expected_dictionary)
        assert torch.equal(
            projection.coefficient.weight.T,
            expected_coefficients[:, layer * 4:(layer + 1) * 4],
        )


def test_fresh_covariance_matches_per_layer_accumulation(tmp_path):
    class Backbone(DummyBackbone):
        def forward(self, input_ids, **kwargs):
            for layer in self.layers:
                input_ids = layer.self_attn.k_proj(input_ids)
            return input_ids

    torch.manual_seed(8)
    model = DummyModel(layers=2, size=2)
    model.model = Backbone(layers=2, size=2)
    batches = [torch.randn(3, 2) * scale for scale in [1e3, 1e-2, 1.0]]
    expected = [None, None]
    with torch.no_grad():
        for batch in batches:
            current = batch
            for layer, module in enumerate(model.model.layers):
                gram = current.T @ current
                expected[layer] = gram if expected[layer] is None else expected[layer] + gram
                current = module.self_attn.k_proj(current)
    groups = {"attention_input": [[0, 1]]}
    collect_grouped_covariances(
        model, [{"input_ids": batch} for batch in batches], groups, tmp_path, {},
    )
    actual, rows = load_group_covariance(tmp_path, "q", [0, 1], "dictllm", "sum")
    assert torch.equal(actual, expected[0] + expected[1])
    assert rows == 18
    assert cache_is_complete(tmp_path, groups, {}, "dictllm")


def test_faster_hessian_reaches_per_column_sparsity():
    torch.manual_seed(2)
    coefficient = torch.randn(8, 5)
    inverse = compute_hessian_inverse(torch.eye(8), damping=1e-4, method="cholesky")
    pruned = faster_hessian_prune_from_inverse(
        coefficient, inverse, sparsity=0.5, block_size=3
    )
    assert torch.isfinite(pruned).all()
    assert torch.count_nonzero(pruned == 0, dim=0).tolist() == [4] * 5


def test_mod_update_reduces_weight_reconstruction_error():
    torch.manual_seed(3)
    dense = torch.randn(9, 12)
    coefficient = torch.randn(4, 12)
    before_dictionary = torch.randn(9, 4)
    before = torch.mean((dense - before_dictionary @ coefficient).square())
    after_dictionary = mod_update_dictionary(dense, coefficient, damping=1e-4)
    after = torch.mean((dense - after_dictionary @ coefficient).square())
    assert after < before


def test_reference_topology_visits_each_module_once():
    model = DummyModel(layers=32, size=2)
    targets = ["q", "k", "v", "gate", "up", "o", "down"]
    groups = {}
    ranks = {}
    for target in targets:
        group_size = 1 if target in {"o", "down"} else 2
        groups[target] = build_groups(32, group_size)
        ranks[target] = []
        for group_index, layers in enumerate(groups[target]):
            replace_projection_group(model, target, group_index, layers, rank=1)
            ranks[target].append(1)
    model.config.dictllm_targets = targets
    model.config.dictllm_groups = groups
    model.config.dictllm_ranks = ranks
    assert topology_counts(model) == {"dictionaries": 144, "coefficients": 224}
