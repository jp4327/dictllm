import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import transformers

from dictllm import eval_lm_harness as evaluation


def test_defaults():
    args = evaluation.parse_args(["checkpoint"])
    assert tuple(args.tasks) == evaluation.DEFAULT_TASKS
    assert args.num_fewshot == 0
    assert args.batch_size == 1
    assert args.dtype == "auto"
    assert args.summary_metric == "acc"
    assert args.add_bos_token is False


@pytest.mark.parametrize("option,value,expected", [
    ("--add-bos-token", "true", True),
    ("--add-bos-token", "false", False),
    ("--add_bos_token", "False", False),
])
def test_bos_option(option, value, expected):
    assert evaluation.parse_args(["checkpoint", option, value]).add_bos_token is expected


@pytest.mark.parametrize("options", [
    ["--tasks", ","], ["--batch-size", "0"], ["--num-fewshot", "-1"],
    ["--limit", "0"], ["--max-length", "1"], ["--add-bos-token", "maybe"],
])
def test_invalid_options(options):
    with pytest.raises(SystemExit):
        evaluation.parse_args(["checkpoint", *options])


def test_native_model_uses_native_loader_and_stored_dtype(monkeypatch):
    calls = []
    model = torch.nn.Linear(2, 2)
    class Loader:
        @staticmethod
        def from_pretrained(source, **kwargs):
            calls.append((source, kwargs))
            return model
    monkeypatch.setattr(evaluation, "model_class", lambda architecture: Loader)
    config = SimpleNamespace(architectures=["DictLlamaForCausalLM"],
                             dictllm_enabled=True, model_type="llama")
    assert evaluation.load_model("checkpoint", config, "auto", "cpu") is model
    assert all(calls[0][1][key] == value for key, value in evaluation._dtype_kwargs("auto").items())
    assert "device_map" not in calls[0][1]
    assert not model.training


def test_dense_model_uses_auto_loader(monkeypatch):
    calls = []
    model = torch.nn.Linear(2, 2)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained",
                        lambda source, **kwargs: calls.append(kwargs) or model)
    config = SimpleNamespace(architectures=["LlamaForCausalLM"], model_type="llama")
    evaluation.load_model("dense", config, "float16", "cpu")
    assert all(calls[0][key] is value for key, value in evaluation._dtype_kwargs(torch.float16).items())


@pytest.mark.parametrize("config", [
    SimpleNamespace(architectures=["ShareLlamaForCausalLM"]),
    SimpleNamespace(architectures=["DictLlamaForCausalLM"]),
    SimpleNamespace(architectures=[], dictllm_quantization={"packed": True}),
])
def test_unsupported_layout_does_not_fall_back_to_dense(config):
    with pytest.raises(ValueError):
        evaluation.load_model("checkpoint", config, "auto", "cpu")


def test_offline_packed_checkpoint_rejected_before_loading(tmp_path):
    (tmp_path / "packed_nvfp4_manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="Packed NVFP4"):
        evaluation.main([str(tmp_path)])


def test_missing_output_checkpoint_rejected():
    with pytest.raises(FileNotFoundError):
        evaluation.main(["outputs/nonexistent_model/03_refit"])


@pytest.mark.parametrize("failure", [False, "recursion"])
def test_legacy_llama_tokenizer_fallback(monkeypatch, failure):
    def auto(*args, **kwargs):
        if failure == "recursion":
            raise RecursionError()
        return False
    calls = []
    tokenizer = SimpleNamespace(pad_token_id=2, bos_token_id=1, eos_token_id=2)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", auto)
    monkeypatch.setattr(transformers.LlamaTokenizer, "from_pretrained",
                        lambda *args, **kwargs: calls.append(kwargs) or tokenizer)
    assert evaluation.load_tokenizer("checkpoint", SimpleNamespace(model_type="llama")) is tokenizer
    assert calls[0] == dict(legacy=True, bos_token="<s>", eos_token="</s>", unk_token="<unk>")


def test_summary_does_not_confuse_acc_with_acc_norm(capsys):
    evaluation.print_summary({"hellaswag": {"acc,none": 0.2, "acc_norm,none": 0.4}}, "mixed")
    output = capsys.readouterr().out
    assert "acc_norm" in output and "0.4000" in output
    evaluation.print_summary({"hellaswag": {"acc,none": 0.2}}, "mixed")
    assert "N/A" in capsys.readouterr().out


def test_acc_summary_never_substitutes_acc_norm(capsys):
    evaluation.print_summary({
        "hellaswag": {"acc,none": 0.2, "acc_norm,none": 0.4},
        "piqa": {"acc_norm,none": 0.8},
    }, "acc")
    output = capsys.readouterr().out
    assert "0.2000" in output
    assert "0.4000" not in output and "0.8000" not in output
    assert "acc_norm" not in output
    assert "N/A" in output


def test_mathqa_loader_restored_after_error(monkeypatch):
    import datasets
    calls = []
    original = lambda path, **kwargs: calls.append((path, kwargs))
    monkeypatch.setattr(datasets, "__version__", "2.19.0")
    monkeypatch.setattr(datasets, "load_dataset", original)
    with pytest.raises(RuntimeError):
        with evaluation.mathqa_compatibility(["mathqa"]):
            datasets.load_dataset("math_qa")
            datasets.load_dataset("other")
            raise RuntimeError("evaluation failed")
    assert calls == [("math_qa", {"trust_remote_code": True}), ("other", {})]
    assert datasets.load_dataset is original


@pytest.mark.parametrize("bos_options,expected_bos", [([], False), (["--add-bos-token", "true"], True)])
@pytest.mark.parametrize("packed", [False, True])
def test_evaluation_saves_full_results_and_provenance(tmp_path, monkeypatch, bos_options, expected_bos, packed):
    import datasets
    checkpoint = tmp_path / "03_refit"
    checkpoint.mkdir()
    harness_calls = []
    evaluation_calls = []
    fake_lm_eval = ModuleType("lm_eval")
    def simple_evaluate(**kwargs):
        evaluation_calls.append(kwargs)
        return {"results": {"openbookqa": {"acc,none": 0.5, "acc_norm,none": 0.6}},
                "versions": {"openbookqa": 1}}
    fake_lm_eval.simple_evaluate = simple_evaluate
    fake_huggingface = ModuleType("lm_eval.models.huggingface")
    fake_huggingface.HFLM = lambda **kwargs: harness_calls.append(kwargs) or SimpleNamespace(
        add_bos_token=kwargs["add_bos_token"],
    )
    monkeypatch.setitem(sys.modules, "lm_eval", fake_lm_eval)
    monkeypatch.setitem(sys.modules, "lm_eval.models", ModuleType("lm_eval.models"))
    monkeypatch.setitem(sys.modules, "lm_eval.models.huggingface", fake_huggingface)
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda source: object())
    monkeypatch.setattr(evaluation, "load_model", lambda *args: torch.nn.Linear(2, 2))
    monkeypatch.setattr(evaluation, "load_tokenizer", lambda *args: SimpleNamespace(
        bos_token_id=1, eos_token_id=2,
    ))
    original_dataset_loader = datasets.load_dataset
    argv = [str(checkpoint), "--tasks", "openbookqa", "--limit", "2", "--device", "cpu", *bos_options]
    if packed:
        from dictllm.nvfp4 import evaluation as packed_evaluation

        (checkpoint / "packed_nvfp4_manifest.json").write_text("{}")
        monkeypatch.setattr(packed_evaluation, "read_evaluation_config", lambda _: object())
        monkeypatch.setattr(packed_evaluation, "load_evaluation_model", lambda *args: torch.nn.Linear(2, 2))
        argv.extend(["--packed-runtime", "dequantize", "--cache-dequantized"])
    evaluation.main(argv)
    report = json.loads((checkpoint / "lm_harness_results.json").read_text())
    assert report["results"]["openbookqa"] == {"acc,none": 0.5, "acc_norm,none": 0.6}
    assert report["versions"] == {"openbookqa": 1}
    assert report["dictllm_evaluation"]["limit"] == 2
    assert report["dictllm_evaluation"]["bos_token_id"] == 1
    assert harness_calls[0]["batch_size"] == 1
    assert harness_calls[0]["add_bos_token"] is expected_bos
    assert report["dictllm_evaluation"]["add_bos_token"] is expected_bos
    assert report["dictllm_evaluation"]["add_bos_token_resolved"] is expected_bos
    if packed:
        assert report["dictllm_evaluation"]["runtime"] == "dequantize"
        assert report["dictllm_evaluation"]["activation_quantization"] is False
        assert report["dictllm_evaluation"]["cache_dequantized"] is True
    assert evaluation_calls[0]["num_fewshot"] == 0
    assert evaluation_calls[0]["log_samples"] is False
    assert datasets.load_dataset is original_dataset_loader
    with pytest.raises(FileExistsError):
        evaluation.main(argv)
