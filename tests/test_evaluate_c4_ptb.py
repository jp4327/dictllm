import json
from types import SimpleNamespace

import pytest
import torch

from dictllm import evaluate_c4_ptb as evaluation


def test_ptb_preprocessing(tmp_path):
    path = tmp_path / "ptb.test.txt"
    path.write_text(" first sentence \n\tsecond sentence  \n", encoding="utf-8")
    args = evaluation.parse_args([str(tmp_path), "--ptb-data-file", str(path)])
    text, info = evaluation.prepare_test_text("ptb", args)
    assert text == "first sentence\n\nsecond sentence"
    assert info["records"] == 2
    assert info["strip_sentences"] is True


def test_ptb_whitespace_protocol_preserves_spaces_and_empty_sentences(tmp_path):
    path = tmp_path / "ptb.test.txt"
    path.write_text(" first sentence \n\n\tsecond sentence  \n", encoding="utf-8")
    args = evaluation.parse_args([
        str(tmp_path), "--ptb-data-file", str(path),
        "--ptb-protocol", "whitespace_preserved",
    ])
    text, info = evaluation.prepare_test_text("ptb", args)
    assert text == " first sentence \n\n\n\n\tsecond sentence  "
    assert info["records"] == 3
    assert info["strip_sentences"] is False
    assert info["protocol"] == "preserve_sentence_whitespace_double_newline_join"
    assert info["encoding"] == "utf-8"
    assert info["sentence_separator"] == "\n\n"
    assert len(info["data_sha256"]) == 64


def test_whitespace_result_is_separate_and_records_tokenization(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "llama", "dictllm_enabled": True,
    }))
    original = tmp_path / "final_ppl_ptb.json"
    original.write_text('{"perplexity": 134.39}')
    ptb = tmp_path / "ptb.txt"
    ptb.write_text(" some text \n", encoding="utf-8")
    calls = []

    class Tokenizer:
        bos_token_id = 0

        def __call__(self, text, **kwargs):
            calls.append((text, kwargs))
            return SimpleNamespace(input_ids=torch.arange(8).reshape(1, 8))

    monkeypatch.setattr(evaluation, "load_tokenizer", lambda _: Tokenizer())
    monkeypatch.setattr(evaluation, "load_checkpoint_model", lambda *args: object())
    monkeypatch.setattr(evaluation, "compute_perplexity", lambda *args, **kwargs: 48.94)
    argv = [str(tmp_path), "--datasets", "ptb", "--ptb-data-file", str(ptb),
            "--ptb-protocol", "whitespace_preserved"]
    evaluation.main(argv)
    output = tmp_path / "final_ppl_ptb_whitespace_preserved.json"
    report = json.loads(output.read_text())
    assert report["perplexity"] == 48.94
    assert report["token_count"] == 8
    assert report["bos_token_id"] == 0
    assert report["add_special_tokens"] is True
    assert len(report["token_sha256"]) == 64
    assert calls == [(" some text ", {"return_tensors": "pt", "add_special_tokens": True})]
    with pytest.raises(FileExistsError):
        evaluation.main(argv)
    evaluation.main([*argv, "--overwrite"])
    assert original.read_text() == '{"perplexity": 134.39}'


def test_c4_preprocessing_and_limit(tmp_path):
    path = tmp_path / "c4.json"
    path.write_text('\n'.join(json.dumps({"text": text}) for text in ["one", "two", "three"]))
    args = evaluation.parse_args([
        str(tmp_path), "--c4-data-file", str(path), "--c4-documents", "2",
        "--dataset-cache-dir", str(tmp_path / "cache"),
    ])
    text, info = evaluation.prepare_test_text("c4", args)
    assert text == "one\n\ntwo"
    assert info["records"] == 2


@pytest.mark.parametrize("options", [
    ["--context-length", "1"], ["--stride", "0"],
    ["--stride", "2049"], ["--c4-documents", "0"],
])
def test_invalid_options(options):
    with pytest.raises(SystemExit):
        evaluation.parse_args(["checkpoint", *options])


def test_both_datasets_load_model_once_and_save_results(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({
        "model_type": "llama", "dictllm_enabled": True,
    }))
    c4 = tmp_path / "c4.json"
    ptb = tmp_path / "ptb.txt"
    c4.touch()
    ptb.touch()
    loads = []
    monkeypatch.setattr(evaluation, "load_checkpoint_model",
                        lambda *args: loads.append(args) or object())
    monkeypatch.setattr(evaluation, "load_tokenizer", lambda _: (
        lambda text, **kwargs: SimpleNamespace(input_ids=torch.ones((1, 8)))
    ))
    monkeypatch.setattr(evaluation, "prepare_test_text",
                        lambda name, args: ("test", {"records": 2, "data_file": name}))
    monkeypatch.setattr(evaluation, "compute_perplexity", lambda *args, **kwargs: 8.5)
    argv = [str(checkpoint), "--c4-data-file", str(c4), "--ptb-data-file", str(ptb)]
    evaluation.main(argv)
    assert loads == [(checkpoint, "llama", "bfloat16")]
    for name in ("c4", "ptb"):
        report = json.loads((checkpoint / f"final_ppl_{name}.json").read_text())
        assert report["dataset_name"] == name
        assert report["perplexity"] == 8.5
        assert report["token_count"] == 8
    with pytest.raises(FileExistsError):
        evaluation.main(argv)
    assert len(loads) == 1


def test_single_dataset_does_not_require_other_data(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "llama", "dictllm_enabled": True,
    }))
    ptb = tmp_path / "ptb.txt"
    ptb.write_text("test text")
    monkeypatch.setattr(evaluation, "load_tokenizer", lambda _: (
        lambda text, **kwargs: SimpleNamespace(input_ids=torch.ones((1, 8)))
    ))
    monkeypatch.setattr(evaluation, "load_checkpoint_model", lambda *args: object())
    monkeypatch.setattr(evaluation, "compute_perplexity", lambda *args, **kwargs: 9.0)
    evaluation.main([str(tmp_path), "--datasets", "ptb", "--ptb-data-file", str(ptb),
                     "--c4-data-file", str(tmp_path / "missing.json")])
    assert (tmp_path / "final_ppl_ptb.json").exists()
    assert not (tmp_path / "final_ppl_c4.json").exists()


def test_defaults_use_huggingface():
    args = evaluation.parse_args(["checkpoint"])
    assert args.c4_data_file is None
    assert args.ptb_data_file is None
    assert args.c4_revision == args.ptb_revision == "main"


def test_hf_c4_selects_only_validation_shard(tmp_path, monkeypatch):
    from datasets import Dataset

    calls = []
    raw = Dataset.from_dict({"text": ["one", "two", "three"]})
    monkeypatch.setattr(evaluation, "load_dataset", lambda *a, **kw: calls.append((a, kw)) or raw)
    args = evaluation.parse_args([
        "checkpoint", "--c4-documents", "2", "--c4-revision", "fixed-commit",
        "--dataset-cache-dir", str(tmp_path),
    ])
    text, info = evaluation.prepare_test_text("c4", args)
    assert text == "one\n\ntwo"
    assert info["records"] == 2
    assert info["dataset_file"] == evaluation.C4_VALIDATION_FILE
    assert info["dataset_revision"] == "fixed-commit"
    assert calls == [((evaluation.C4_DATASET_ID, "en"), {
        "data_files": {"validation": evaluation.C4_VALIDATION_FILE},
        "split": "validation", "revision": "fixed-commit", "cache_dir": str(tmp_path),
        "verification_mode": "no_checks",
    })]


def test_hf_ptb_matches_local_research_text(tmp_path, monkeypatch):
    from datasets import Dataset

    path = tmp_path / "ptb.txt"
    path.write_text(" first \n\n second \n", encoding="utf-8")
    calls = []
    raw = Dataset.from_dict({"sentence": ["first", "", "second"]})
    monkeypatch.setattr(evaluation, "load_dataset", lambda *a, **kw: calls.append((a, kw)) or raw)
    text, info = evaluation.prepare_test_text("ptb", evaluation.parse_args(["checkpoint"]))
    local_text, local_info = evaluation.prepare_test_text("ptb", evaluation.parse_args([
        "checkpoint", "--ptb-data-file", str(path),
    ]))
    assert text == local_text == "first\n\n\n\nsecond"
    assert info["text_sha256"] == local_info["text_sha256"]
    assert info["dataset_id"] == evaluation.PTB_DATASET_ID
    assert calls == [((evaluation.PTB_DATASET_ID, "penn_treebank"), {
        "split": "test", "revision": "main", "cache_dir": None,
        **({"trust_remote_code": False} if int(evaluation.datasets.__version__.split(".")[0]) < 4 else {}),
    })]


@pytest.mark.parametrize("message", [
    "Dataset scripts are no longer supported, but found ptb_text_only.py",
    "Please pass trust_remote_code=True",
])
def test_ptb_script_compatibility_fallback(tmp_path, monkeypatch, message):
    path = tmp_path / "download.txt"
    path.write_text(" first \n second \n", encoding="utf-8")
    calls = []

    def unsupported(*args, **kwargs):
        raise RuntimeError(message)

    monkeypatch.setattr(evaluation, "load_dataset", unsupported)
    manager = SimpleNamespace(download=lambda url: calls.append(url) or str(path))
    monkeypatch.setattr(evaluation, "DownloadManager", lambda **kwargs: manager)
    text, info = evaluation.prepare_test_text("ptb", evaluation.parse_args(["checkpoint"]))
    assert text == "first\n\nsecond"
    assert calls == [evaluation.PTB_TEST_URL]
    assert info["data_source"] == "huggingface_download_manager_upstream"
    assert len(info["data_sha256"]) == 64


def test_ptb_does_not_hide_unrelated_hf_errors(monkeypatch):
    def failed(*args, **kwargs):
        raise ValueError("corrupt dataset")

    monkeypatch.setattr(evaluation, "load_dataset", failed)
    with pytest.raises(ValueError, match="corrupt dataset"):
        evaluation.prepare_test_text("ptb", evaluation.parse_args(["checkpoint"]))


def test_ptb_pinned_revision_cannot_silently_fallback(monkeypatch):
    def unsupported(*args, **kwargs):
        raise RuntimeError("Dataset scripts are no longer supported")

    monkeypatch.setattr(evaluation, "load_dataset", unsupported)
    with pytest.raises(ValueError, match="pinned PTB revision"):
        evaluation.prepare_test_text("ptb", evaluation.parse_args([
            "checkpoint", "--ptb-revision", "fixed-commit",
        ]))


def test_remote_whitespace_ptb_keeps_raw_spaces(tmp_path, monkeypatch):
    path = tmp_path / "download.txt"
    path.write_text(" first \n\n second \n", encoding="utf-8")

    def no_script(*args, **kwargs):
        pytest.fail("Whitespace protocol must not use the stripping HF script")

    monkeypatch.setattr(evaluation, "load_dataset", no_script)
    monkeypatch.setattr(evaluation, "DownloadManager", lambda **kwargs: (
        SimpleNamespace(download=lambda url: str(path))
    ))
    text, info = evaluation.prepare_test_text("ptb", evaluation.parse_args([
        "checkpoint", "--ptb-protocol", "whitespace_preserved",
    ]))
    assert text == " first \n\n\n\n second "
    assert info["strip_sentences"] is False


def test_explicit_missing_file_does_not_download(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "llama", "dictllm_enabled": True,
    }))

    def unexpected(*args, **kwargs):
        pytest.fail("Missing explicit files must fail before downloading or loading a model")

    monkeypatch.setattr(evaluation, "load_dataset", unexpected)
    monkeypatch.setattr(evaluation, "load_checkpoint_model", unexpected)
    with pytest.raises(FileNotFoundError, match="Evaluation data not found"):
        evaluation.main([
            str(tmp_path), "--datasets", "ptb",
            "--ptb-data-file", str(tmp_path / "missing.txt"),
        ])


def test_cache_requires_packed_runtime():
    with pytest.raises(SystemExit):
        evaluation.parse_args(["checkpoint", "--cache-dequantized"])


def test_packed_checkpoint_requires_explicit_runtime(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"offline_nvfp4": True}))
    with pytest.raises(ValueError, match="requires --packed-runtime"):
        evaluation.main([str(tmp_path)])


def test_packed_research_ptb_uses_adapter_and_custom_output(tmp_path, monkeypatch):
    from dictllm.nvfp4 import checkpoint_adapter, evaluation as packed_evaluation

    (tmp_path / "config.json").write_text(json.dumps({"offline_nvfp4": True}))
    ptb = tmp_path / "ptb.txt"
    ptb.write_text(" text with spaces \n", encoding="utf-8")
    calls = []
    config = SimpleNamespace(to_dict=lambda: {"model_type": "llama"})
    monkeypatch.setattr(packed_evaluation, "read_evaluation_config", lambda _: config)
    model = torch.nn.Linear(2, 2)
    monkeypatch.setattr(packed_evaluation, "load_evaluation_model",
                        lambda *args: calls.append(args) or model)

    def tokenize(text, **kwargs):
        assert text == "text with spaces"
        assert kwargs == {"return_tensors": "pt"}
        return SimpleNamespace(input_ids=torch.ones((1, 8), dtype=torch.long))

    monkeypatch.setattr(checkpoint_adapter, "load_tokenizer", lambda *args: tokenize)
    monkeypatch.setattr(evaluation, "compute_perplexity", lambda *args, **kwargs: 8.5)
    output = tmp_path / "results"
    evaluation.main([
        str(tmp_path), "--datasets", "ptb", "--ptb-data-file", str(ptb),
        "--device", "cpu", "--output-dir", str(output),
        "--packed-runtime", "dequantize", "--cache-dequantized",
    ])
    assert calls == [(tmp_path, "bfloat16", "cpu", True)]
    report = json.loads((output / "final_ppl_ptb.json").read_text())
    assert report["strip_sentences"] is True
    assert report["runtime"] == "dequantize"
    assert report["activation_quantization"] is False
    assert report["cache_dequantized"] is True
    assert not (tmp_path / "final_ppl_ptb.json").exists()
