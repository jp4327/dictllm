import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_lm_harness.sh"


@pytest.fixture
def fake_interpreter(tmp_path):
    interpreter = tmp_path / "python"
    output = tmp_path / "call.json"
    interpreter.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['TEST_CALL'], 'w') as handle:\n"
        "    json.dump({'args': sys.argv[1:], 'cwd': os.getcwd(),\n"
        "               'pythonpath': os.environ['PYTHONPATH'],\n"
        "               'gpu': os.environ.get('CUDA_VISIBLE_DEVICES')}, handle)\n"
    )
    interpreter.chmod(0o755)
    env = {**os.environ, "PYTHON_BIN": str(interpreter),
           "TEST_CALL": str(output), "CUDA_VISIBLE_DEVICES": "1"}
    return env, output


def test_shell_syntax_and_help():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    result = subprocess.run(["bash", str(SCRIPT), "--help"],
                            capture_output=True, text=True, check=True)
    assert "CHECKPOINT_OR_HF_ID" in result.stdout


def test_missing_model_fails_before_python(fake_interpreter):
    env, output = fake_interpreter
    result = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True)
    assert result.returncode == 2
    assert not output.exists()


@pytest.mark.parametrize("packed", [False, True])
def test_launcher_defaults_path_handling_and_extra_options(tmp_path, fake_interpreter, packed):
    env, output = fake_interpreter
    checkpoint = tmp_path / "model with spaces"
    checkpoint.mkdir()
    if packed:
        (checkpoint / "packed_nvfp4_manifest.json").write_text("{}")
    subprocess.run([
        "bash", str(SCRIPT), checkpoint.name,
        "--batch-size", "2", "--output-path", "custom output.json", "--overwrite",
    ], env=env, cwd=tmp_path, check=True)
    call = json.loads(output.read_text())
    args = call["args"]
    assert args[:3] == ["-m", "dictllm.eval_lm_harness", checkpoint.name]
    assert args[args.index("--add-bos-token") + 1] == "false"
    assert args[args.index("--summary-metric") + 1] == "acc"
    assert args[args.index("--num-fewshot") + 1] == "0"
    assert ("--packed-runtime" in args) is packed
    if packed:
        assert args[args.index("--packed-runtime") + 1] == "dequantize"
    assert args[-5:] == ["--batch-size", "2", "--output-path", "custom output.json", "--overwrite"]
    assert call["cwd"] == str(tmp_path)
    assert call["pythonpath"].split(os.pathsep)[0] == str(ROOT / "src")
    assert call["gpu"] == "1"


def test_dense_model_id_passed_unchanged(tmp_path, fake_interpreter):
    env, output = fake_interpreter
    subprocess.run(["bash", str(SCRIPT), "org/model", "--output-path", "dense.json"],
                   env=env, cwd=tmp_path, check=True)
    args = json.loads(output.read_text())["args"]
    assert args[2] == "org/model"
    assert "--packed-runtime" not in args
