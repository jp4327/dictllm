"""Atomic stage artifacts and manifest utilities."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
from contextlib import contextmanager
from pathlib import Path

import torch
import transformers
import yaml

from . import __version__
from .constants import MANIFEST_NAME


def config_hash(config: dict) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    os.replace(temporary, path)


def read_manifest(checkpoint: str | Path) -> dict:
    path = Path(checkpoint) / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"DictLLM manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_is_complete(checkpoint: Path, stage: str, digest: str) -> bool:
    try:
        manifest = read_manifest(checkpoint)
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return (
        manifest.get("stage") == stage
        and manifest.get("status") == "complete"
        and manifest.get("config_sha256") == digest
    )


@contextmanager
def stage_workspace(output: Path, overwrite: bool = False):
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if not overwrite:
            raise FileExistsError(
                f"Stage output already exists: {output}. Use resume or --overwrite."
            )
        shutil.rmtree(output)
    temporary = output.parent / f".{output.name}.tmp-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        yield temporary
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def package_versions() -> dict[str, str]:
    return {
        "dictllm": __version__,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": str(torch.version.cuda),
    }


def model_file_bytes(checkpoint: Path) -> int:
    suffixes = {".safetensors", ".bin"}
    return sum(path.stat().st_size for path in checkpoint.iterdir() if path.suffix in suffixes)


def make_manifest(
    *,
    config: dict,
    stage: str,
    completed_stages: list[str],
    output: Path,
    parent: Path | None,
    summary: dict,
    evaluation: dict,
) -> dict:
    return {
        "schema_version": 1,
        "status": "complete",
        "stage": stage,
        "completed_stages": completed_stages,
        "checkpoint_dir": str(output.resolve()),
        "parent_checkpoint": str(parent.resolve()) if parent is not None else None,
        "source_model": config["model"]["name_or_path"],
        "architecture": config["model"]["architecture"],
        "config_sha256": config_hash(config),
        "versions": package_versions(),
        "calibration": config["calibration"],
        "factorization": config["factorization"],
        "pruning": config["pruning"] if stage in {"prune", "refit"} else None,
        "refit": config["refit"] if stage == "refit" else None,
        "serialization": config["serialization"],
        "summary": summary,
        "evaluation": evaluation,
    }

