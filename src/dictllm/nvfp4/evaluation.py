"""Evaluation-only adapters for verified identity-packed checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

from .checkpoint_adapter import base_config, restore_base_config
from .packed_nvfp4 import MANIFEST, SCHEMA_VERSION, load_packed_model


def read_evaluation_config(checkpoint):
    """Restore the base config without passing packed weights to AutoModel."""
    checkpoint = Path(checkpoint).expanduser().resolve()
    data = json.loads((checkpoint / "config.json").read_text())
    manifest = json.loads((checkpoint / MANIFEST).read_text())
    if (manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("status") != "verified"
            or manifest.get("transform_class") != "identity"
            or manifest.get("export_requantizes") is not False):
        raise ValueError("Expected a verified identity-packed NVFP4 LLaMA checkpoint.")
    return base_config(restore_base_config(data))


def load_evaluation_model(checkpoint, dtype, device, cache_dequantized=False):
    import torch

    read_evaluation_config(checkpoint)
    return load_packed_model(
        checkpoint, runtime="dequantize", device=device,
        dtype=None if dtype == "auto" else getattr(torch, dtype),
        cache_dequantized=cache_dequantized,
    )


def evaluation_metadata(cache_dequantized):
    return {
        "checkpoint_format": "qutlass_compatible_packed_nvfp4",
        "runtime": "dequantize",
        "transform_class": "identity",
        "activation_quantization": False,
        "activation_format": "floating_point",
        "cache_dequantized": cache_dequantized,
        "memory_benchmark": False,
    }
