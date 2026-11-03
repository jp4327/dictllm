"""Model loading helpers with Transformers-version compatibility."""

from __future__ import annotations

from pathlib import Path

import torch
import transformers
from transformers import AutoTokenizer


DTYPES = {
    "float64": torch.float64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def dtype_from_name(name: str) -> torch.dtype:
    try:
        return DTYPES[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {name}") from exc


def model_class(architecture: str):
    if architecture == "llama":
        from .modeling_llama import DictLlamaForCausalLM

        return DictLlamaForCausalLM
    raise ValueError(f"Unsupported architecture: {architecture}")


def _dtype_kwargs(dtype: torch.dtype) -> dict:
    major = int(transformers.__version__.split(".", 1)[0])
    return {"dtype" if major >= 5 else "torch_dtype": dtype}


def load_source_model(model_config: dict, device_map=None):
    cls = model_class(model_config["architecture"])
    kwargs = {
        **_dtype_kwargs(dtype_from_name(model_config["dtype"])),
        "low_cpu_mem_usage": True,
        "revision": model_config.get("revision"),
        "cache_dir": model_config.get("cache_dir"),
    }
    if device_map is not None:
        kwargs["device_map"] = device_map
    attention = model_config.get("attention_implementation")
    if attention:
        kwargs["attn_implementation"] = attention
    return cls.from_pretrained(model_config["name_or_path"], **kwargs)


def load_checkpoint_model(checkpoint: str | Path, architecture: str, dtype_name: str):
    cls = model_class(architecture)
    return cls.from_pretrained(
        str(checkpoint),
        low_cpu_mem_usage=True,
        **_dtype_kwargs(dtype_from_name(dtype_name)),
    )


def load_tokenizer(
    source: str | Path,
    revision: str | None = None,
    cache_dir: str | None = None,
):
    tokenizer = AutoTokenizer.from_pretrained(
        str(source), revision=revision, cache_dir=cache_dir
    )
    # Older Transformers can return False when fast LLaMA conversion fails.
    if not callable(tokenizer):
        tokenizer = AutoTokenizer.from_pretrained(
            str(source), revision=revision, cache_dir=cache_dir, use_fast=False
        )
    if not callable(tokenizer):
        raise TypeError(f"AutoTokenizer returned an invalid object for {source}.")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither a padding token nor an EOS token.")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
