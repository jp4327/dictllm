"""Calibration and perplexity datasets."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Subset
from transformers import default_data_collator


def _tokenize(examples, tokenizer):
    return tokenizer(examples["text"])


def _group_text(examples, context_length: int):
    concatenated = {
        key: sum(examples[key], [])
        for key in examples
        if key in {"input_ids", "attention_mask"}
    }
    total = (len(concatenated["input_ids"]) // context_length) * context_length
    return {
        key: [values[start:start + context_length] for start in range(0, total, context_length)]
        for key, values in concatenated.items()
    }


def load_wikitext_raw(split: str, cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    return load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split=split,
        cache_dir=str(cache_dir),
    )


def prepare_wikitext_calibration(tokenizer, calibration: dict, cache_dir: Path):
    train_raw = load_wikitext_raw(calibration.get("split", "train"), cache_dir)
    workers = int(calibration.get("num_workers", 4))
    tokenized = train_raw.map(
        partial(_tokenize, tokenizer=tokenizer),
        batched=True,
        num_proc=workers,
        remove_columns=train_raw.column_names,
        desc="Tokenize WikiText-2 calibration split",
    )
    grouped = tokenized.map(
        partial(_group_text, context_length=int(calibration["context_length"])),
        batched=True,
        batch_size=1024,
        num_proc=workers,
        desc="Build calibration sequences",
    )
    sample_count = int(calibration["samples"])
    if sample_count > len(grouped):
        raise ValueError(
            f"Requested {sample_count} calibration samples, but only {len(grouped)} exist."
        )
    generator = torch.Generator().manual_seed(int(calibration["seed"]))
    if calibration.get("selection", "random") == "first":
        indices = list(range(sample_count))
    else:
        indices = torch.randperm(len(grouped), generator=generator)[:sample_count].tolist()
    return DataLoader(
        Subset(grouped, indices),
        batch_size=int(calibration["batch_size"]),
        shuffle=False,
        collate_fn=default_data_collator,
        pin_memory=True,
        num_workers=workers,
    )


def prepare_wikitext_test(tokenizer, cache_dir: Path):
    test_raw = load_wikitext_raw("test", cache_dir)
    return tokenizer("\n\n".join(test_raw["text"]), return_tensors="pt")
