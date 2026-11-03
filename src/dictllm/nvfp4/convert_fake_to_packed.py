#!/usr/bin/env python
"""Read-only fake-NVFP4 conversion to EXACT QuTLASS-compatible packed bytes."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import shutil

import torch
from safetensors.torch import save_file
from .checkpoint_adapter import PACKED_MODEL_TYPES, architecture, restore_base_config

from .fp4_reference import DTYPES, prepare_reference, scan_levels, validate_settings
from .packed_nvfp4 import MANIFEST, SCHEMA_VERSION, decode, encode_verified, padded_shape
from .verify_checkpoint import TensorStore, write_json


PAYLOAD_FIELDS = ("qweight", "scale_bytes", "global_scale_bytes")


def resolve_checkpoint(value):
    path = Path(value).expanduser().resolve()
    if not (path / "config.json").is_file() and (path / "wikitext/config.json").is_file():
        path = path / "wikitext"
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"No checkpoint config in {path}.")
    return path


class ShardWriter:
    def __init__(self, directory, max_bytes):
        self.directory, self.max_bytes = directory, max_bytes
        self.pending, self.weight_map = {}, {}
        self.pending_bytes = self.total_bytes = 0
        self.shards = 0

    def add(self, name, tensor):
        if name in self.weight_map or name in self.pending:
            raise ValueError(f"Duplicate output tensor: {name}")
        size = tensor.numel() * tensor.element_size()
        if self.pending and self.pending_bytes + size > self.max_bytes:
            self.flush()
        # Clone because legacy aliases can share a storage and safetensors rejects
        # shared pointers. Tensors above the shard limit get their own shard.
        self.pending[name] = tensor.detach().cpu().contiguous().clone()
        self.pending_bytes += size
        self.total_bytes += size
        if self.pending_bytes >= self.max_bytes:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        self.shards += 1
        filename = f"model-{self.shards:05d}.safetensors"
        save_file(self.pending, self.directory / filename, metadata={"format": "pt"})
        self.weight_map.update({name: filename for name in self.pending})
        self.pending = {}
        self.pending_bytes = 0

    def finish(self):
        self.flush()
        write_json(self.directory / "model.safetensors.index.json", {
            "metadata": {"total_size": self.total_bytes}, "weight_map": self.weight_map,
        })


def validate_summary(summary):
    validate_settings(summary)
    if summary.get("partial_quantization"):
        raise ValueError("Partial debug quantization checkpoints are not supported.")
    records = summary.get("per_tensor", [])
    if not records:
        raise ValueError("Quantization summary has no tensor records.")
    aliases = []
    for record in records:
        if (record.get("tensor_type") not in {"basis", "coefficient"}
                or record.get("dtype") not in DTYPES or len(record["shape"]) != 2):
            raise ValueError("Invalid factor record.")
        keys = record.get("aliases", [record["name"]])
        if record["name"] not in keys or any(not name.endswith(".weight") for name in keys):
            raise ValueError("Invalid tensor aliases.")
        aliases.extend(keys)
    if len(aliases) != len(set(aliases)):
        raise ValueError("Quantization summary has overlapping aliases.")
    return records, set(aliases)


def convert_checkpoint(checkpoint, output, *, source_checkpoint=None, chunk_rows=64, shard_mb=512):
    checkpoint = resolve_checkpoint(checkpoint)
    output = Path(output).expanduser().resolve()
    if min(chunk_rows, shard_mb) < 1:
        raise ValueError("Chunk rows and shard MB must be positive.")
    summary = json.loads((checkpoint / "zero_preserving_summary.json").read_text())
    records, aliases = validate_summary(summary)
    source = resolve_checkpoint(source_checkpoint or summary["source_checkpoint"])
    config = json.loads((checkpoint / "config.json").read_text())
    architecture(config)
    for parent in (source, checkpoint):
        if output == parent or output in parent.parents or parent in output.parents:
            raise ValueError("Output must be separate from and not nested with either input.")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}.")
    manifest = {"schema_version": SCHEMA_VERSION, "status": "pending_verification",
                "format": "nvfp4", "representation": "qutlass_compatible_packed_nvfp4",
                "packing_backend": "exact_offline_encoder", "qutlass_packing_api_used": False,
                "export_requantizes": False, "transform_class": "identity", "group_size": 16,
                "scale_precision": "e4m3", "preserves_sparse_mask": True,
                "supported_runtimes": ["dequantize", "qutlass"],
                "default_runtime": "dequantize", "fake_checkpoint": str(checkpoint),
                "source_checkpoint": str(source), "per_tensor": []}
    with ExitStack() as stack:
        original = TensorStore(source, stack)
        fake = TensorStore(checkpoint, stack)
        if not aliases <= set(fake.entries) or not aliases <= set(original.entries):
            raise ValueError("Missing source/fake factor aliases.")
        output.mkdir(parents=True, exist_ok=False)
        writer = ShardWriter(output, shard_mb * 1024 * 1024)
        for number, record in enumerate(records, 1):
            name = record["name"]
            shape = record["shape"]
            record_aliases = record.get("aliases", [name])
            if any(fake.shape(key) != shape or original.shape(key) != shape for key in record_aliases):
                raise ValueError(f"Shape mismatch: {name}")
            reference = prepare_reference(original, record, summary, chunk_rows)
            storage_rows, storage_columns = padded_shape(shape)
            payload = {
                "qweight": torch.zeros(storage_rows, storage_columns // 2, dtype=torch.uint8),
                "scale_bytes": torch.zeros(storage_rows, storage_columns // 16, dtype=torch.uint8),
                "global_scale_bytes": torch.tensor([reference["global_scale"]], dtype=torch.float32).view(torch.uint8),
            }
            source_zeros = saved_zeros = boundary_groups = 0
            for start in range(0, shape[0], chunk_rows):
                before = original.rows(name, start, start + chunk_rows)
                after = fake.rows(name, start, start + chunk_rows)
                if after.dtype != reference["source_dtype"] or not torch.isfinite(after).all():
                    raise ValueError(f"Fake weight dtype/non-finite mismatch: {name}")
                for alias in record_aliases:
                    if (not torch.equal(after, fake.rows(alias, start, start + chunk_rows))
                            or not torch.equal(before, original.rows(alias, start, start + chunk_rows))):
                        raise ValueError(f"Shared alias mismatch: {alias}")
                if ((before == 0) & (after != 0)).any():
                    raise ValueError(f"Revived source zeros: {name}")
                checked = scan_levels(before, after, reference, return_details=True)
                encoded = encode_verified(after, checked["details"], reference["global_scale"])
                stop = start + after.shape[0]
                payload["qweight"][start:stop, :encoded["qweight"].shape[1]] = encoded["qweight"]
                payload["scale_bytes"][start:stop, :encoded["scale_bytes"].shape[1]] = encoded["scale_bytes"]
                source_zeros += int((before == 0).sum())
                saved_zeros += int((after == 0).sum())
                boundary_groups += checked["cuda_reciprocal_scale_groups"]
            prefix = name.removesuffix(".weight")
            keys = {field: f"{prefix}.{field}" for field in PAYLOAD_FIELDS}
            for field, key in keys.items():
                writer.add(key, payload[field])
            packed_bytes = sum(value.numel() * value.element_size() for value in payload.values())
            spec = {"name": name, "aliases": record_aliases, "tensor_type": record["tensor_type"],
                    "shape": shape, "storage_shape": [storage_rows, storage_columns],
                    "dtype": record["dtype"], "keys": keys, "global_scale": reference["global_scale"],
                    "source_zeros": source_zeros, "saved_zeros": saved_zeros,
                    "cuda_reciprocal_scale_groups": boundary_groups, "packed_bytes": packed_bytes,
                    "source_storage_bytes": shape[0] * shape[1] * torch.empty((), dtype=after.dtype).element_size(),
                    "fp16_reference_bytes": shape[0] * shape[1] * 2}
            manifest["per_tensor"].append(spec)
            print(f"[{number}/{len(records)}] exact packed={name} shape={shape} zeros={saved_zeros}", flush=True)
        # Unselected factors, embeddings, norms, LM head, etc. pass through exactly.
        for name in sorted(set(fake.entries) - aliases):
            writer.add(name, fake.tensor(name))
        writer.finish()
        manifest["state_keys"] = sorted(writer.weight_map)
        manifest["tensor_storage_bytes"] = writer.total_bytes
    manifest["packed_factor_bytes"] = sum(item["packed_bytes"] for item in manifest["per_tensor"])
    manifest["factor_gain_vs_fp16"] = sum(item["fp16_reference_bytes"] for item in manifest["per_tensor"]) / manifest["packed_factor_bytes"]
    manifest["factor_gain_vs_saved_dtype"] = sum(item["source_storage_bytes"] for item in manifest["per_tensor"]) / manifest["packed_factor_bytes"]
    config["offline_nvfp4"] = {"schema_version": SCHEMA_VERSION, "manifest": MANIFEST,
                              "transform_class": "identity", "requires_custom_loader": True,
                              "base_architectures": config["architectures"], "base_model_type": config["model_type"]}
    # Force generic AutoConfig/AutoModel to fail instead of ignoring the packed
    # keys and randomly initializing all factor weights as ordinary linears.
    config["architectures"] = [
        "IdentityPackedDictLlamaForCausalLM"
    ]
    config["model_type"] = PACKED_MODEL_TYPES[config["model_type"]]
    write_json(output / "config.json", config)
    for filename in ["generation_config.json", "tokenizer.model", "tokenizer.json", "tokenizer_config.json",
                     "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt"]:
        if (checkpoint / filename).is_file():
            shutil.copyfile(checkpoint / filename, output / filename)
    write_json(output / "fake_quantization_summary.json", summary)
    write_json(output / MANIFEST, manifest)
    print("Packed shards saved; verifying exact weights, scales, aliases, and padding from disk.", flush=True)
    report = verify_conversion(output, allow_pending=True, chunk_rows=chunk_rows, progress=True)
    write_json(output / "conversion_verification.json", report)
    if not report["passed"]:
        raise RuntimeError(f"Conversion verification failed: {report['errors']}")
    manifest["status"] = "verified"
    temporary = output / (MANIFEST + ".tmp")
    write_json(temporary, manifest)
    temporary.replace(output / MANIFEST)
    return manifest, report


def verify_conversion(checkpoint, *, fake_checkpoint=None, source_checkpoint=None,
                      chunk_rows=64, allow_pending=False, progress=False):
    checkpoint = resolve_checkpoint(checkpoint)
    manifest = json.loads((checkpoint / MANIFEST).read_text())
    if (manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("transform_class") != "identity"
            or (not allow_pending and manifest.get("status") != "verified")
            or manifest.get("export_requantizes") is not False):
        raise ValueError("Unsupported or incomplete packed manifest.")
    config = json.loads((checkpoint / "config.json").read_text())
    metadata = config.get("offline_nvfp4", {})
    if metadata.get("requires_custom_loader") is not True or metadata.get("transform_class") != "identity":
        raise ValueError("Missing packed custom-loader configuration.")
    restore_base_config(config)
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive.")
    fake_path = resolve_checkpoint(fake_checkpoint or manifest["fake_checkpoint"])
    source_path = resolve_checkpoint(source_checkpoint or manifest["source_checkpoint"])
    errors, rows = [], []
    with ExitStack() as stack:
        packed = TensorStore(checkpoint, stack)
        fake = TensorStore(fake_path, stack)
        original = TensorStore(source_path, stack)
        summary = json.loads((fake_path / "zero_preserving_summary.json").read_text())
        original_records, aliases = validate_summary(summary)
        expected_records = {item["name"]: item for item in original_records}
        records = manifest["per_tensor"]
        if len(records) != len(expected_records) or {item["name"] for item in records} != set(expected_records):
            raise ValueError("Packed records do not match all fake-quantized factors.")
        expected_keys = set(fake.entries) - aliases
        total_packed_bytes = 0
        for number, spec in enumerate(records, 1):
            name, shape = spec["name"], spec["shape"]
            record = expected_records[name]
            if spec["aliases"] != record.get("aliases", [name]) or shape != record["shape"] or spec["dtype"] != record["dtype"]:
                raise ValueError(f"Manifest/source record mismatch: {name}")
            storage_rows, storage_columns = padded_shape(shape)
            if spec["storage_shape"] != [storage_rows, storage_columns]:
                raise ValueError(f"Invalid storage padding: {name}")
            prefix = name.removesuffix(".weight")
            keys = {field: f"{prefix}.{field}" for field in PAYLOAD_FIELDS}
            if spec["keys"] != keys:
                raise ValueError(f"Invalid packed tensor keys: {name}")
            expected_keys.update(keys.values())
            payload = {field: packed.tensor(key) for field, key in keys.items()}
            if (any(value.dtype != torch.uint8 for value in payload.values())
                    or tuple(payload["qweight"].shape) != (storage_rows, storage_columns // 2)
                    or tuple(payload["scale_bytes"].shape) != (storage_rows, storage_columns // 16)
                    or tuple(payload["global_scale_bytes"].shape) != (4,)):
                raise ValueError(f"Invalid packed layout/dtype: {name}")
            reference = prepare_reference(original, record, summary, chunk_rows)
            G = float(payload["global_scale_bytes"].view(torch.float32)[0])
            if G != float(torch.tensor(reference["global_scale"], dtype=torch.float32)):
                errors.append(f"{name}: global scale changed")
            scales = payload["scale_bytes"].view(torch.float8_e4m3fn).float()
            if not torch.isfinite(scales).all() or (scales < 0).any():
                errors.append(f"{name}: invalid E4M3 scales")
            mismatch = revived = zeros = source_zeros = 0
            for start in range(0, storage_rows, chunk_rows):
                stop = min(start + chunk_rows, storage_rows)
                restored = decode(payload["qweight"][start:stop], payload["scale_bytes"][start:stop],
                                  payload["global_scale_bytes"], dtype=DTYPES[spec["dtype"]])
                actual_rows = max(0, min(stop, shape[0]) - start)
                if (restored[actual_rows:] != 0).any() or (restored[:actual_rows, shape[1]:] != 0).any():
                    errors.append(f"{name}: nonzero padding")
                if not actual_rows:
                    continue
                real = restored[:actual_rows, :shape[1]]
                saved = fake.rows(name, start, start + actual_rows)
                before = original.rows(name, start, start + actual_rows)
                if saved.dtype != DTYPES[spec["dtype"]]:
                    errors.append(f"{name}: fake dtype differs from the manifest")
                mismatch += int((real != saved).sum())
                revived += int(((before == 0) & (real != 0)).sum())
                zeros += int((real == 0).sum())
                source_zeros += int((before == 0).sum())
                for alias in spec["aliases"]:
                    if (not torch.equal(saved, fake.rows(alias, start, start + actual_rows))
                            or not torch.equal(before, original.rows(alias, start, start + actual_rows))):
                        errors.append(f"{name}: source/fake alias mismatch: {alias}")
                checked = scan_levels(before, saved, reference, return_details=True)
                expected = encode_verified(saved, checked["details"], reference["global_scale"])
                if not torch.equal(payload["qweight"][start:start + actual_rows, :expected["qweight"].shape[1]], expected["qweight"]):
                    errors.append(f"{name}: FP4 code bytes changed")
                if not torch.equal(payload["scale_bytes"][start:start + actual_rows, :expected["scale_bytes"].shape[1]], expected["scale_bytes"]):
                    errors.append(f"{name}: block scale bytes changed")
            group_columns = (shape[1] + 15) // 16
            if (payload["qweight"][shape[0]:].any() or payload["scale_bytes"][shape[0]:].any()
                    or payload["qweight"][:shape[0], group_columns * 8:].any()
                    or payload["scale_bytes"][:shape[0], group_columns:].any()):
                errors.append(f"{name}: padding bytes must be zero")
            if mismatch or revived or zeros != spec["saved_zeros"] or source_zeros != spec["source_zeros"]:
                errors.append(f"{name}: weight/zero round-trip mismatch")
            size = sum(value.numel() for value in payload.values())
            if size != spec["packed_bytes"]:
                errors.append(f"{name}: byte accounting mismatch")
            total_packed_bytes += size
            rows.append({"name": name, "shape": shape, "packed_bytes": size,
                         "weight_mismatches": mismatch, "revived_zeros": revived,
                         "zeros": zeros, "sparsity": zeros / (shape[0] * shape[1])})
            if progress and (number % 16 == 0 or number == len(records)):
                print(f"[verify {number}/{len(records)}] exact_round_trip_checked", flush=True)
        if set(packed.entries) != expected_keys or set(manifest["state_keys"]) != expected_keys:
            errors.append("Unexpected/missing payload or unselected tensor keys")
        for name in sorted(set(fake.entries) - aliases):
            if name not in packed.entries or not torch.equal(packed.tensor(name), fake.tensor(name)):
                errors.append(f"Unselected tensor changed: {name}")
        if total_packed_bytes != manifest["packed_factor_bytes"]:
            errors.append("Aggregate packed-factor bytes differ from manifest")
        actual_bytes = sum(packed.tensor(key).numel() * packed.tensor(key).element_size() for key in packed.entries)
        if actual_bytes != manifest["tensor_storage_bytes"]:
            errors.append("Whole-checkpoint tensor byte accounting mismatch")
    fp16_bytes = sum(item["shape"][0] * item["shape"][1] * 2 for item in records)
    return {"passed": not errors, "errors": errors, "exact_weight_round_trip": not any(row["weight_mismatches"] for row in rows),
            "total_revived_zeros": sum(row["revived_zeros"] for row in rows),
            "transform_class": "identity", "qutlass_packing_api_used": False,
            "packed_factor_bytes": total_packed_bytes, "factor_gain_vs_fp16": fp16_bytes / total_packed_bytes,
            "per_tensor": rows, "gpu_runtime_tested": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Fake checkpoint for conversion; packed checkpoint for --verify-only")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-checkpoint", help="Override pre-quantization checkpoint location")
    parser.add_argument("--fake-checkpoint", help="Override fake checkpoint location in verification mode")
    parser.add_argument("--chunk-rows", type=int, default=64)
    parser.add_argument("--shard-mb", type=int, default=512)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        report = verify_conversion(args.checkpoint, source_checkpoint=args.source_checkpoint,
                                   fake_checkpoint=args.fake_checkpoint, chunk_rows=args.chunk_rows, progress=True)
        if args.output:
            write_json(args.output, report)
    else:
        if args.output is None or args.fake_checkpoint:
            parser.error("Conversion requires --output; --fake-checkpoint is for verification only.")
        manifest, report = convert_checkpoint(args.checkpoint, args.output, source_checkpoint=args.source_checkpoint,
                                             chunk_rows=args.chunk_rows, shard_mb=args.shard_mb)
        print(f"saved_packed_checkpoint={args.output.resolve()}")
        print(f"factor_gain_vs_saved_dtype={manifest['factor_gain_vs_saved_dtype']:.4f}x")
    print(f"passed={report['passed']} exact_round_trip={report['exact_weight_round_trip']} "
          f"revived_zeros={report['total_revived_zeros']} factor_gain_vs_fp16={report['factor_gain_vs_fp16']:.4f}x")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
