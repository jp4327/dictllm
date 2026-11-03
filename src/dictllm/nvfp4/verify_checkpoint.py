"""Read-only source-mask and shape verification for floating-point exports."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path

import torch


class TensorStore:
    def __init__(self, directory, stack):
        directory = Path(directory)
        self.entries = {}
        self.readers = {}
        safe_index = directory / "model.safetensors.index.json"
        bin_index = directory / "pytorch_model.bin.index.json"
        index = safe_index if safe_index.exists() else bin_index
        weight_map = json.loads(index.read_text())["weight_map"] if index.exists() else None
        if weight_map:
            files = sorted(set(weight_map.values()))
        elif (directory / "model.safetensors").exists():
            files = ["model.safetensors"]
        else:
            files = ["pytorch_model.bin"]
        actual = {}
        for filename in files:
            path = (directory / filename).resolve()
            if path.parent != directory.resolve():
                raise ValueError("Shard filenames must stay within the checkpoint directory.")
            if filename.endswith(".safetensors"):
                from safetensors import safe_open
                reader = stack.enter_context(safe_open(path, framework="pt", device="cpu"))
                entries = {name: reader.get_slice(name) for name in reader.keys()}
                self.readers.update({name: reader for name in entries})
            else:
                entries = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            for name, tensor in entries.items():
                if name in self.entries:
                    raise ValueError(f"Duplicate tensor {name}")
                self.entries[name] = tensor
                actual[name] = filename
        if weight_map is not None and actual != weight_map:
            raise ValueError("Shard index does not match serialized tensors.")

    def shape(self, name):
        tensor = self.entries[name]
        return list(tensor.shape) if isinstance(tensor, torch.Tensor) else tensor.get_shape()

    def rows(self, name, start, end):
        return self.entries[name][start:end]

    def tensor(self, name):
        entry = self.entries[name]
        return entry if isinstance(entry, torch.Tensor) else self.readers[name].get_tensor(name)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def verify_checkpoint(source, checkpoint, records, chunk_rows=128, *, quantization_summary=None, progress=False):
    if chunk_rows < 1 or not records:
        raise ValueError("Verification needs positive chunk_rows and nonempty tensor records.")
    errors, results = [], []
    names = [record["name"] for record in records]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate verification records.")
    if quantization_summary is not None:
        from .fp4_reference import close_scale, prepare_reference, scan_levels, validate_settings
        validate_settings(quantization_summary)
    with ExitStack() as stack:
        parent = TensorStore(source, stack)
        saved = TensorStore(checkpoint, stack)
        for number, record in enumerate(records, 1):
            name = record["name"]
            shape = record["shape"]
            aliases = record.get("aliases", [name])
            if name not in aliases:
                errors.append(f"{name}: canonical tensor absent from alias list")
            if any(key not in parent.entries or key not in saved.entries or
                   parent.shape(key) != shape or saved.shape(key) != shape for key in aliases):
                errors.append(f"{name}: missing or shape-mismatched factor/alias")
                continue
            if progress:
                print(f"[{number}/{len(records)}] verify={name}", flush=True)
            reference, grid_check = None, None
            if quantization_summary is not None:
                try:
                    reference = prepare_reference(parent, record, quantization_summary, chunk_rows)
                except ValueError as exc:
                    errors.append(str(exc))
                grid_check = {"groups_checked": 0, "all_zero_groups": 0,
                              "cpu_cuda_scale_boundary_groups": 0, "cuda_reciprocal_scale_groups": 0,
                              "underflow_clamped_groups": 0, "off_grid_elements": 0,
                              "max_normalized_grid_residual": 0., "effective_scale_min": None,
                              "effective_scale_max": None, "encoded_e4m3_scale_min": None,
                              "encoded_e4m3_scale_max": None, "magnitude_code_histogram": [0] * 8,
                              "scales_reconstructed_from_source": True,
                              "per_tensor_global_scale": reference["global_scale"] if reference else None,
                              "expected_per_tensor_global_scale": reference["expected_global_scale"] if reference else None}
            counts = {"zeros_before": 0, "zeros_after": 0, "revived_zeros": 0,
                      "additional_zeros": 0, "nonfinite": 0, "source_nonfinite": 0,
                      "alias_mismatches": 0}
            numel = 0
            for start in range(0, shape[0], chunk_rows):
                before = parent.rows(name, start, start + chunk_rows)
                after = saved.rows(name, start, start + chunk_rows)
                if not before.is_floating_point() or not after.is_floating_point():
                    errors.append(f"{name}: expected floating-point simulation weights")
                    break
                numel += before.numel()
                if reference is not None:
                    if after.dtype != reference["source_dtype"]:
                        errors.append(f"{name}: saved dtype differs from quantization dtype")
                        reference = None
                    else:
                        checked = scan_levels(before, after, reference)
                        for field in ["groups_checked", "all_zero_groups", "underflow_clamped_groups", "off_grid_elements",
                                      "cpu_cuda_scale_boundary_groups", "cuda_reciprocal_scale_groups"]:
                            grid_check[field] += checked[field]
                        grid_check["max_normalized_grid_residual"] = max(
                            grid_check["max_normalized_grid_residual"], checked["max_normalized_grid_residual"])
                        for field in ["effective_scale_min", "encoded_e4m3_scale_min"]:
                            grid_check[field] = checked[field] if grid_check[field] is None else min(grid_check[field], checked[field])
                        for field in ["effective_scale_max", "encoded_e4m3_scale_max"]:
                            grid_check[field] = checked[field] if grid_check[field] is None else max(grid_check[field], checked[field])
                        grid_check["magnitude_code_histogram"] = [a + b for a, b in zip(
                            grid_check["magnitude_code_histogram"], checked["magnitude_code_histogram"])]
                counts["zeros_before"] += int((before == 0).sum())
                counts["zeros_after"] += int((after == 0).sum())
                counts["revived_zeros"] += int(((before == 0) & (after != 0)).sum())
                counts["additional_zeros"] += int(((before != 0) & (after == 0)).sum())
                counts["nonfinite"] += int((~torch.isfinite(after)).sum())
                counts["source_nonfinite"] += int((~torch.isfinite(before)).sum())
                for alias in aliases:
                    counts["alias_mismatches"] += int(not torch.equal(saved.rows(alias, start, start + chunk_rows), after))
                    counts["alias_mismatches"] += int(not torch.equal(parent.rows(alias, start, start + chunk_rows), before))
            if counts["revived_zeros"] or counts["nonfinite"] or counts["source_nonfinite"] or counts["alias_mismatches"]:
                errors.append(f"{name}: zero, finite-value, or alias check failed")
            if numel == 0:
                errors.append(f"{name}: no floating-point elements verified")
            if grid_check is not None:
                if grid_check["off_grid_elements"]:
                    errors.append(f"{name}: {grid_check['off_grid_elements']} elements outside the reconstructed FP4 grid")
                if reference is not None:
                    for field in ["effective_scale_min", "effective_scale_max"]:
                        if not close_scale(float(record.get(field, float("nan"))), grid_check[field]):
                            errors.append(f"{name}: reconstructed {field} differs from summary")
            results.append({"name": name, "tensor_type": record["tensor_type"], "shape": shape,
                            "num_parameters": numel, **counts,
                            "sparsity_before": counts["zeros_before"] / numel if numel else None,
                            "sparsity_after": counts["zeros_after"] / numel if numel else None,
                            **({"nvfp4_grid": grid_check} if grid_check is not None else {})})
    targets = {}
    for kind in {record["tensor_type"] for record in results}:
        selected = [record for record in results if record["tensor_type"] == kind]
        totals = {field: sum(record[field] for record in selected) for field in
                  ["num_parameters", "zeros_before", "zeros_after", "revived_zeros", "additional_zeros"]}
        totals["sparsity_before"] = totals["zeros_before"] / totals["num_parameters"] if totals["num_parameters"] else None
        totals["sparsity_after"] = totals["zeros_after"] / totals["num_parameters"] if totals["num_parameters"] else None
        targets[kind] = totals
    return {"passed": not errors, "source_checkpoint": str(Path(source).resolve()),
            "checkpoint": str(Path(checkpoint).resolve()), "scope": "selected factors and their aliases",
            "packed_storage": False, "errors": errors, "per_tensor": results,
            "total_revived_zeros": sum(item["revived_zeros"] for item in results),
            "totals_by_target": targets, "nvfp4_grid_checked": quantization_summary is not None,
            "total_off_grid_elements": sum(item.get("nvfp4_grid", {}).get("off_grid_elements", 0) for item in results),
            "scale_verification_basis": "source-reconstructed, not serialized scale arrays" if quantization_summary else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, help="New report path; never overwrites an existing file.")
    parser.add_argument("--chunk-rows", type=int, default=128)
    parser.add_argument("--check-nvfp4", action="store_true",
                        help="Also check E2M1 levels, per-16 E4M3 scales, and tensor-global scales from source.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    summary = json.loads((args.checkpoint / "zero_preserving_summary.json").read_text())
    if summary.get("transform_class") != "identity" or summary.get("packed_storage") is not False:
        raise ValueError("Expected a rotation-free floating-point verification export.")
    source = args.source_checkpoint or summary["source_checkpoint"]
    report = verify_checkpoint(source, args.checkpoint, summary["per_tensor"], args.chunk_rows,
                               quantization_summary=summary if args.check_nvfp4 else None,
                               progress=not args.quiet)
    if args.output:
        write_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "per_tensor"}, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
