"""Checkpoint retention around the existing disk-reloaded compression stages."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid

from .constants import STAGE_ORDER
from .io import checkpoint_is_complete, config_hash, write_json
from .stages import factorize_stage, prune_stage, refit_stage


REPORT_FILES = (
    "config.json", "dictllm_manifest.json", "dictllm_config.yaml",
    "final_ppl_wikitext.json", "zero_preserving_summary.json",
    "zero_preservation_verification.json", "fake_quantization_summary.json",
    "conversion_verification.json", "packed_nvfp4_manifest.json",
)


def quantization_output(config, export_mode=None):
    return config.output_root / f"04_quantized_{export_mode or config.data['quantization']['export_mode']}"


def _reports(checkpoint, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for name in REPORT_FILES:
        source = checkpoint / name
        if source.is_file():
            shutil.copyfile(source, destination / name)


def _publish(source, destination, overwrite=False):
    """Move a verified owned workspace, preserving any prior output on failure."""
    if destination.is_symlink():
        raise ValueError(f"Refusing to replace a symlink: {destination}")
    backup = None
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {destination}; use --overwrite.")
        backup = destination.with_name(f".{destination.name}.previous-{uuid.uuid4().hex}")
        os.replace(destination, backup)
    try:
        os.replace(source, destination)
    except BaseException:
        if backup is not None:
            os.replace(backup, destination)
        raise
    if backup is not None:
        shutil.rmtree(backup)


def _rewrite_output_paths(checkpoint, before, after):
    def rewrite(value):
        if isinstance(value, str) and (value == str(before) or value.startswith(str(before) + "/")):
            return str(after) + value[len(str(before)):]
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if isinstance(value, dict):
            return {key: rewrite(item) for key, item in value.items()}
        return value

    for path in checkpoint.glob("*.json"):
        write_json(path, rewrite(json.loads(path.read_text())))


def quantize_stage(config, checkpoint=None, output=None, *, export_mode=None,
                   skip_evaluation=False, overwrite=False, keep_intermediates=False):
    from .nvfp4.run_pipeline import main as quantize

    source = Path(checkpoint or config.stage_output("refit")).resolve()
    target = Path(output or quantization_output(config, export_mode)).resolve()
    if target == source or source in target.parents or target in source.parents:
        raise ValueError("Source and output checkpoints must be separate, non-nested directories.")
    if target.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {target}; use --overwrite.")
    settings = config.data["quantization"]
    mode = export_mode or settings["export_mode"]
    target.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".quantize-", dir=target.parent))
    temporary = work / "checkpoint"
    options = [
        "--source-checkpoint", str(source), "--output-path", str(temporary),
        "--export-mode", mode, "--method", settings["method"],
        "--quant-targets", ",".join(settings["targets"]),
        "--w-group-size", str(settings["w_group_size"]),
        "--scale-precision", settings["scale_precision"],
        "--blocksize", str(settings["blocksize"]), "--rel-damp", str(settings["rel_damp"]),
        "--calibration-size", str(settings["calibration_size"]),
        "--calibration-batch-size", str(settings["calibration_batch_size"]),
        "--calibration-selection", settings["calibration_selection"],
        "--seed", str(config.data["pipeline"]["seed"]),
        "--context-length", str(config.data["evaluation"]["context_length"]),
        "--stride", str(config.data["evaluation"]["stride"]),
        "--dataset-cache-dir", str(config.resolve_path(config.data["calibration"]["dataset_cache_dir"])),
        "--device", settings["device"], "--no-activation-metrics",
    ]
    if not skip_evaluation and config.data["evaluation"]["enabled"]:
        options.append("--eval-perplexity")
    if settings["max_modules"] is not None:
        raise ValueError("Partial quantization is excluded from the minimal pipeline.")
    try:
        quantize(options)
        fake = temporary if mode == "fake" else work / "checkpoint_fake"
        _reports(fake, config.output_root / "reports/quantize_fake")
        if mode == "packed":
            _reports(temporary, config.output_root / "reports/quantize_packed")
            if keep_intermediates:
                retained_fake = target.with_name(target.name + "_fake")
                _rewrite_output_paths(fake, fake, retained_fake)
                _publish(fake, retained_fake, overwrite)
                manifest_path = temporary / "packed_nvfp4_manifest.json"
                manifest = json.loads(manifest_path.read_text())
                manifest["fake_checkpoint"] = str(retained_fake)
                write_json(manifest_path, manifest)
        _rewrite_output_paths(temporary, temporary, target)
        _publish(temporary, target, overwrite)
    except BaseException:
        print(f"Quantization failed; recovery workspace retained: {work}", flush=True)
        raise
    shutil.rmtree(work)
    print(f"saved_quantized_checkpoint={target}", flush=True)
    return target


def run_pipeline(config, *, quantize=False, policy=None, overwrite=False,
                 skip_evaluation=False, force_calibration=False, dry_run=False):
    policy = policy or config.data["checkpointing"]["policy"]
    if policy not in {"final", "all", "none"}:
        raise ValueError("Checkpoint policy must be final, all, or none.")
    stages = config.stages
    if stages != list(STAGE_ORDER[:len(stages)]):
        raise ValueError("run requires a pipeline prefix starting with factorize; use individual stage commands otherwise.")
    if quantize and stages[-1] != "refit":
        raise ValueError("run --quantize requires factorize, prune, and refit.")
    final = quantization_output(config) if quantize else config.stage_output(stages[-1])
    report_path = config.output_root / "pipeline_report.json"
    digest = config_hash(config.snapshot())
    if dry_run:
        print(f"checkpoint_policy={policy}\nstages={stages}\nquantize={quantize}")
        print(f"retained_checkpoint={final if policy != 'none' else 'none (reports only)'}")
        print("execution=fresh disk reloads; transient weights removed only after success")
        return None if policy == "none" else final
    if report_path.is_file() and config.data["pipeline"]["resume"] and not overwrite:
        saved = json.loads(report_path.read_text())
        if (saved.get("status") == "complete" and saved.get("config_sha256") == digest
                and saved.get("quantize") == quantize and saved.get("checkpoint_policy") == policy
                and (policy == "none" or (final / "config.json").is_file())):
            print(f"resume: completed pipeline at {config.output_root}")
            return None if policy == "none" else final
    for stage in stages:
        path = config.stage_output(stage)
        will_write = policy == "all" or (policy == "final" and not quantize and stage == stages[-1])
        resumable = (policy == "all" and config.data["pipeline"]["resume"]
                     and checkpoint_is_complete(path, stage, digest))
        if will_write and path.exists() and not overwrite and not resumable:
            raise FileExistsError(f"Output exists: {path}; use --overwrite.")
    if quantize and policy != "none" and final.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {final}; use --overwrite.")
    config.output_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".pipeline-", dir=config.output_root))
    report = {"status": "running", "config_sha256": digest, "checkpoint_policy": policy,
              "quantize": quantize, "execution": "fresh_checkpoint_reloads", "stages": []}
    previous = None
    try:
        for stage in stages:
            retained = policy == "all" or (policy == "final" and not quantize and stage == stages[-1])
            output = config.stage_output(stage) if retained else work / config.stage_output(stage).name
            if (policy == "all" and config.data["pipeline"]["resume"] and not overwrite
                    and checkpoint_is_complete(output, stage, digest)):
                previous = output
                print(f"resume: {stage} at {output}")
            else:
                kwargs = dict(output=output, overwrite=overwrite, skip_evaluation=skip_evaluation)
                if stage == "factorize":
                    previous = factorize_stage(config, force_calibration=force_calibration, **kwargs)
                elif stage == "prune":
                    previous = prune_stage(config, checkpoint=previous, **kwargs)
                else:
                    previous = refit_stage(config, checkpoint=previous, **kwargs)
            _reports(previous, config.output_root / "reports" / stage)
            report["stages"].append({"stage": stage, "checkpoint": str(previous), "retained": retained})
        if quantize:
            destination = final if policy != "none" else work / final.name
            previous = quantize_stage(config, checkpoint=previous, output=destination,
                                      skip_evaluation=skip_evaluation, overwrite=overwrite,
                                      keep_intermediates=policy == "all")
            report["stages"].append({"stage": "quantize", "checkpoint": str(previous),
                                      "retained": policy != "none"})
        report.update(status="complete", final_checkpoint=str(final) if policy != "none" else None)
        write_json(report_path, report)
    except BaseException as exc:
        report.update(status="failed", error=str(exc), recovery_workspace=str(work))
        write_json(report_path, report)
        print(f"Pipeline failed; recovery workspace retained: {work}", flush=True)
        raise
    shutil.rmtree(work)
    return None if policy == "none" else final
