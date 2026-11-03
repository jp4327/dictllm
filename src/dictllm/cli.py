"""Minimal LLaMA compression CLI; quantization always uses the identity path."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .pipeline import quantize_stage, run_pipeline
from .stages import factorize_stage, prune_stage, refit_stage


def build_parser():
    parser = argparse.ArgumentParser(prog="dictllm")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["factorize", "prune", "refit", "quantize", "run"]:
        command = commands.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
        command.add_argument("--overwrite", action="store_true")
        command.add_argument("--skip-evaluation", action="store_true")
        if name != "run":
            command.add_argument("--output", type=Path)
        if name in {"prune", "refit", "quantize"}:
            command.add_argument("--checkpoint", type=Path)
        if name in {"factorize", "run"}:
            command.add_argument("--force-calibration", action="store_true")
        if name == "quantize":
            command.add_argument("--export-mode", choices=["fake", "packed"])
            command.add_argument("--keep-intermediates", action="store_true")
        if name == "run":
            command.add_argument("--quantize", action="store_true", help="Append identity NVFP4 quantization after refit.")
            command.add_argument("--checkpoint-policy", choices=["final", "all", "none"])
            command.add_argument("--dry-run", action="store_true")
    inspect = commands.add_parser("inspect")
    inspect.add_argument("checkpoint", type=Path)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "inspect":
        for name in ["dictllm_manifest.json", "packed_nvfp4_manifest.json", "zero_preserving_summary.json"]:
            path = args.checkpoint / name
            if path.is_file():
                print(path.read_text())
                return
        raise FileNotFoundError(f"No DictLLM metadata found under {args.checkpoint}")
    config = load_config(args.config)
    common = dict(overwrite=args.overwrite, skip_evaluation=args.skip_evaluation)
    if args.command == "run":
        run_pipeline(config, quantize=args.quantize, policy=args.checkpoint_policy,
                     force_calibration=args.force_calibration, dry_run=args.dry_run, **common)
    elif args.command == "factorize":
        factorize_stage(config, output=args.output, force_calibration=args.force_calibration, **common)
    elif args.command == "prune":
        prune_stage(config, checkpoint=args.checkpoint, output=args.output, **common)
    elif args.command == "refit":
        refit_stage(config, checkpoint=args.checkpoint, output=args.output, **common)
    else:
        quantize_stage(config, checkpoint=args.checkpoint, output=args.output,
                       export_mode=args.export_mode, keep_intermediates=args.keep_intermediates, **common)


if __name__ == "__main__":
    main()
