#!/usr/bin/env python
"""Run the migrated identity quantizer, optionally followed by exact packing."""

from __future__ import annotations

import argparse
from pathlib import Path

from .run_quantization import main as quantize_main, parse_args, resolve_checkpoint


def plan(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--export-mode", choices=("fake", "packed"), default="fake")
    parser.add_argument("--fake-output-path", type=Path,
                        help="Retained intermediate fake checkpoint for packed export; defaults to OUTPUT_fake.")
    parser.add_argument("--chunk-rows", type=int, default=64)
    parser.add_argument("--shard-mb", type=int, default=512)
    parser.add_argument("--dry-run", action="store_true", help="Validate and print paths without loading weights or data.")
    args, quantizer_options = parser.parse_known_args(argv)
    source = resolve_checkpoint(args.source_checkpoint)
    output = args.output_path.expanduser().resolve()
    if args.export_mode == "fake" and args.fake_output_path is not None:
        parser.error("--fake-output-path is only used for packed export.")
    if min(args.chunk_rows, args.shard_mb) < 1:
        parser.error("--chunk-rows and --shard-mb must be positive.")
    fake = output if args.export_mode == "fake" else (
        args.fake_output_path.expanduser().resolve() if args.fake_output_path is not None
        else output.with_name(output.name + "_fake")
    )
    paths = [source, fake] + ([output] if args.export_mode == "packed" else [])
    for index, left in enumerate(paths):
        for right in paths[index + 1:]:
            if left == right or left in right.parents or right in left.parents:
                parser.error("Source, fake, and packed checkpoints must be separate, non-nested directories.")
    for path in paths[1:]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}.")
    quantizer_argv = ["--source-checkpoint", str(source), "--output-path", str(fake), *quantizer_options]
    settings = parse_args(quantizer_argv)
    if resolve_checkpoint(settings.source_checkpoint) != source or settings.output_path.resolve() != fake:
        parser.error("Do not override source/output with alternate quantizer option names.")
    if args.export_mode == "packed" and (
        settings.w_group_size != 16 or settings.scale_precision != "e4m3" or settings.max_modules is not None
    ):
        parser.error("Packed export needs group size 16, E4M3 scales, and a complete selected-target run (no --max-modules).")
    return args, source, fake, output, quantizer_argv


def main(argv=None):
    args, source, fake, output, quantizer_argv = plan(argv)
    print(f"source={source}\nexport_mode={args.export_mode}\nfake_checkpoint={fake}")
    if args.export_mode == "packed":
        print(f"packed_checkpoint={output}\nruntime_default=dequantize (no activation quantization)")
    if args.dry_run:
        return
    quantize_main(quantizer_argv)
    if args.export_mode == "packed":
        from .convert_fake_to_packed import convert_checkpoint

        _, report = convert_checkpoint(fake, output, source_checkpoint=source,
                                       chunk_rows=args.chunk_rows, shard_mb=args.shard_mb)
        if not report["passed"]:
            raise RuntimeError("Packed export did not pass exact offline verification.")
        print(f"saved_packed_checkpoint={output}")


if __name__ == "__main__":
    main()
