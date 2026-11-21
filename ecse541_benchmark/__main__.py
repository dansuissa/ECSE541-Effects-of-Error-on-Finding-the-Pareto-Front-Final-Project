"""Small command line interface: download, build, check, audit, and space."""

import argparse
import json
import sys
from pathlib import Path

from .audit import audit_benchmark, write_data_report, write_space_report
from .benchmark import load_benchmark
from .extract import download_source, extract_file


def main(argv=None):
    parser = argparse.ArgumentParser(description="Part 1: reproducible NAS-Bench-201 inputs")
    commands = parser.add_subparsers(dest="command", required=True)
    space = commands.add_parser("space", help="Exhaustive architecture checks and figure; no dataset needed")
    space.add_argument("--out", type=Path, default=Path("results"))
    download = commands.add_parser("download", help="Download the authors' original benchmark (~4.7 GB)")
    download.add_argument("--out", type=Path, required=True)
    build = commands.add_parser("build", help="Extract all final CIFAR-100 validation trials")
    build.add_argument("--source", type=Path, required=True)
    build.add_argument("--out", type=Path, default=Path("data"))
    build.add_argument("--reports", type=Path, default=Path("results"))
    build.add_argument("--trust-source", action="store_true")
    build.add_argument("--expected-sha256")
    build.add_argument("--verify-api", action="store_true", help="Compare every mean with the optional authors' API")
    for command in ("check", "audit"):
        sub = commands.add_parser(command, help="Validate a saved bundle" if command == "check" else "Generate measured-data audit and figure")
        sub.add_argument("directory", type=Path)
        if command == "audit":
            sub.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    try:
        if args.command == "space":
            result = write_space_report(args.out)
        elif args.command == "download":
            result = {"downloaded": str(download_source(args.out))}
        elif args.command == "build":
            if args.out.exists() and any(args.out.iterdir()):
                raise FileExistsError(f"Output directory must be empty: {args.out}")
            benchmark = extract_file(
                args.source, trust_source=args.trust_source, expected_sha256=args.expected_sha256,
                progress=lambda n: print(f"Read {n:,} architecture records", file=sys.stderr, flush=True),
                verify_api=args.verify_api,
            )
            benchmark.save(args.out)
            # Check the exact exported bytes before reporting a successful build.
            benchmark = load_benchmark(args.out)
            write_space_report(args.reports)
            result = write_data_report(benchmark, args.reports)
        else:
            benchmark = load_benchmark(args.directory)
            result = write_data_report(benchmark, args.out) if args.command == "audit" else audit_benchmark(benchmark)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, ImportError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
