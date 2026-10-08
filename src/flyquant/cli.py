"""FlyQuant command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from flyquant import DATASET_ID, UPSTREAM_COMMIT, __version__
from flyquant.paths import (
    CONNECTOME_BIN,
    EXPERIMENTS_DIR,
    META_JSON,
    NEURONS_BIN,
    REFERENCE_DIR,
    REPORTS_DIR,
    RESULTS_DIR,
    ensure_output_dirs,
)


def cmd_inspect(_: argparse.Namespace) -> int:
    from flyquant.reference.adapter import get_upstream_commit
    from flyquant.reference.web_loader import estimate_web_load_bytes, web_assets_available

    print(f"FlyQuant {__version__}")
    print(f"Dataset target: {DATASET_ID}")
    print(f"Upstream path: {REFERENCE_DIR}")
    print(f"Upstream commit (pinned): {UPSTREAM_COMMIT}")
    print(f"Upstream commit (resolved): {get_upstream_commit()}")
    print(f"Web assets present: {web_assets_available()}")
    if web_assets_available():
        print(f"  connectome.bin: {CONNECTOME_BIN} ({CONNECTOME_BIN.stat().st_size} bytes)")
        print(f"  neurons.bin:    {NEURONS_BIN} ({NEURONS_BIN.stat().st_size} bytes)")
        print(f"  meta.json:      {META_JSON} ({META_JSON.stat().st_size} bytes)")
        print(f"  approx file bytes: {estimate_web_load_bytes()}")
    print("Available experiments (conceptual):")
    print("  escape          — Poisson LC4/LPLC2 → DNp01 probe (FlyQuant harness)")
    print("  escape_subgraph — same on ~4k-neuron escape neighbourhood")
    print("  upstream_01     — full geometric looming (needs derived connectome)")
    print("Compression methods: reference, fp16, int8, int4, lossless_graph, prune")
    return 0


def cmd_baseline(args: argparse.Namespace) -> int:
    from flyquant.benchmarks.harness import BenchmarkHarness

    ensure_output_dirs()
    subgraph = None
    duration = args.duration_ms
    source = args.source
    min_ram = 0.0
    if args.experiment in ("escape", "escape_poisson"):
        subgraph = None
        min_ram = 3.5
    elif args.experiment == "escape_subgraph":
        subgraph = "escape"
        duration = args.duration_ms or 80.0
    else:
        print(f"Unknown experiment '{args.experiment}'", file=sys.stderr)
        return 2

    cfg = {
        "name": f"baseline_{args.experiment}",
        "method": "reference",
        "experiment": args.experiment,
        "connectome_source": source,
        "subgraph": subgraph,
        "seed": args.seed,
        "duration_ms": duration,
        "rate_hz": args.rate_hz,
        "min_ram_gb": min_ram if not args.force else 0.0,
        "execution_dtype": "fp32",
    }
    harness = BenchmarkHarness()
    rec = harness.run_config(cfg, use_cache=not args.no_cache)
    print(json.dumps(rec.to_dict(), indent=2, default=str))
    return 0 if rec.status in ("ok", "unexecuted") else 1


def cmd_compress(args: argparse.Namespace) -> int:
    from flyquant.benchmarks.harness import BenchmarkHarness

    ensure_output_dirs()
    subgraph = "escape" if args.experiment == "escape_subgraph" else None
    cfg = {
        "name": f"{args.method}_{args.experiment}",
        "method": args.method,
        "experiment": args.experiment,
        "connectome_source": args.source,
        "subgraph": subgraph,
        "seed": args.seed,
        "duration_ms": args.duration_ms,
        "rate_hz": args.rate_hz,
        "min_ram_gb": 0.0 if args.force or subgraph else 3.5,
        "gzip": True,
        "native_fp16_execution": True,
        "prune_method": args.prune_method,
        "prune_level": args.prune_level,
    }
    harness = BenchmarkHarness()
    rec = harness.run_config(cfg, use_cache=not args.no_cache)
    print(json.dumps(rec.to_dict(), indent=2, default=str))
    return 0 if rec.status in ("ok", "unexecuted") else 1


def cmd_verify(args: argparse.Namespace) -> int:
    from flyquant.compression.graph import (
        compress_graph_lossless,
        graph_identical,
        load_graph_artefact,
        reconstruct_csr,
    )
    from flyquant.reference.adapter import load_reference_connectome

    subgraph = "escape" if args.experiment == "escape_subgraph" else None
    c = load_reference_connectome(source=args.source, subgraph=subgraph)
    if args.artefact:
        art = load_graph_artefact(Path(args.artefact))
        recon = reconstruct_csr(art)
    else:
        art = compress_graph_lossless(c.w)
        recon = reconstruct_csr(art)
    result = graph_identical(c.w, recon)
    print(json.dumps(result, indent=2))
    return 0 if result["identical"] else 1


def cmd_benchmark(args: argparse.Namespace) -> int:
    from flyquant.benchmarks.harness import BenchmarkHarness

    harness = BenchmarkHarness()
    if args.suite:
        records = harness.run_suite(Path(args.suite), use_cache=not args.no_cache)
    else:
        variants = [v.strip() for v in args.compare.split(",") if v.strip()]
        records = []
        for name in variants:
            method = "reference" if name in ("reference", "fp32", "baseline") else name
            cfg = {
                "name": f"bench_{name}",
                "method": method if method != "fp32" else "fp32",
                "experiment": args.experiment,
                "connectome_source": args.source,
                "subgraph": "escape" if args.experiment == "escape_subgraph" else None,
                "seed": args.seed,
                "duration_ms": args.duration_ms,
                "rate_hz": args.rate_hz,
                "min_ram_gb": 0.0 if args.force else (0.0 if args.experiment == "escape_subgraph" else 3.5),
            }
            records.append(harness.run_config(cfg, use_cache=not args.no_cache))
    payload = [r.to_dict() for r in records]
    print(json.dumps(payload, indent=2, default=str))
    return 0 if all(r.status in ("ok", "unexecuted") for r in records) else 1


def cmd_report(_: argparse.Namespace) -> int:
    from flyquant.reports.generate import generate_report

    paths = generate_report()
    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))
    print(f"Markdown report: {paths['markdown']}")
    return 0


def cmd_suite(args: argparse.Namespace) -> int:
    from flyquant.benchmarks.harness import BenchmarkHarness

    path = Path(args.suite)
    if not path.exists():
        path = EXPERIMENTS_DIR / args.suite
    harness = BenchmarkHarness()
    records = harness.run_suite(path, use_cache=not args.no_cache)
    print(json.dumps([r.to_dict() for r in records], indent=2, default=str))
    return 0 if all(r.status in ("ok", "unexecuted") for r in records) else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="flyquant",
        description="Compress and benchmark a fruit-fly connectome LIF model",
    )
    p.add_argument("--version", action="version", version=f"flyquant {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("inspect", help="Show upstream model and asset status")
    s.set_defaults(func=cmd_inspect)

    s = sub.add_parser("baseline", help="Run reference baseline experiment")
    s.add_argument("--experiment", default="escape_subgraph",
                   choices=["escape", "escape_subgraph", "escape_poisson"])
    s.add_argument("--source", default="auto", choices=["auto", "web", "derived", "synthetic"])
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--duration-ms", type=float, default=80.0)
    s.add_argument("--rate-hz", type=float, default=150.0)
    s.add_argument("--force", action="store_true", help="Ignore RAM gate")
    s.add_argument("--no-cache", action="store_true")
    s.set_defaults(func=cmd_baseline)

    s = sub.add_parser("compress", help="Create/evaluate a compressed variant")
    s.add_argument("--method", required=True,
                   choices=["fp16", "fp32", "int8", "int4", "lossless_graph", "prune"])
    s.add_argument("--experiment", default="escape_subgraph")
    s.add_argument("--source", default="auto", choices=["auto", "web", "derived", "synthetic"])
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--duration-ms", type=float, default=80.0)
    s.add_argument("--rate-hz", type=float, default=150.0)
    s.add_argument("--prune-method", default="keep_fraction",
                   choices=["threshold", "percentile", "keep_fraction"])
    s.add_argument("--prune-level", type=float, default=0.9)
    s.add_argument("--force", action="store_true")
    s.add_argument("--no-cache", action="store_true")
    s.set_defaults(func=cmd_compress)

    s = sub.add_parser("verify", help="Verify lossless graph round-trip")
    s.add_argument("--experiment", default="escape_subgraph")
    s.add_argument("--source", default="auto", choices=["auto", "web", "derived", "synthetic"])
    s.add_argument("--artefact", default=None, help="Path to graph .json meta")
    s.set_defaults(func=cmd_verify)

    s = sub.add_parser("benchmark", help="Compare variants against reference")
    s.add_argument("--compare", default="reference,fp16,int8,lossless_graph")
    s.add_argument("--suite", default=None)
    s.add_argument("--experiment", default="escape_subgraph")
    s.add_argument("--source", default="auto", choices=["auto", "web", "derived", "synthetic"])
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--duration-ms", type=float, default=80.0)
    s.add_argument("--rate-hz", type=float, default=150.0)
    s.add_argument("--force", action="store_true")
    s.add_argument("--no-cache", action="store_true")
    s.set_defaults(func=cmd_benchmark)

    s = sub.add_parser("suite", help="Run a YAML experiment suite")
    s.add_argument("suite", help="Path or name under experiments/")
    s.add_argument("--no-cache", action="store_true")
    s.set_defaults(func=cmd_suite)

    s = sub.add_parser("report", help="Generate comparison report from results/")
    s.set_defaults(func=cmd_report)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
