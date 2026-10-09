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
    ensure_output_dirs,
)


def _add_common_run_args(s: argparse.ArgumentParser) -> None:
    s.add_argument("--source", default="auto", choices=["auto", "web", "derived", "synthetic"])
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--duration-ms", type=float, default=80.0)
    s.add_argument("--rate-hz", type=float, default=150.0)
    s.add_argument(
        "--execution-dtype",
        default="fp32",
        choices=["fp32", "fp16", "fp64"],
        help="Runtime state dtype for v/g/ring (and wdata unless overridden)",
    )
    s.add_argument("--force", action="store_true", help="Ignore RAM gate")
    s.add_argument("--no-cache", action="store_true")
    s.add_argument("--record-spikes", action="store_true")
    s.add_argument("--record-v", action="store_true", help="Sample membrane traces")


def _subgraph_for_experiment(experiment: str) -> str | None:
    if experiment == "escape_subgraph":
        return "escape"
    return None


def cmd_inspect(_: argparse.Namespace) -> int:
    from flyquant.reference.adapter import derived_data_status, get_upstream_commit
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
    derived = derived_data_status()
    print(f"Derived connectome available: {derived['available']}")
    if not derived["available"]:
        print(f"  reason: {derived['reason']}")
    print("Available experiments:")
    print("  escape / escape_poisson — Poisson LC4/LPLC2 → DNp01")
    print("  escape_subgraph         — same on ~4k-neuron neighbourhood")
    print("  looming                 — geometric looming (needs derived data)")
    print("  escape_controls         — lesion controls (needs derived data)")
    print(
        "Methods: reference, fp16, int8, int4, state_fp16, state_fp64, "
        "lossless_graph, prune, pipeline"
    )
    return 0


def cmd_baseline(args: argparse.Namespace) -> int:
    from flyquant.benchmarks.harness import BenchmarkHarness

    ensure_output_dirs()
    experiment = args.experiment
    subgraph = _subgraph_for_experiment(experiment)
    min_ram = 0.0
    if experiment in ("escape", "escape_poisson") and subgraph is None:
        min_ram = 3.5
    if experiment in ("looming", "escape_controls"):
        min_ram = 3.5

    cfg = {
        "name": f"baseline_{experiment}",
        "method": "reference",
        "experiment": experiment,
        "connectome_source": args.source,
        "subgraph": subgraph,
        "seed": args.seed,
        "duration_ms": args.duration_ms,
        "rate_hz": args.rate_hz,
        "min_ram_gb": 0.0 if args.force else min_ram,
        "execution_dtype": args.execution_dtype,
        "record_spikes": args.record_spikes,
        "record_v": args.record_v,
    }
    harness = BenchmarkHarness()
    rec = harness.run_config(cfg, use_cache=not args.no_cache)
    print(json.dumps(rec.to_dict(), indent=2, default=str))
    return 0 if rec.status in ("ok", "unexecuted") else 1


def cmd_compress(args: argparse.Namespace) -> int:
    from flyquant.benchmarks.harness import BenchmarkHarness

    ensure_output_dirs()
    subgraph = _subgraph_for_experiment(args.experiment)
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
        "pack_int4": True,
        "prune_method": args.prune_method,
        "prune_level": args.prune_level,
        "execution_dtype": args.execution_dtype,
        "record_spikes": args.record_spikes,
        "record_v": args.record_v,
    }
    if args.method == "pipeline":
        cfg["steps"] = [
            {
                "op": "prune",
                "prune_method": args.prune_method,
                "prune_level": args.prune_level,
            },
            {"op": "quantise", "precision": "int8"},
        ]
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

    subgraph = _subgraph_for_experiment(args.experiment)
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
        subgraph = _subgraph_for_experiment(args.experiment)
        for name in variants:
            method = "reference" if name in ("reference", "baseline") else name
            if name == "fp32":
                method = "fp32"
            cfg = {
                "name": f"bench_{name}",
                "method": method,
                "experiment": args.experiment,
                "connectome_source": args.source,
                "subgraph": subgraph,
                "seed": args.seed,
                "duration_ms": args.duration_ms,
                "rate_hz": args.rate_hz,
                "min_ram_gb": 0.0 if args.force or subgraph else 3.5,
                "execution_dtype": args.execution_dtype,
                "pack_int4": True,
                "record_spikes": args.record_spikes,
                "record_v": args.record_v,
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

    exp_choices = [
        "escape",
        "escape_subgraph",
        "escape_poisson",
        "looming",
        "escape_controls",
    ]

    s = sub.add_parser("baseline", help="Run reference baseline experiment")
    s.add_argument("--experiment", default="escape_subgraph", choices=exp_choices)
    _add_common_run_args(s)
    s.set_defaults(func=cmd_baseline)

    s = sub.add_parser("compress", help="Create/evaluate a compressed variant")
    s.add_argument(
        "--method",
        required=True,
        choices=[
            "fp16",
            "fp32",
            "int8",
            "int4",
            "state_fp16",
            "state_fp64",
            "lossless_graph",
            "prune",
            "pipeline",
        ],
    )
    s.add_argument("--experiment", default="escape_subgraph", choices=exp_choices)
    s.add_argument(
        "--prune-method",
        default="keep_fraction",
        choices=[
            "threshold",
            "percentile",
            "keep_fraction",
            "importance_magnitude_outdegree",
            "importance_dnp01_path",
        ],
    )
    s.add_argument("--prune-level", type=float, default=0.9)
    _add_common_run_args(s)
    s.set_defaults(func=cmd_compress)

    s = sub.add_parser("verify", help="Verify lossless graph round-trip")
    s.add_argument("--experiment", default="escape_subgraph", choices=exp_choices)
    s.add_argument("--source", default="auto", choices=["auto", "web", "derived", "synthetic"])
    s.add_argument("--artefact", default=None, help="Path to graph .json meta")
    s.set_defaults(func=cmd_verify)

    s = sub.add_parser("benchmark", help="Compare variants against reference")
    s.add_argument(
        "--compare",
        default="reference,fp16,int8,int4,state_fp16,lossless_graph",
    )
    s.add_argument("--suite", default=None)
    s.add_argument("--experiment", default="escape_subgraph", choices=exp_choices)
    _add_common_run_args(s)
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
