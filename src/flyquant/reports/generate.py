"""Generate machine-readable aggregate + human-readable Markdown comparison."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from flyquant import UPSTREAM_COMMIT, __version__
from flyquant.paths import REPORTS_DIR, RESULTS_DIR, ensure_output_dirs
from flyquant.reference.adapter import get_upstream_commit


def _load_results(results_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(results_dir.glob("*.json")):
        data = json.loads(path.read_text())
        data["_path"] = str(path)
        rows.append(data)
    return rows


def _pareto_front(points: list[dict], maximize_fidelity: str) -> list[dict]:
    usable = [
        p
        for p in points
        if p.get("compression_ratio") and p.get(maximize_fidelity) is not None
    ]
    front = []
    for p in usable:
        dominated = False
        for q in usable:
            if q is p:
                continue
            better_or_eq = (
                q["compression_ratio"] >= p["compression_ratio"]
                and q["wall_time_s"] <= p["wall_time_s"]
                and q[maximize_fidelity] >= p[maximize_fidelity]
            )
            strictly_better = (
                q["compression_ratio"] > p["compression_ratio"]
                or q["wall_time_s"] < p["wall_time_s"]
                or q[maximize_fidelity] > p[maximize_fidelity]
            )
            if better_or_eq and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(p)
    return front


def _row_from_result(r: dict) -> dict[str, Any]:
    status = r.get("status", "unknown")
    cfg = r.get("config") or {}
    name = cfg.get("name") or Path(r.get("_path", "unknown")).stem
    if status != "ok":
        return {
            "name": name,
            "status": status,
            "method": cfg.get("method"),
            "family": _family(cfg.get("method")),
            "reason": r.get("reason") or r.get("error"),
        }

    hyp = r.get("hypothesis") or {}
    neural = r.get("neural_fidelity") or {}
    beh = r.get("behavioural_fidelity") or {}
    variant = r.get("variant") or {}
    return {
        "name": name,
        "status": status,
        "method": variant.get("method") or cfg.get("method"),
        "family": _family(variant.get("method") or cfg.get("method")),
        "n_neurons": (r.get("model_counts") or {}).get("n_neurons"),
        "csr_nbytes": (r.get("model_counts") or {}).get("csr_nbytes"),
        "artefact_bytes": variant.get("artefact_bytes")
        or (variant.get("graph_storage") or {}).get("stored_nbytes")
        or variant.get("packed_nbytes"),
        "compression_ratio": r.get("compression_ratio_storage_vs_csr"),
        "compression_ratio_basis": variant.get("compression_ratio_basis"),
        "storage_precision": variant.get("storage_precision"),
        "execution_precision": variant.get("execution_precision"),
        "wall_time_s": hyp.get("wall_time_s"),
        "peak_rss_bytes": (r.get("resources") or {}).get("peak_rss_bytes"),
        "state_arrays_bytes": (variant.get("hypothesis_engine_state") or {}).get(
            "state_arrays_bytes"
        ),
        "gf_spikes": hyp.get("gf_spikes"),
        "rate_corr_active_ref": neural.get("rate_corr_active_ref"),
        "spike_match": neural.get("spike_count_exact_match_fraction"),
        "escape_agreement": beh.get("escape_agreement"),
        "false_negative": beh.get("false_negative"),
        "weight_mae": (variant.get("weight_reconstruction_error") or {}).get("mae"),
        "packed_nbytes": variant.get("packed_nbytes"),
        "unpacked_nbytes": variant.get("unpacked_nbytes"),
        "path": r.get("_path"),
    }


def _family(method: str | None) -> str:
    if not method:
        return "unknown"
    if method in ("fp16", "fp32", "int8", "int4"):
        return "weight_precision"
    if method in ("state_fp16", "state_fp64"):
        return "state_precision"
    if method == "prune" or method == "pipeline":
        return "prune_or_pipeline"
    if method == "lossless_graph":
        return "lossless"
    if method == "reference":
        return "reference"
    return "other"


def generate_report(
    results_dir: Path | None = None,
    out_dir: Path | None = None,
) -> dict[str, Path]:
    ensure_output_dirs()
    results_dir = Path(results_dir) if results_dir else RESULTS_DIR
    out_dir = Path(out_dir) if out_dir else REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = _load_results(results_dir)
    rows = [_row_from_result(r) for r in raw]

    ok_points = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        fid = row.get("rate_corr_active_ref")
        if fid is None:
            fid = row.get("spike_match")
        ok_points.append(
            {
                "name": row["name"],
                "compression_ratio": row.get("compression_ratio") or 1.0,
                "wall_time_s": row.get("wall_time_s") or 0.0,
                "fidelity": float(fid) if fid is not None else 0.0,
                "family": row.get("family"),
            }
        )
    pareto = _pareto_front(ok_points, "fidelity")

    fingerprints = {
        "flyquant_version": __version__,
        "upstream_commit_pinned": UPSTREAM_COMMIT,
        "upstream_commit_resolved": get_upstream_commit(),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
    }

    aggregate = {
        "fingerprints": fingerprints,
        "n_records": len(rows),
        "n_ok": sum(1 for r in rows if r.get("status") == "ok"),
        "n_error": sum(1 for r in rows if r.get("status") == "error"),
        "n_unexecuted": sum(1 for r in rows if r.get("status") == "unexecuted"),
        "rows": rows,
        "pareto_front": pareto,
        "by_family": {
            fam: [r for r in rows if r.get("family") == fam]
            for fam in sorted({r.get("family") for r in rows})
        },
    }
    json_path = out_dir / "comparison.json"
    json_path.write_text(json.dumps(aggregate, indent=2, default=str))

    csv_path = out_dir / "comparison.csv"
    fieldnames = [
        "name",
        "status",
        "method",
        "family",
        "storage_precision",
        "execution_precision",
        "compression_ratio",
        "compression_ratio_basis",
        "wall_time_s",
        "rate_corr_active_ref",
        "escape_agreement",
        "gf_spikes",
        "artefact_bytes",
        "state_arrays_bytes",
        "reason",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    md_lines = [
        "# FlyQuant comparison report",
        "",
        "## Software fingerprints",
        "",
        f"- FlyQuant: `{fingerprints['flyquant_version']}`",
        f"- Upstream pinned: `{fingerprints['upstream_commit_pinned']}`",
        f"- Upstream resolved: `{fingerprints['upstream_commit_resolved']}`",
        f"- Python: `{fingerprints['python_version']}`",
        f"- NumPy: `{fingerprints['numpy_version']}`",
        "",
        f"Records: {aggregate['n_records']} "
        f"(ok={aggregate['n_ok']}, error={aggregate['n_error']}, "
        f"unexecuted={aggregate['n_unexecuted']})",
        "",
        "Compressed variants are compared against the same reference configuration.",
        "Storage compression is not claimed as live-memory savings unless measured.",
        "Statuses: **ok** = measured; **unexecuted** = skipped (data/RAM); **error** = failed.",
        "",
        "## Results table",
        "",
        "| name | status | family | method | store | exec | ratio | wall_s | rate_corr | escape_ok |",
        "|---|---|---|---|---|---|---:|---:|---:|---|",
    ]
    for r in rows:
        md_lines.append(
            "| {name} | {status} | {family} | {method} | {storage_precision} | {execution_precision} | "
            "{compression_ratio} | {wall_time_s} | {rate_corr_active_ref} | {escape_agreement} |".format(
                name=r.get("name"),
                status=r.get("status"),
                family=r.get("family"),
                method=r.get("method"),
                storage_precision=r.get("storage_precision"),
                execution_precision=r.get("execution_precision"),
                compression_ratio=_fmt(r.get("compression_ratio")),
                wall_time_s=_fmt(r.get("wall_time_s")),
                rate_corr_active_ref=_fmt(r.get("rate_corr_active_ref")),
                escape_agreement=r.get("escape_agreement"),
            )
        )

    md_lines += ["", "## By family", ""]
    for fam, fam_rows in aggregate["by_family"].items():
        md_lines.append(f"### {fam} ({len(fam_rows)} rows)")
        for r in fam_rows:
            md_lines.append(
                f"- `{r.get('name')}` status={r.get('status')} "
                f"ratio={_fmt(r.get('compression_ratio'))} "
                f"corr={_fmt(r.get('rate_corr_active_ref'))}"
            )
        md_lines.append("")

    md_lines += [
        "## Pareto front (compression ↑, time ↓, fidelity ↑)",
        "",
    ]
    if not pareto:
        md_lines.append("_No completed runs with compression ratios yet._")
    else:
        for p in pareto:
            md_lines.append(
                f"- **{p['name']}**: ratio={p['compression_ratio']:.3f}, "
                f"wall={p['wall_time_s']:.3f}s, fidelity={p['fidelity']:.4f}"
            )

    md_lines += [
        "",
        "## Plots",
        "",
        "See `plots/` for weight-precision, state-precision, prune, and Pareto figures.",
        "",
        "## Caveats",
        "",
        "- Fidelity is to the computational model, not a living animal.",
        "- Small neural errors can cause large behavioural divergences (thresholded spikes).",
        "- INT8/INT4 use reconstruct-then-simulate; not native integer kernels.",
        "- INT4 nibble packing is storage-only.",
        "- `state_fp16` keeps float32 weights; weight-`fp16` also lowers wdata dtype.",
        "",
    ]
    md_path = out_dir / "comparison.md"
    md_path.write_text("\n".join(md_lines))

    plot_dir = out_dir / "plots"
    try:
        _make_plots(rows, plot_dir)
    except Exception as exc:
        (out_dir / "plots_error.txt").write_text(str(exc))

    return {
        "json": json_path,
        "markdown": md_path,
        "csv": csv_path,
        "plots": plot_dir,
    }


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _make_plots(rows: list[dict], plot_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    ok = [r for r in rows if r.get("status") == "ok"]

    def scatter(subset, xkey, ykey, fname, xlabel, ylabel):
        xs, ys, labels = [], [], []
        for r in subset:
            if r.get(xkey) is None or r.get(ykey) is None:
                continue
            yv = r[ykey]
            if isinstance(yv, bool):
                yv = float(yv)
            xs.append(r[xkey])
            ys.append(yv)
            labels.append(r.get("name") or "")
        if not xs:
            return
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(xs, ys)
        for x, y, lab in zip(xs, ys, labels):
            ax.annotate(lab, (x, y), fontsize=7, alpha=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / fname, dpi=120)
        plt.close(fig)

    scatter(
        ok,
        "compression_ratio",
        "rate_corr_active_ref",
        "compression_vs_neural.png",
        "Compression ratio",
        "Rate correlation (active ref)",
    )
    for r in ok:
        if isinstance(r.get("escape_agreement"), bool):
            r = dict(r)
    ok_bool = []
    for r in ok:
        rr = dict(r)
        if isinstance(rr.get("escape_agreement"), bool):
            rr["escape_agreement"] = float(rr["escape_agreement"])
        ok_bool.append(rr)
    scatter(
        ok_bool,
        "compression_ratio",
        "escape_agreement",
        "compression_vs_behaviour.png",
        "Compression ratio",
        "Escape agreement (1/0)",
    )
    scatter(
        ok,
        "compression_ratio",
        "wall_time_s",
        "compression_vs_speed.png",
        "Compression ratio",
        "Simulation wall time (s)",
    )
    scatter(
        [r for r in ok if r.get("family") == "weight_precision"],
        "weight_mae",
        "rate_corr_active_ref",
        "weight_precision_ladder.png",
        "Weight reconstruction MAE (mV)",
        "Rate correlation",
    )
    scatter(
        [r for r in ok if r.get("family") == "state_precision"],
        "state_arrays_bytes",
        "rate_corr_active_ref",
        "state_precision_ladder.png",
        "State arrays bytes",
        "Rate correlation",
    )
    scatter(
        [r for r in ok if r.get("family") == "prune_or_pipeline"],
        "compression_ratio",
        "rate_corr_active_ref",
        "prune_curve.png",
        "Compression / nnz ratio",
        "Rate correlation",
    )
    scatter(
        ok,
        "artefact_bytes",
        "peak_rss_bytes",
        "size_vs_rss.png",
        "Artefact bytes",
        "Peak RSS bytes",
    )
