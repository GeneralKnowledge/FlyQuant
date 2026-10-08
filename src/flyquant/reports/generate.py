"""Generate machine-readable aggregate + human-readable Markdown comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from flyquant.paths import REPORTS_DIR, RESULTS_DIR, ensure_output_dirs


def _load_results(results_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name.endswith("_error.json"):
            data = json.loads(path.read_text())
            data["_path"] = str(path)
            rows.append(data)
            continue
        data = json.loads(path.read_text())
        data["_path"] = str(path)
        rows.append(data)
    return rows


def _pareto_front(points: list[dict], maximize_fidelity: str) -> list[dict]:
    """
    Pareto over compression_ratio (max), wall_time (min), fidelity (max).
    """
    usable = [p for p in points if p.get("compression_ratio") and p.get(maximize_fidelity) is not None]
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
    cfg = r.get("config") or r.get("results", {}).get("config") or {}
    # files may be the full record
    name = cfg.get("name") or r.get("config", {}).get("name") or Path(r.get("_path", "unknown")).stem
    if status != "ok":
        return {
            "name": name,
            "status": status,
            "method": cfg.get("method"),
            "reason": r.get("reason") or r.get("error"),
        }

    hyp = r.get("hypothesis") or {}
    neural = r.get("neural_fidelity") or {}
    beh = r.get("behavioural_fidelity") or {}
    variant = r.get("variant") or {}
    return {
        "name": name,
        "status": status,
        "method": (r.get("variant") or {}).get("method") or cfg.get("method"),
        "n_neurons": (r.get("model_counts") or {}).get("n_neurons"),
        "csr_nbytes": (r.get("model_counts") or {}).get("csr_nbytes"),
        "artefact_bytes": variant.get("artefact_bytes")
        or (variant.get("graph_storage") or {}).get("stored_nbytes"),
        "compression_ratio": r.get("compression_ratio_storage_vs_csr"),
        "storage_precision": variant.get("storage_precision"),
        "execution_precision": variant.get("execution_precision"),
        "wall_time_s": hyp.get("wall_time_s"),
        "peak_rss_bytes": (r.get("resources") or {}).get("peak_rss_bytes"),
        "gf_spikes": hyp.get("gf_spikes"),
        "rate_corr_active_ref": neural.get("rate_corr_active_ref"),
        "spike_match": neural.get("spike_count_exact_match_fraction"),
        "escape_agreement": beh.get("escape_agreement"),
        "false_negative": beh.get("false_negative"),
        "weight_mae": (variant.get("weight_reconstruction_error") or {}).get("mae"),
        "path": r.get("_path"),
    }


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
        ok_points.append({
            "name": row["name"],
            "compression_ratio": row.get("compression_ratio") or 1.0,
            "wall_time_s": row.get("wall_time_s") or 0.0,
            "fidelity": float(fid) if fid is not None else 0.0,
        })
    pareto = _pareto_front(ok_points, "fidelity")

    aggregate = {
        "n_records": len(rows),
        "n_ok": sum(1 for r in rows if r.get("status") == "ok"),
        "n_error": sum(1 for r in rows if r.get("status") == "error"),
        "n_unexecuted": sum(1 for r in rows if r.get("status") == "unexecuted"),
        "rows": rows,
        "pareto_front": pareto,
    }
    json_path = out_dir / "comparison.json"
    json_path.write_text(json.dumps(aggregate, indent=2, default=str))

    md_lines = [
        "# FlyQuant comparison report",
        "",
        f"Records: {aggregate['n_records']} "
        f"(ok={aggregate['n_ok']}, error={aggregate['n_error']}, "
        f"unexecuted={aggregate['n_unexecuted']})",
        "",
        "Compressed variants are compared against the same reference configuration.",
        "Storage compression is not claimed as live-memory savings unless measured.",
        "",
        "## Results table",
        "",
        "| name | status | method | store | exec | ratio | wall_s | rate_corr | escape_ok |",
        "|---|---|---|---|---|---:|---:|---:|---|",
    ]
    for r in rows:
        md_lines.append(
            "| {name} | {status} | {method} | {storage_precision} | {execution_precision} | "
            "{compression_ratio} | {wall_time_s} | {rate_corr_active_ref} | {escape_agreement} |".format(
                name=r.get("name"),
                status=r.get("status"),
                method=r.get("method"),
                storage_precision=r.get("storage_precision"),
                execution_precision=r.get("execution_precision"),
                compression_ratio=_fmt(r.get("compression_ratio")),
                wall_time_s=_fmt(r.get("wall_time_s")),
                rate_corr_active_ref=_fmt(r.get("rate_corr_active_ref")),
                escape_agreement=r.get("escape_agreement"),
            )
        )

    md_lines += [
        "",
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
        "See `plots/` if matplotlib generation succeeded.",
        "",
        "## Caveats",
        "",
        "- Fidelity is to the computational model, not a living animal.",
        "- Small neural errors can cause large behavioural divergences (thresholded spikes).",
        "- INT8/INT4 use reconstruct-then-simulate unless noted; they are not native kernels.",
        "",
    ]
    md_path = out_dir / "comparison.md"
    md_path.write_text("\n".join(md_lines))

    plot_dir = out_dir / "plots"
    try:
        _make_plots(rows, plot_dir)
    except Exception as exc:
        (out_dir / "plots_error.txt").write_text(str(exc))

    return {"json": json_path, "markdown": md_path, "plots": plot_dir}


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
    ok = [r for r in rows if r.get("status") == "ok" and r.get("compression_ratio")]

    def scatter(xkey, ykey, fname, xlabel, ylabel):
        xs, ys, labels = [], [], []
        for r in ok:
            if r.get(xkey) is None or r.get(ykey) is None:
                continue
            xs.append(r[xkey])
            ys.append(r[ykey])
            labels.append(r.get("name") or "")
        if not xs:
            return
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(xs, ys)
        for x, y, lab in zip(xs, ys, labels):
            ax.annotate(lab, (x, y), fontsize=8, alpha=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / fname, dpi=120)
        plt.close(fig)

    scatter(
        "compression_ratio",
        "rate_corr_active_ref",
        "compression_vs_neural.png",
        "Compression ratio (storage vs CSR)",
        "Rate correlation (active ref)",
    )
    scatter(
        "compression_ratio",
        "escape_agreement",
        "compression_vs_behaviour.png",
        "Compression ratio",
        "Escape agreement (1/0)",
    )
    # map bool to float for plot
    for r in ok:
        if isinstance(r.get("escape_agreement"), bool):
            r["escape_agreement"] = float(r["escape_agreement"])
    scatter(
        "compression_ratio",
        "wall_time_s",
        "compression_vs_speed.png",
        "Compression ratio",
        "Simulation wall time (s)",
    )
    scatter(
        "artefact_bytes",
        "peak_rss_bytes",
        "size_vs_rss.png",
        "Artefact bytes",
        "Peak RSS bytes",
    )
    scatter(
        "weight_mae",
        "rate_corr_active_ref",
        "quant_error_vs_fidelity.png",
        "Weight reconstruction MAE (mV)",
        "Rate correlation",
    )
