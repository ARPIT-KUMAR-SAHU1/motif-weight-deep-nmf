"""
dashboard.py - Local, dependency-light web UI for exploring the experiment
results and explaining the research.

Run with:  python run_dashboard.py
Then open: http://127.0.0.1:8000

This module only READS results/*.csv, results/summary.json and figures/*.png.
It never re-runs an experiment and never modifies research outputs.
"""
import csv
import json
import math
import os
import re
from collections import defaultdict

from flask import Flask, render_template, jsonify, send_from_directory, abort

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "figures")

app = Flask(__name__, template_folder="templates", static_folder="static")

# Information already present in the project's CP1 report (Team8_...pdf).
TEAM = {
    "title": "Stability-Guided Selection of the Motif Weight in Deep NMF-Based Community Detection",
    "name": "Arpit Sahu",
    "roll": "S24BCAU0195",
    "team": "Team 8 (Individual / Solo)",
    "course": "CSET347 - BCA Semester V",
    "institution": "Bennett University",
    "instructor": "Dr. Soumita Das",
    "track": "Topic 4 - Deep Learning for Community Detection",
}


# ---------------------------------------------------------------------------
# Low-level readers / formatters
# ---------------------------------------------------------------------------

def _read_csv(name):
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _num(value):
    """Return a finite float or None (handles blanks, 'nan', bad strings)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) or math.isinf(v) else v


def _fmt(value):
    v = _num(value)
    return "N/A" if v is None else f"{v:.3f}"


def _fmt_pair(mean, std):
    m, s = _fmt(mean), _fmt(std)
    return "N/A" if m == "N/A" else f"{m} ± {s}"


def _mean(values):
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _summary():
    path = os.path.join(RESULTS_DIR, "summary.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                pass
    return {
        "generated_at": "not yet generated", "mode": "-", "experiments": [],
        "datasets": [], "environment": {}, "method": {},
    }


def _figures():
    if not os.path.isdir(FIGURES_DIR):
        return []
    return sorted(f for f in os.listdir(FIGURES_DIR) if f.lower().endswith(".png"))


def _csv_files():
    if not os.path.isdir(RESULTS_DIR):
        return []
    return sorted(f for f in os.listdir(RESULTS_DIR) if f.lower().endswith(".csv"))


def _dataset_overview():
    """Aggregate dataset statistics directly from summary.json."""
    datasets = _summary().get("datasets", [])
    return {
        "count": len(datasets),
        "nodes": sum(int(d.get("n") or 0) for d in datasets),
        "edges": sum(int(d.get("edges") or 0) for d in datasets),
        "labeled": sum(1 for d in datasets if d.get("labeled")),
        "synthetic": sum(1 for d in datasets if d.get("synthetic")),
    }


def _metric_for(rows):
    """Which metric is meaningful for a set of rows: ACC/NMI if labeled."""
    if any(_num(r.get("NMI_mean")) is not None or _num(r.get("NMI")) is not None
           for r in rows):
        return "NMI"
    return "Modularity"


# ---------------------------------------------------------------------------
# E1: clean benchmark
# ---------------------------------------------------------------------------

def _e1_table():
    """Group E1 rows by dataset, format metrics and mark the best method."""
    grouped = defaultdict(list)
    for row in _read_csv("e1_clean_benchmark.csv"):
        grouped[row["Dataset"]].append(row)
    out = {}
    for dataset, rows in grouped.items():
        for row in rows:
            for metric in ("ACC", "NMI", "ARI", "AMI", "Modularity"):
                row[metric] = _fmt_pair(row.get(metric + "_mean"),
                                        row.get(metric + "_std"))
            row["ours"] = row.get("Method", "").startswith("DMRA-")
        metric = _metric_for(rows)
        key = metric + "_mean"
        pairs = [(v, i) for i, r in enumerate(rows)
                 if (v := _num(r.get(key))) is not None]
        if pairs:
            rows[max(pairs)[1]]["best"] = True
        out[dataset] = rows
    return out


def _selected_alpha(dataset, e1):
    """Read the alpha used by the stability-selected method in E1."""
    for row in e1.get(dataset, []):
        m = re.search(r"DMRA-stab\(a=([0-9.]+)\)", row.get("Method", ""))
        if m:
            return float(m.group(1))
    return None


def _oracle_alpha(dataset, e1):
    for row in e1.get(dataset, []):
        m = re.search(r"DMRA-oracle\(a=([0-9.]+)\)", row.get("Method", ""))
        if m:
            return float(m.group(1))
    return None


# ---------------------------------------------------------------------------
# E2: alpha sensitivity / stability series
# ---------------------------------------------------------------------------

def _e2_series():
    """Data for a dependency-free stability-vs-alpha chart, read from CSV."""
    grouped = defaultdict(lambda: defaultdict(lambda: {"m": [], "s": []}))
    metric_names = {}
    for row in _read_csv("e2_alpha_sensitivity.csv"):
        ds = row["Dataset"]
        alpha = _num(row.get("Alpha"))
        if alpha is None:
            continue
        grouped[ds][alpha]["m"].append(_num(row.get("Mean")))
        grouped[ds][alpha]["s"].append(_num(row.get("Stability")))
        metric_names.setdefault(ds, row.get("Metric", "Modularity"))

    figures = _figures()
    out = {}
    for ds, by_alpha in grouped.items():
        alphas = sorted(by_alpha)
        figure = f"e2_alpha_sensitivity_{ds}.png"
        out[ds] = {
            "alphas": alphas,
            "stability": [_mean(by_alpha[a]["s"]) for a in alphas],
            "metric": [_mean(by_alpha[a]["m"]) for a in alphas],
            "metric_name": metric_names.get(ds, "Modularity"),
            "figure": figure if figure in figures else None,
        }
    return out


def _e2_table():
    grouped = defaultdict(list)
    for row in _read_csv("e2_alpha_sensitivity.csv"):
        grouped[row["Dataset"]].append(row)
    out = {}
    for ds, rows in grouped.items():
        rows = sorted(rows, key=lambda r: _num(r.get("Alpha")) or 0.0)
        for row in rows:
            row["AlphaLabel"] = _fmt(row.get("Alpha"))
            row["Score"] = _fmt_pair(row.get("Mean"), row.get("Std"))
            row["StabilityLabel"] = _fmt(row.get("Stability"))
        out[ds] = rows
    return out


# ---------------------------------------------------------------------------
# E3: robustness series
# ---------------------------------------------------------------------------

def _e3_series():
    """Mean evaluated metric per config and noise level (over seeds)."""
    grouped = defaultdict(lambda: defaultdict(list))
    details = defaultdict(dict)
    for row in _read_csv("e3_robustness.csv"):
        ds = row["Dataset"]
        cfg = row.get("Config", "")
        frac = _num(row.get("NoiseFrac"))
        if frac is None:
            continue
        pct = int(round(frac * 100))
        metric = row.get("EvaluatedMetric", "Modularity")
        details[ds]["metric"] = metric
        grouped[ds][(cfg, pct)].append(_num(row.get(metric)))
    out = {}
    for ds, series in grouped.items():
        levels = sorted({pct for _, pct in series})
        configs = {}
        for cfg, pct in series:
            configs.setdefault(cfg, {})[pct] = _mean(series[(cfg, pct)])
        out[ds] = {
            "metric": details[ds].get("metric", "Modularity"),
            "levels": levels,
            "configs": configs,
        }
    return out


# ---------------------------------------------------------------------------
# E4: ablation
# ---------------------------------------------------------------------------

def _e4_table():
    grouped = defaultdict(list)
    for row in _read_csv("e4_ablation.csv"):
        grouped[row["Dataset"]].append(row)
    out = {}
    for ds, rows in grouped.items():
        for row in rows:
            for metric in ("ACC", "NMI", "ARI", "AMI", "Modularity"):
                row[metric] = _fmt(row.get(metric))
        out[ds] = rows
    return out


# ---------------------------------------------------------------------------
# Data-driven findings (computed from the CSV files, never hardcoded)
# ---------------------------------------------------------------------------

def _findings(e1, e2, e3, e4):
    result = {}

    # E1
    lines = []
    for ds, rows in e1.items():
        metric = _metric_for(rows)
        key = metric + "_mean"
        pairs = [(v, r.get("Method", "")) for r in rows
                 if (v := _num(r.get(key))) is not None]
        if not pairs:
            continue
        best_val, best_name = max(pairs)
        stab = next((r for r in rows if r.get("Method", "").startswith("DMRA-stab")), None)
        stab_val = _num(stab.get(key)) if stab else None
        if stab_val is not None:
            lines.append(
                f"{ds}: stability-selected alpha reached {metric} {stab_val:.3f}; "
                f"strongest compared method was {best_name} ({best_val:.3f}).")
        else:
            lines.append(
                f"{ds}: strongest compared method was {best_name} ({best_val:.3f}).")
    result["e1"] = lines or ["No benchmark rows are available yet."]

    # E2
    lines = []
    for ds, data in e2.items():
        pairs = [(s, a) for a, s in zip(data["alphas"], data["stability"])
                 if s is not None]
        if not pairs:
            continue
        peak_score, peak_alpha = max(pairs)
        selected = _selected_alpha(ds, e1)
        text = (f"{ds}: stability is highest at alpha = {peak_alpha:.1f} "
                f"(score {peak_score:.3f})")
        if selected is not None:
            text += f"; the selected alpha used in E1 was {selected:.1f}"
        lines.append(text + ".")
    result["e2"] = lines or ["No alpha-sensitivity rows are available yet."]

    # E3
    lines = []
    for ds, data in e3.items():
        configs = data["configs"]
        levels = data["levels"]
        if not levels:
            continue
        first, last = levels[0], levels[-1]
        for cfg in configs:
            if "stability" in cfg.lower():
                a, b = configs[cfg].get(first), configs[cfg].get(last)
                if a is not None and b is not None:
                    lines.append(
                        f"{ds}, {cfg}: {data['metric']} changes from {a:.3f} at "
                        f"{first}% noise to {b:.3f} at {last}% noise.")
    result["e3"] = lines or ["No robustness rows are available yet."]

    # E4
    lines = []
    for ds, rows in e4.items():
        metric = "NMI" if any(_num(r.get("NMI")) is not None for r in rows) else "Modularity"
        by_cfg = defaultdict(list)
        for r in rows:
            by_cfg[r.get("Config", "")].append(_num(r.get(metric)))
        ranked = [(v, cfg) for cfg, vals in by_cfg.items()
                  if (v := _mean(vals)) is not None]
        if ranked:
            best_val, best_cfg = max(ranked)
            lines.append(f"{ds}: best {metric} ({best_val:.3f}) with {best_cfg}.")
    result["e4"] = lines or ["No ablation rows are available yet."]

    return result


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

def _payload():
    summary = _summary()
    e1, e2, e3, e4 = _e1_table(), _e2_series(), _e3_series(), _e4_table()
    e2_table = _e2_table()
    return {
        "summary": summary,
        "team": TEAM,
        "datasets": summary.get("datasets", []),
        "overview": _dataset_overview(),
        "candidates": summary.get("method", {}).get("candidates", []),
        "e1": e1,
        "e2": e2,
        "e2_table": e2_table,
        "e3": e3,
        "e4": e4,
        "findings": _findings(e1, e2, e3, e4),
        "figures": _figures(),
        "csv_files": _csv_files(),
    }


@app.route("/health")
def health():
    """Health probe used by launchers/monitors. Always returns JSON OK."""
    return jsonify({
        "status": "ok",
        "results": os.path.isdir(RESULTS_DIR) and bool(_csv_files()),
        "figures": len(_figures()),
        "generated_at": _summary().get("generated_at"),
    })


@app.route("/favicon.ico")
def favicon():
    return "", 204


@app.route("/v1/models")
def models():
    """Compatibility endpoint used by some local launchers/monitors."""
    return jsonify({
        "object": "list",
        "data": [{
            "id": "sg-dmnmf",
            "object": "model",
            "owned_by": "research",
        }],
    })


@app.route("/")
def index():
    return render_template("index.html", **_payload())


@app.route("/api/results")
def api_results():
    return jsonify(_payload())


@app.route("/figures/<path:filename>")
def figures(filename):
    if filename not in _figures():
        abort(404)
    return send_from_directory(FIGURES_DIR, filename)


@app.route("/results/<path:filename>")
def results(filename):
    allowed = _csv_files() + ["summary.json"]
    if filename not in allowed:
        abort(404)
    return send_from_directory(RESULTS_DIR, filename)


def main(host="127.0.0.1", port=8000, open_browser=True):
    print(f"\n  SG-DMNMF dashboard -> http://{host}:{port}")
    print("  Press Ctrl+C to stop.\n")
    if open_browser:
        try:
            import webbrowser
            webbrowser.open(f"http://{host}:{port}")
        except Exception:
            pass
    app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
