#!/usr/bin/env python
"""
generate_summary_stats.py
=========================
Aggregate the Paper-1 (TSC) experiment artifacts under ``results/paper1_mappo/``
into a single dense Markdown brief (``summary_stats.md``) for drafting Section VI
(Results) of the IEEE T-ITS paper.

What it reads (all written by the existing pipeline — paths verified against the
live repo layout):

    results/paper1_mappo/
      eval_tables/<network>_<ts>/summary.json   <- eval_compare.py  (MAIN result)
                               /table.csv
      noise_robustness/<ts>/summary.json         <- III-E noise sweep
      <campaign_id>/campaign_manifest.json       <- parallel_launcher.py (job status)
      <campaign_id>/models/<job>/<run_id>/{bench.json,run_config.json}
      pilot_<network>/<run_id>/pilot_summary.json

The script is *data-driven*: the metric set, method set and seed counts are taken
from whatever the artifacts actually contain, so it keeps working as more seeds /
methods / metrics (e.g. emissions) appear. Nothing is fabricated — metrics that
the tripinfo harness does not emit (instantaneous queue length, average speed,
speed variance) are reported as unavailable rather than invented.

Grouping (per the campaign design):
    method key  ->  (Algorithm, Observation mode)
    mappo            MAPPO        proxy        (deployable policy — the headline)
    privileged       MAPPO        privileged   (VI-C upper bound)
    ippo             IPPO         proxy        (centralised-training ablation)
    webster/actuated/maxpressure/sotl/fixed    classical baselines

Usage (from project root):
    python scripts/generate_summary_stats.py
    python scripts/generate_summary_stats.py --results-dir results/paper1_mappo \
        --output summary_stats.md --campaign-id campaign_20260618_010101
"""
from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # mirror eval_compare.py's dependency; degrade gracefully if absent
    from scipy.stats import mannwhitneyu as _mannwhitneyu
except Exception:  # pragma: no cover
    _mannwhitneyu = None

ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------- #
# Metric + method metadata (kept aligned with eval_compare.METRIC_KEYS)
# --------------------------------------------------------------------------- #

# Preferred render order; any extra metric found in the data is appended after.
METRIC_ORDER: Tuple[str, ...] = (
    "mean_waiting_time_s",
    "mean_travel_time_s",
    "mean_time_loss_s",
    "p95_waiting_time_s",
    "n_trips",
    "total_co2_mg",
    "total_fuel_mg",
)
METRIC_LABELS: Dict[str, str] = {
    "mean_waiting_time_s": "Waiting (s)",
    "mean_travel_time_s": "Travel time (s)",
    "mean_time_loss_s": "Time loss (s)",
    "p95_waiting_time_s": "P95 waiting (s)",
    "n_trips": "Served trips",
    "total_co2_mg": "CO2 (mg/ep)",
    "total_fuel_mg": "Fuel (mg/ep)",
}
LOWER_IS_BETTER: Dict[str, bool] = {
    "mean_waiting_time_s": True,
    "mean_travel_time_s": True,
    "mean_time_loss_s": True,
    "p95_waiting_time_s": True,
    "n_trips": False,
    "total_co2_mg": True,
    "total_fuel_mg": True,
}
EMISSION_METRICS = ("total_co2_mg", "total_fuel_mg")

# Primary metrics the paper emphasises (used for the significance block).
PRIMARY_METRICS = ("mean_waiting_time_s", "mean_travel_time_s",
                   "mean_time_loss_s", "p95_waiting_time_s")

# Metrics the user listed but tripinfo does not provide -> flagged, not faked.
UNAVAILABLE_METRICS = {
    "queue length": "instantaneous queue is a training-time SUMO signal, not in tripinfo",
    "average speed": "not emitted by eval_compare's tripinfo parser",
    "speed variance (sigma)": "not emitted by eval_compare's tripinfo parser",
}

METHOD_LABELS: Dict[str, str] = {
    "mappo": "MAPPO (proxy)",
    "privileged": "MAPPO (privileged)",
    "ippo": "IPPO (proxy)",
    "webster": "Webster",
    "actuated": "Actuated",
    "maxpressure": "MaxPressure",
    "sotl": "SOTL",
    "fixed": "Fixed-Time",
}
METHOD_GROUP: Dict[str, Tuple[str, str]] = {
    "mappo": ("MAPPO", "proxy"),
    "privileged": ("MAPPO", "privileged"),
    "ippo": ("IPPO", "proxy"),
}
METHOD_ORDER: Tuple[str, ...] = (
    "mappo", "privileged", "ippo",
    "webster", "actuated", "maxpressure", "sotl", "fixed",
)
LEARNED_METHODS = {"mappo", "privileged", "ippo"}
REFERENCE_METHOD = "mappo"        # MAPPO-proxy = the deployable headline policy
SIG_COMPARATOR = "maxpressure"    # the strongest classical baseline (Reviewer Q5)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def load_json(path: Path) -> Optional[dict]:
    """json.load tolerates the NaN/Infinity literals these files contain."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def is_finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


_TS_RE = re.compile(r"(\d{8}_\d{6})")


def _ts_key(p: Path) -> str:
    """Sort key: the trailing YYYYMMDD_HHMMSS stamp, else mtime as a fallback."""
    m = _TS_RE.search(p.name)
    return m.group(1) if m else f"00000000_000000_{p.stat().st_mtime:.0f}"


def rel_to_root(p: Path) -> str:
    """Repo-relative POSIX path when under ROOT, else the absolute path."""
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def metric_label(key: str) -> str:
    return METRIC_LABELS.get(key, key.replace("_", " "))


def method_label(key: str) -> str:
    return METHOD_LABELS.get(key, key)


def fmt_mean_std(metric: str, mean: float, std: float) -> str:
    if not is_finite(mean):
        return "—"
    if metric == "n_trips":
        return f"{mean:.0f} ± {std:.0f}" if is_finite(std) else f"{mean:.0f}"
    if metric in EMISSION_METRICS:  # large mg magnitudes -> thousands-separated ints
        s = f"{std:,.0f}" if is_finite(std) else "—"
        return f"{mean:,.0f} ± {s}"
    s = f"{std:.2f}" if is_finite(std) else "—"
    return f"{mean:.2f} ± {s}"


def fmt_p(p: float) -> str:
    if not is_finite(p):
        return "n/a"
    if p < 1e-3:
        return "<0.001"
    return f"{p:.3f}"


def fmt_num(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}f}" if is_finite(x) else "—"


# --------------------------------------------------------------------------- #
# Statistics (recomputed straight from per-seed value arrays)
# --------------------------------------------------------------------------- #

def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    sp2 = ((a.size - 1) * np.var(a, ddof=1) + (b.size - 1) * np.var(b, ddof=1)) / (a.size + b.size - 2)
    sp = math.sqrt(sp2)
    if sp == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / sp)


def mann_whitney_p(a: Sequence[float], b: Sequence[float]) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2 or _mannwhitneyu is None:
        return float("nan")
    try:
        return float(_mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except Exception:
        return float("nan")


def pct_improvement(ref_mean: float, other_mean: float, lower_is_better: bool) -> float:
    if not (is_finite(ref_mean) and is_finite(other_mean)) or other_mean == 0:
        return float("nan")
    direction = -1.0 if lower_is_better else 1.0
    return direction * (ref_mean - other_mean) / abs(other_mean) * 100.0


def effect_label(d: float) -> str:
    if not is_finite(d):
        return ""
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #

def discover_eval_tables(results_dir: Path) -> Dict[str, dict]:
    """Latest eval_compare summary.json per network (keyed by `scenario`)."""
    out: Dict[str, Tuple[str, dict, Path]] = {}
    for summ in sorted((results_dir / "eval_tables").glob("*/summary.json")):
        data = load_json(summ)
        if not data or "aggregates" not in data:
            continue
        scenario = data.get("scenario", summ.parent.name)
        key = _ts_key(summ.parent)
        if scenario not in out or key > out[scenario][0]:
            out[scenario] = (key, data, summ.parent)
    return {net: {"data": d, "dir": p} for net, (_, d, p) in out.items()}


def discover_noise_sweep(results_dir: Path) -> Optional[dict]:
    summaries = sorted((results_dir / "noise_robustness").glob("*/summary.json"),
                       key=lambda p: _ts_key(p.parent))
    if not summaries:
        return None
    data = load_json(summaries[-1])
    if not data:
        return None
    return {"data": data, "dir": summaries[-1].parent}


def discover_campaign(results_dir: Path, campaign_id: Optional[str]) -> Optional[dict]:
    if campaign_id:
        path = results_dir / campaign_id / "campaign_manifest.json"
        data = load_json(path)
        return {"data": data, "dir": path.parent} if data else None
    manifests = sorted(results_dir.glob("campaign_*/campaign_manifest.json"),
                       key=lambda p: _ts_key(p.parent))
    if not manifests:
        return None
    data = load_json(manifests[-1])
    return {"data": data, "dir": manifests[-1].parent} if data else None


def discover_pilots(results_dir: Path) -> List[dict]:
    out = []
    for summ in sorted(results_dir.glob("pilot_*/*/pilot_summary.json"),
                       key=lambda p: _ts_key(p.parent)):
        data = load_json(summ)
        if data:
            out.append({"data": data, "dir": summ.parent})
    return out


# --------------------------------------------------------------------------- #
# Aggregate-block accessors (summary.json["aggregates"][method][metric])
# --------------------------------------------------------------------------- #

def agg_cell(aggregates: dict, method: str, metric: str) -> dict:
    return aggregates.get(method, {}).get(metric, {}) or {}


def metric_keys_present(aggregates: dict) -> List[str]:
    seen: List[str] = []
    for method in aggregates.values():
        for m in method:
            if m not in seen:
                seen.append(m)
    ordered = [m for m in METRIC_ORDER if m in seen]
    ordered += [m for m in seen if m not in ordered]
    return ordered


def ordered_methods(aggregates: dict) -> List[str]:
    present = list(aggregates)
    out = [m for m in METHOD_ORDER if m in present]
    out += [m for m in present if m not in out]
    return out


def emissions_present(aggregates: dict) -> bool:
    for method in aggregates:
        for em in EMISSION_METRICS:
            if is_finite(agg_cell(aggregates, method, em).get("mean")):
                return True
    return False


# --------------------------------------------------------------------------- #
# Markdown renderers
# --------------------------------------------------------------------------- #

def md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    line = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return "\n".join([line, sep, body])


def render_main_table(aggregates: dict, metrics: Sequence[str],
                      skip_emissions: bool) -> str:
    metrics = [m for m in metrics if not (skip_emissions and m in EMISSION_METRICS)]
    headers = ["Method", "Algo", "Obs", "n"] + [metric_label(m) for m in metrics]
    rows: List[List[str]] = []
    for method in ordered_methods(aggregates):
        algo, obs = METHOD_GROUP.get(method, ("classical", "—"))
        # use n from the first metric that has samples
        n = 0
        for m in metrics:
            n = max(n, int(agg_cell(aggregates, method, m).get("n", 0) or 0))
        row = [f"**{method_label(method)}**", algo, obs, str(n)]
        for m in metrics:
            cell = agg_cell(aggregates, method, m)
            row.append(fmt_mean_std(m, cell.get("mean", float("nan")),
                                    cell.get("std", float("nan"))))
        rows.append(row)
    return md_table(headers, rows)


def render_significance(aggregates: dict, harness_stats: dict,
                        reference: str, comparator: str) -> List[str]:
    lines: List[str] = []
    if reference not in aggregates:
        lines.append(f"_Reference method `{reference}` not present in this eval — "
                     f"significance section skipped._")
        return lines

    ref_n = max((int(agg_cell(aggregates, reference, m).get("n", 0) or 0)
                 for m in PRIMARY_METRICS), default=0)

    # (a) headline: MAPPO-proxy vs MaxPressure on the primary metrics
    if comparator in aggregates:
        lines.append(f"**{method_label(reference)} vs {method_label(comparator)}** "
                     f"(Mann–Whitney U two-sided, Cohen's d; Holm-adjusted p from harness):\n")
        headers = ["Metric", method_label(reference), method_label(comparator),
                   "Δ% (ref better)", "p (recomputed)", "p_holm (harness)", "Cohen's d"]
        rows = []
        for metric in PRIMARY_METRICS:
            ref_cell = agg_cell(aggregates, reference, metric)
            cmp_cell = agg_cell(aggregates, comparator, metric)
            ref_vals = ref_cell.get("values", [])
            cmp_vals = cmp_cell.get("values", [])
            p_re = mann_whitney_p(ref_vals, cmp_vals)
            d = cohens_d(ref_vals, cmp_vals)
            pct = pct_improvement(ref_cell.get("mean", float("nan")),
                                  cmp_cell.get("mean", float("nan")),
                                  LOWER_IS_BETTER.get(metric, True))
            hstat = harness_stats.get(metric, {}).get(comparator, {})
            p_holm = hstat.get("p_holm", float("nan"))
            rows.append([
                metric_label(metric),
                fmt_mean_std(metric, ref_cell.get("mean", float("nan")), ref_cell.get("std", float("nan"))),
                fmt_mean_std(metric, cmp_cell.get("mean", float("nan")), cmp_cell.get("std", float("nan"))),
                (f"{pct:+.1f}%" if is_finite(pct) else "—"),
                fmt_p(p_re),
                fmt_p(p_holm),
                (f"{d:+.2f} ({effect_label(d)})" if is_finite(d) else "—"),
            ])
        lines.append(md_table(headers, rows))
    else:
        lines.append(f"_Comparator `{comparator}` absent from this eval._")

    if ref_n < 2:
        lines.append(
            f"\n> ⚠ **Significance not computable:** `{method_label(reference)}` was "
            f"evaluated from only **n={ref_n}** checkpoint sample(s). Mann–Whitney U "
            f"needs ≥2 per group, so recomputed p-values are `n/a`. Provide one "
            f"trained checkpoint per seed (per-seed `best_model.pt`) so eval_compare "
            f"produces n=5 policy samples."
        )

    # (b) full table: every method vs the reference (harness-adjusted p_holm + d)
    others = [m for m in ordered_methods(aggregates) if m != reference]
    if others:
        lines.append(f"\n**All methods vs {method_label(reference)}** "
                     f"(% = reference improvement; p_holm family-adjusted across baselines):\n")
        headers = ["Metric"] + [method_label(m) for m in others]
        rows = []
        for metric in PRIMARY_METRICS:
            row = [metric_label(metric)]
            for m in others:
                st = harness_stats.get(metric, {}).get(m, {})
                pct = st.get("pct_improvement_ref_vs_method", float("nan"))
                p_holm = st.get("p_holm", float("nan"))
                star = "*" if (is_finite(p_holm) and p_holm < 0.05) else ""
                cell = f"{pct:+.1f}%{star}" if is_finite(pct) else "—"
                row.append(cell)
            rows.append(row)
        lines.append(md_table(headers, rows))
        lines.append("\n_`*` = Holm-adjusted p < 0.05 vs reference._")
    return lines


def render_emissions(aggregates: dict) -> List[str]:
    lines: List[str] = []
    headers = ["Method"] + [metric_label(m) for m in EMISSION_METRICS]
    rows = []
    for method in ordered_methods(aggregates):
        cells = [fmt_mean_std(em, agg_cell(aggregates, method, em).get("mean", float("nan")),
                              agg_cell(aggregates, method, em).get("std", float("nan")))
                 for em in EMISSION_METRICS]
        if any(c != "—" for c in cells):
            rows.append([f"**{method_label(method)}**"] + cells)
    if not rows:
        return []
    lines.append(md_table(headers, rows))
    return lines


def render_noise_sweep(noise: dict) -> Tuple[List[str], List[str]]:
    """Returns (markdown lines, anomaly flags)."""
    data = noise["data"]
    conditions = data.get("conditions", [])
    lines: List[str] = []
    anomalies: List[str] = []

    lines.append(f"- Checkpoint: `{data.get('checkpoint')}`  |  network: "
                 f"`{data.get('network')}`  |  obs_mode: `{data.get('obs_mode')}`  |  "
                 f"calibrated: `{data.get('calibrated')}`")
    seeds = data.get("seeds", [])
    lines.append(f"- Seeds: {seeds} (n={len(seeds)})\n")

    headers = ["Condition", "Travel time (s)", "Waiting (s)", "P95 waiting (s)",
               "Served trips", "n"]
    rows = []
    for c in conditions:
        rows.append([
            f"`{c.get('condition')}`",
            (f"{c.get('mean_travel_time_s'):.2f} ± {c.get('std_travel_time_s', float('nan')):.2f}"
             if is_finite(c.get("mean_travel_time_s")) else "—"),
            fmt_num(c.get("mean_waiting_time_s", float("nan"))),
            fmt_num(c.get("p95_waiting_time_s", float("nan"))),
            f"{c.get('n_trips', float('nan')):.0f}" if is_finite(c.get("n_trips")) else "—",
            str(c.get("n_seeds", "")),
        ])
    lines.append(md_table(headers, rows))

    # Monotonicity check on the scale_* noise ladder (more noise -> worse).
    scale_order = ["scale_0", "scale_0.5", "scale_1", "scale_2"]
    by_name = {c.get("condition"): c for c in conditions}
    ladder = [by_name[s] for s in scale_order if s in by_name]
    for metric, lbl in (("mean_waiting_time_s", "waiting"),
                        ("mean_travel_time_s", "travel time")):
        vals = [(c.get("condition"), c.get(metric)) for c in ladder
                if is_finite(c.get(metric))]
        for (n0, v0), (n1, v1) in zip(vals, vals[1:]):
            if v1 < v0 - 1e-6:  # performance IMPROVED with more noise — non-monotone
                anomalies.append(
                    f"Noise sweep non-monotonicity ({lbl}): `{n1}` ({v1:.1f}s) < "
                    f"`{n0}` ({v0:.1f}s) — more sensing noise produced *better* "
                    f"{lbl}; expected a monotone degradation."
                )
    return lines, anomalies


# --------------------------------------------------------------------------- #
# Campaign / anomaly analysis
# --------------------------------------------------------------------------- #

def _arm_of(job: dict) -> str:
    """Strip the trailing _seed<n> so all seeds of one arm share a key."""
    name = job.get("name", "")
    return re.sub(r"_seed\d+$", "", name)


def analyse_campaign(campaign: Optional[dict]) -> Tuple[List[str], List[str]]:
    """Returns (status markdown lines, anomaly flags)."""
    lines: List[str] = []
    anomalies: List[str] = []
    if not campaign:
        lines.append("_No `campaign_*/campaign_manifest.json` found under the results "
                     "dir — the main-campaign training matrix has not produced a local "
                     "manifest (it may be running remotely). Job-level status checks "
                     "are skipped; eval coverage below is derived from the eval tables._")
        return lines, anomalies

    data = campaign["data"]
    jobs = data.get("jobs", [])
    counts = data.get("counts", {})
    lines.append(f"- Campaign dir: `{campaign['dir'].name}`  |  updated: "
                 f"`{data.get('updated_at')}`  |  total jobs: {len(jobs)}")
    lines.append(f"- Status counts: " + ", ".join(
        f"{k}={v}" for k, v in counts.items()) + "\n")

    # group by arm -> {seed: status}
    arms: Dict[str, Dict[int, str]] = {}
    all_seeds: set = set()
    for j in jobs:
        arm = _arm_of(j)
        arms.setdefault(arm, {})[j.get("seed")] = j.get("status")
        all_seeds.add(j.get("seed"))
    expected_seeds = sorted(s for s in all_seeds if s is not None)

    headers = ["Arm", "Seeds done", "Statuses", "Complete?"]
    rows = []
    for arm in sorted(arms):
        per_seed = arms[arm]
        done = sorted(s for s, st in per_seed.items() if st in ("done", "skipped"))
        statuses = ",".join(f"{s}:{per_seed[s]}" for s in sorted(per_seed))
        complete = set(done) >= set(expected_seeds) and len(expected_seeds) > 0
        rows.append([f"`{arm}`",
                     f"{len(done)}/{len(expected_seeds)}",
                     statuses,
                     "✅" if complete else "⚠"])
        if not complete:
            missing = sorted(set(expected_seeds) - set(done))
            bad = [s for s, st in per_seed.items() if st in ("failed", "killed")]
            detail = []
            if missing:
                detail.append(f"missing/incomplete seeds {missing}")
            if bad:
                detail.append(f"failed seeds {sorted(bad)}")
            anomalies.append(f"Campaign arm `{arm}` incomplete: " + "; ".join(detail))
    lines.append(md_table(headers, rows))

    for j in jobs:
        if j.get("status") in ("failed", "killed"):
            anomalies.append(
                f"Job `{j.get('name')}` {j.get('status')} "
                f"(exit={j.get('exit_code')}, log=`{Path(str(j.get('log_path'))).name}`)."
            )
    return lines, anomalies


def analyse_eval_coverage(eval_tables: Dict[str, dict]) -> List[str]:
    """Cross-network anomaly flags from the eval aggregates themselves."""
    anomalies: List[str] = []
    for net, info in eval_tables.items():
        aggregates = info["data"].get("aggregates", {})
        # per-method seed counts on a primary metric
        n_by_method = {m: int(agg_cell(aggregates, m, "mean_waiting_time_s").get("n", 0) or 0)
                       for m in aggregates}
        ns = [n for n in n_by_method.values() if n > 0]
        if not ns:
            anomalies.append(f"[{net}] eval table present but no method has finite "
                             f"primary-metric samples.")
            continue
        max_n = max(ns)
        for method in ordered_methods(aggregates):
            n = n_by_method.get(method, 0)
            if method in LEARNED_METHODS and n < 2:
                anomalies.append(
                    f"[{net}] `{method_label(method)}` evaluated from n={n} checkpoint "
                    f"sample(s) — point estimate only, no per-seed variance/significance."
                )
            elif 0 < n < max_n:
                anomalies.append(
                    f"[{net}] `{method_label(method)}` has n={n} seeds vs {max_n} for "
                    f"the fullest method — uneven seed coverage."
                )
    return anomalies


# --------------------------------------------------------------------------- #
# Document assembly
# --------------------------------------------------------------------------- #

def build_document(results_dir: Path, campaign_id: Optional[str]) -> str:
    eval_tables = discover_eval_tables(results_dir)
    noise = discover_noise_sweep(results_dir)
    campaign = discover_campaign(results_dir, campaign_id)
    pilots = discover_pilots(results_dir)

    out: List[str] = []
    out.append("# Paper-1 (TSC) — Section VI Results Summary\n")
    out.append(f"_Auto-generated by `scripts/generate_summary_stats.py` on "
               f"{datetime.now().isoformat(timespec='seconds')}._\n")
    out.append(f"_Results root: `{results_dir.as_posix()}`._\n")

    # ---- Data sources ----------------------------------------------------- #
    out.append("## Data Sources\n")
    src_rows = []
    for net, info in sorted(eval_tables.items()):
        src_rows.append([f"eval table ({net})", f"`{rel_to_root(info['dir'])}`"])
    if noise:
        src_rows.append(["noise sweep", f"`{rel_to_root(noise['dir'])}`"])
    if campaign:
        src_rows.append(["campaign manifest", f"`{rel_to_root(campaign['dir'])}`"])
    if pilots:
        src_rows.append([f"pilots ({len(pilots)})",
                         f"`{rel_to_root(pilots[-1]['dir'])}` (latest)"])
    if src_rows:
        out.append(md_table(["Artifact", "Path"], src_rows) + "\n")
    else:
        out.append("> ⚠ No recognised artifacts found under the results dir.\n")

    note = ", ".join(f"**{k}** ({v})" for k, v in UNAVAILABLE_METRICS.items())
    out.append(f"> **Metric availability note:** the tripinfo eval harness "
               f"(`eval_compare.py`) reports travel time, waiting time, time loss, "
               f"P95 waiting, served trips, and (with `--emissions`) CO2/fuel. "
               f"It does **not** emit {note}. Source those from training logs if the "
               f"paper requires them.\n")

    all_anomalies: List[str] = []

    # ---- Stage 1: Main campaign ------------------------------------------- #
    out.append("## Stage 1: Main Campaign Results\n")
    if not eval_tables:
        out.append("> ⚠ No eval tables found under `eval_tables/` — run "
                   "`eval_compare.py` (or `build_paper_artifacts.py`) once the "
                   "training campaign produces per-seed checkpoints.\n")
    for net in sorted(eval_tables):
        info = eval_tables[net]
        data = info["data"]
        aggregates = data.get("aggregates", {})
        harness_stats = data.get("stats", {})
        metrics = metric_keys_present(aggregates)
        seeds = data.get("seeds", [])
        skip_em = not emissions_present(aggregates)

        out.append(f"### Network: {net}\n")
        out.append(f"- Split: `{data.get('split')}`  |  eval seeds: {seeds} "
                   f"(n={len(seeds)})  |  max_steps: {data.get('max_steps')}  |  "
                   f"green ∈ [{data.get('min_green_time')}, {data.get('max_green_time')}] s  |  "
                   f"reference: `{data.get('reference')}`\n")
        out.append("**Aggregate metrics (mean ± std across seeds):**\n")
        out.append(render_main_table(aggregates, metrics, skip_emissions=skip_em) + "\n")

        out.append("**Statistical significance (Reviewer Q5):**\n")
        out.extend(s + "\n" for s in render_significance(
            aggregates, harness_stats, REFERENCE_METHOD, SIG_COMPARATOR))

        if not skip_em:
            out.append("\n**Emissions (post-hoc, `--emissions`):**\n")
            out.extend(s + "\n" for s in render_emissions(aggregates))
        else:
            out.append("\n_Emissions not logged for this run (re-run eval_compare "
                       "with `--emissions` for TABLE-10)._\n")

    all_anomalies.extend(analyse_eval_coverage(eval_tables))

    # ---- Stage: Noise robustness ------------------------------------------ #
    out.append("## Stage 2: Sensing-Noise Robustness (III-E / FIG-6)\n")
    if noise:
        noise_lines, noise_anoms = render_noise_sweep(noise)
        out.extend(l + "\n" for l in noise_lines)
        all_anomalies.extend(noise_anoms)
    else:
        out.append("> No noise sweep found under `noise_robustness/`.\n")

    # ---- Pilot context (optional) ----------------------------------------- #
    if pilots:
        out.append("## Pilot Benchmarks (single-seed context)\n")
        headers = ["Network", "Algo", "Seed", "Steps", "SPS", "Travel (s)",
                   "Waiting (s)", "P95 (s)", "Trips"]
        rows = []
        for p in pilots:
            d = p["data"]
            pl, be, ev = d.get("pilot", {}), d.get("bench", {}), d.get("eval_heldout", {})
            rows.append([
                str(pl.get("network")), str(pl.get("algo")), str(pl.get("seed")),
                str(pl.get("total_timesteps")), fmt_num(be.get("mean_steps_per_second", float("nan"))),
                fmt_num(ev.get("mean_travel_time_s", float("nan"))),
                fmt_num(ev.get("mean_waiting_time_s", float("nan"))),
                fmt_num(ev.get("p95_waiting_time_s", float("nan"))),
                f"{ev.get('n_trips', float('nan')):.0f}" if is_finite(ev.get("n_trips")) else "—",
            ])
        out.append(md_table(headers, rows) + "\n")

    # ---- Campaign status -------------------------------------------------- #
    out.append("## Campaign / Training Status\n")
    camp_lines, camp_anoms = analyse_campaign(campaign)
    out.extend(l + "\n" for l in camp_lines)
    all_anomalies.extend(camp_anoms)

    # ---- Anomaly detection ------------------------------------------------ #
    out.append("## Anomaly Detection\n")
    if all_anomalies:
        out.append(f"**{len(all_anomalies)} flag(s) raised:**\n")
        for a in all_anomalies:
            out.append(f"- ⚠ {a}")
        out.append("")
    else:
        out.append("✅ No anomalies detected: all campaign arms complete, learned "
                   "methods have multi-seed coverage, and the noise sweep is monotone.\n")

    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", type=str,
                   default=str(ROOT / "results" / "paper1_mappo"),
                   help="root of the Paper-1 result artifacts")
    p.add_argument("--output", type=str, default=str(ROOT / "summary_stats.md"),
                   help="Markdown file to write")
    p.add_argument("--campaign-id", type=str, default=None,
                   help="specific campaign dir under results-dir (default: latest)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise SystemExit(f"[error] results dir not found: {results_dir}")

    doc = build_document(results_dir, args.campaign_id)
    out_path = Path(args.output)
    out_path.write_text(doc, encoding="utf-8")

    n_lines = doc.count("\n")
    print(f"[ok] wrote {out_path}  ({n_lines} lines, {len(doc)} chars)")
    n_flags = doc.count("- ⚠ ")
    if n_flags:
        print(f"[ok] anomaly flags: {n_flags} (see the Anomaly Detection section)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
