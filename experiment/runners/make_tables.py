from __future__ import annotations

"""
make_tables.py
==============
Turn eval_compare.py summary.json files into paper-ready LaTeX (booktabs) tables.

Emits, per scenario summary:
  * a main comparison table (TABLE-4 / TABLE-5): methods x metrics, mean ± 95%CI,
    best per metric bolded, † marking p_holm < 0.05 vs the reference method;
  * an information-restriction-cost table (TABLE-6) whenever both the proxy
    reference (default 'mappo') and the 'privileged' arm are present — the
    relative degradation of the deployable proxy vs the simulator-privileged
    upper bound, per metric.

LaTeX deps: \\usepackage{booktabs}. Numbers come straight from the harness, so
re-running eval_compare and then this script keeps every table cell traceable.

Usage (from project root):
    # explicit summaries
    python experiment/runners/make_tables.py \
        --summaries results/paper1_mappo/eval_tables/n2_corridor_*/summary.json \
                    results/paper1_mappo/eval_tables/n3_grid_*/summary.json

    # auto-pick the newest summary per network under eval_tables/
    python experiment/runners/make_tables.py --auto
    # -> figures/tables/<scenario>_main.tex, <scenario>_restriction_cost.tex
"""

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR_DEFAULT = _ROOT / "figures" / "tables"

METRIC_KEYS = (
    "mean_travel_time_s",
    "mean_waiting_time_s",
    "mean_time_loss_s",
    "p95_waiting_time_s",
    "n_trips",
)
LOWER_IS_BETTER = {
    "mean_travel_time_s": True,
    "mean_waiting_time_s": True,
    "mean_time_loss_s": True,
    "p95_waiting_time_s": True,
    "n_trips": False,
}
METRIC_HEADERS = {
    "mean_travel_time_s": "Travel (s)",
    "mean_waiting_time_s": "Waiting (s)",
    "mean_time_loss_s": "Time loss (s)",
    "p95_waiting_time_s": "P95 wait (s)",
    "n_trips": "Served",
}
METHOD_NAMES = {
    "mappo": "MAPPO (proxy)",
    "ippo": "IPPO",
    "privileged": "MAPPO (priv.)",
    "webster": "Webster",
    "actuated": "Actuated",
    "maxpressure": "Max-pressure",
    "sotl": "SOTL",
    "fixed": "Fixed-time",
}


def _tex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def _best_method(aggregates: Dict[str, dict], metric: str) -> Optional[str]:
    lib = LOWER_IS_BETTER[metric]
    best, best_val = None, None
    for method, by_metric in aggregates.items():
        v = by_metric.get(metric, {}).get("mean")
        if v is None or v != v:  # nan
            continue
        if best_val is None or (v < best_val if lib else v > best_val):
            best, best_val = method, v
    return best


def main_table_tex(summary: dict, scenario: str) -> str:
    aggregates = summary["aggregates"]
    stats = summary.get("stats", {})
    ref = summary.get("reference", "")
    methods = list(aggregates)

    col_fmt = "l" + "r" * len(METRIC_KEYS)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{Controller comparison on the {_tex_escape(scenario)} scenario "
        rf"(mean $\pm$ 95\% CI; \textbf{{best}} per metric; "
        rf"$\dagger$: $p_{{\mathrm{{Holm}}}}<0.05$ vs {_tex_escape(METHOD_NAMES.get(ref, ref))}).}}",
        rf"\label{{tab:cmp_{scenario}}}",
        rf"\begin{{tabular}}{{{col_fmt}}}",
        r"\toprule",
        "Method & " + " & ".join(METRIC_HEADERS[m] for m in METRIC_KEYS) + r" \\",
        r"\midrule",
    ]

    best_per_metric = {m: _best_method(aggregates, m) for m in METRIC_KEYS}

    for method in methods:
        cells: List[str] = [_tex_escape(METHOD_NAMES.get(method, method))]
        for metric in METRIC_KEYS:
            a = aggregates[method].get(metric, {})
            mean = a.get("mean", float("nan"))
            ci = a.get("ci95", 0.0)
            txt = f"{mean:.1f}$\\pm${ci:.1f}" if mean == mean else "--"
            if method != ref:
                p = stats.get(metric, {}).get(method, {}).get("p_holm", float("nan"))
                if p == p and p < 0.05:
                    txt += r"$\dagger$"
            if best_per_metric.get(metric) == method:
                txt = rf"\textbf{{{txt}}}"
            cells.append(txt)
        lines.append(" & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


def restriction_cost_table_tex(summary: dict, scenario: str,
                               proxy: str = "mappo", priv: str = "privileged") -> Optional[str]:
    aggregates = summary["aggregates"]
    if proxy not in aggregates or priv not in aggregates:
        return None

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Information-restriction cost on {_tex_escape(scenario)}: "
        rf"deployable proxy vs.\ simulator-privileged upper bound.}}",
        rf"\label{{tab:restriction_{scenario}}}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Metric & Privileged & Proxy & Cost \\",
        r"\midrule",
    ]
    for metric in METRIC_KEYS:
        pv = aggregates[priv].get(metric, {}).get("mean", float("nan"))
        px = aggregates[proxy].get(metric, {}).get("mean", float("nan"))
        if pv != pv or px != px or pv == 0:
            cost = float("nan")
        else:
            # cost = relative degradation of proxy vs privileged upper bound
            direction = 1.0 if LOWER_IS_BETTER[metric] else -1.0
            cost = direction * (px - pv) / abs(pv) * 100.0
        cost_txt = f"{cost:+.1f}\\%" if cost == cost else "--"
        lines.append(
            f"{METRIC_HEADERS[metric]} & {pv:.1f} & {px:.1f} & {cost_txt} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def emissions_table_tex(summary: dict, scenario: str) -> Optional[str]:
    """TABLE-10: post-hoc CO2/fuel per episode (only if --emissions was run)."""
    aggregates = summary["aggregates"]
    keys = [("total_co2_mg", "CO2 (mg/ep)"), ("total_fuel_mg", "Fuel (mg/ep)")]
    # skip entirely if no method has finite emission data
    has = any(
        aggregates[m].get(k, {}).get("mean", float("nan")) == aggregates[m].get(k, {}).get("mean", float("nan"))
        for m in aggregates for k, _ in keys
    )
    if not has:
        return None
    lines = [
        r"\begin{table}[t]", r"\centering",
        rf"\caption{{Post-hoc emissions on {_tex_escape(scenario)} "
        rf"(mean $\pm$ 95\% CI; never used in the reward).}}",
        rf"\label{{tab:emissions_{scenario}}}",
        r"\begin{tabular}{lrr}", r"\toprule",
        "Method & " + " & ".join(lbl for _, lbl in keys) + r" \\", r"\midrule",
    ]
    for method in aggregates:
        cells = [_tex_escape(METHOD_NAMES.get(method, method))]
        for k, _ in keys:
            a = aggregates[method].get(k, {})
            mean = a.get("mean", float("nan"))
            cells.append(f"{mean:.0f}$\\pm${a.get('ci95', 0.0):.0f}" if mean == mean else "--")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def discover_auto() -> List[Path]:
    """Newest summary.json per network under results/paper1_mappo/eval_tables/."""
    base = _ROOT / "results" / "paper1_mappo" / "eval_tables"
    if not base.exists():
        return []
    by_net: Dict[str, Path] = {}
    for summ in sorted(base.glob("*/summary.json")):
        scen = json.loads(summ.read_text(encoding="utf-8")).get("scenario", summ.parent.name)
        # parent dir is <scenario>_<ts>; keep the lexicographically last (newest ts)
        if scen not in by_net or summ.parent.name > by_net[scen].parent.name:
            by_net[scen] = summ
    return list(by_net.values())


def main() -> None:
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summaries", nargs="*", default=None,
                   help="summary.json paths (globs allowed)")
    p.add_argument("--auto", action="store_true",
                   help="auto-pick newest summary per network under eval_tables/")
    p.add_argument("--out-dir", type=str, default=str(OUT_DIR_DEFAULT))
    args = p.parse_args()

    paths: List[Path] = []
    if args.auto:
        paths = discover_auto()
    if args.summaries:
        for pat in args.summaries:
            paths.extend(Path(x) for x in glob.glob(pat))
    paths = [p for p in dict.fromkeys(paths) if p.exists()]
    if not paths:
        sys.exit("[make_tables] no summary.json found (use --auto or --summaries)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for sp in paths:
        summary = json.loads(sp.read_text(encoding="utf-8"))
        scenario = summary.get("scenario", sp.parent.name)
        main_tex = main_table_tex(summary, scenario)
        (out_dir / f"{scenario}_main.tex").write_text(main_tex, encoding="utf-8")
        print(f"[make_tables] wrote {out_dir / (scenario + '_main.tex')}")

        rc = restriction_cost_table_tex(summary, scenario)
        if rc:
            (out_dir / f"{scenario}_restriction_cost.tex").write_text(rc, encoding="utf-8")
            print(f"[make_tables] wrote {out_dir / (scenario + '_restriction_cost.tex')}")
        else:
            print(f"[make_tables] {scenario}: no proxy+privileged pair → skip restriction-cost")

        em = emissions_table_tex(summary, scenario)
        if em:
            (out_dir / f"{scenario}_emissions.tex").write_text(em, encoding="utf-8")
            print(f"[make_tables] wrote {out_dir / (scenario + '_emissions.tex')}")


if __name__ == "__main__":
    main()
