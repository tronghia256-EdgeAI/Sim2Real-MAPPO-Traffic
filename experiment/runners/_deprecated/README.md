# `_deprecated/` — retired runners (kept for provenance)

These scripts are **superseded and must not be used for paper results**. They
report proxy metrics (mean-queue × step_length, broken per-step throughput)
instead of rigorous SUMO `--tripinfo-output` travel/waiting times.

| Script | Replaced by | Why |
|---|---|---|
| `evaluate.py` | `experiment/runners/eval_compare.py` | proxy metrics; multi-method tripinfo harness with CI + significance tests now canonical |
| `compare_traffic_metrics.py` | `experiment/runners/eval_compare.py` | gutted to a deprecation stub (crashed on `env.step_length`, defaulted to the legacy checkpoint) |

Use the canonical harness instead — see NOTES.md §8 "Baseline vs MAPPO Comparison".
