from __future__ import annotations

"""
DEPRECATED — do not use for paper results.
==========================================

This quick MAPPO-vs-fixed visualiser is superseded by

    experiment/runners/eval_compare.py

Reasons it was retired (see the code review, bugs P0-2 / P1-3):
  * it crashed on ``env.step_length`` (no such attribute — the env exposes
    ``config.sim.step_length``);
  * it defaulted to the LEGACY checkpoint ``20260418_215140`` which NOTES.md
    forbids for paper results (pre-1.1.0 obs schema, inert pressure reward);
  * its metrics were PROXIES (``avg_waiting_proxy`` = mean queue × step_length,
    and a broken per-step ``getArrivedNumber`` "throughput"), not the rigorous
    SUMO ``--tripinfo-output`` travel/waiting times used everywhere else.

Use the unified, tripinfo-based, multi-seed harness instead — it produces the
paper comparison tables (TABLE-4/5) with 95% CI and significance tests across
ALL methods (MAPPO / IPPO / privileged / Webster / actuated / max-pressure /
SOTL / fixed-time):

    python experiment/runners/eval_compare.py \
        --network n3_grid \
        --checkpoint models/mappo/<run_id>/best_model.pt \
        --methods mappo webster actuated maxpressure sotl fixed \
        --seeds 42 123 456 789 1337
"""

import sys

_MSG = __doc__


def main() -> None:
    print(_MSG)
    print(
        "\n[deprecated] This script intentionally does nothing. "
        "Run experiment/runners/eval_compare.py instead."
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
