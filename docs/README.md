# Documentation

Technical reference for **Paper 1 — Camera-Observable MAPPO for Traffic Signal Control**.
The manuscript itself (LaTeX) is maintained on Overleaf; this folder holds the engineering
references behind it.

| Document | What it covers |
|---|---|
| [`state.md`](state.md) | The 26-dimensional observation vector — every feature, its SUMO (training) source, its YOLO+ByteTrack (deployment) proxy, and the train/deploy alignment analysis. |
| [`reward.md`](reward.md) | The six-term PRESSLIGHT-extended reward (revision 1.2.0): equation, per-term derivation, weights, and the sim-to-real alignment table. |
| [`reproduction.md`](reproduction.md) | Turnkey path from a finished training campaign to paper-ready tables and figures. |

The full MDP formulation, architecture, and evaluation protocol are documented in the
manuscript; the runnable commands live in the top-level [`README`](../README.md#-usage).
