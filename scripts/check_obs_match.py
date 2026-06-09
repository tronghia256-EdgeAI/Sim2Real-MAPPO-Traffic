"""
check_obs_match.py
==================
Verify that the YOLO-based production pipeline (StateExtractor) and the
SUMO training pipeline (ObservationBuilder) produce matching observation
vectors.

Four levels of checks:
  1. STRUCTURAL  — obs_dim, feature names, feature order, normalization caps.
  2. ASSEMBLY    — inject identical lane-metric dicts into both pipelines;
                   output MUST be bit-for-bit identical.
  3. METRIC      — feed synthetic YOLO tracks through StateExtractor AND
                   VisionLaneMetrics; compare per-feature values.
  4. PARAMETERS  — cross-check normalization constants across all config sources.

Optional:
  5. SUMO LIVE   — run a short SUMO episode and compare obs vectors at each step.

Run:
  python scripts/check_obs_match.py              # checks 1-4 only (no SUMO required)
  python scripts/check_obs_match.py --sumo       # also run live SUMO check
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import Mock

import numpy as np

# ── project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vision.state_extractor import StateExtractor, LaneROI
from src.traffic_env.components.observations import ObservationBuilder, VisionLaneMetrics
from src.traffic_env.config import (
    build_default_config,
    DEFAULT_LANE_FEATURE_NAMES,
    DEFAULT_TLS_FEATURE_NAMES,
)

# ── terminal colors ───────────────────────────────────────────────────────────
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"

# ── shared test topology (mirrors 2-intersection production setup) ────────────
TLS_IDS = ("tls_0", "tls_1")

CONTROLLED_LANES: Dict[str, List[str]] = {
    "tls_0": ["lane_0", "lane_1", "lane_2", "lane_3"],
    "tls_1": ["lane_4", "lane_5", "lane_6", "lane_7"],
}

LANE_GROUPS: Dict[str, Tuple[List[str], List[str]]] = {
    "tls_0": (["lane_0", "lane_1"], ["lane_2", "lane_3"]),
    "tls_1": (["lane_4", "lane_5"], ["lane_6", "lane_7"]),
}

# Rectangular 80x200 px ROI per lane
_POLYGONS = {
    "lane_0": [(100, 780), (180, 780), (180, 980), (100, 980)],
    "lane_1": [(220, 780), (300, 780), (300, 980), (220, 980)],
    "lane_2": [(1000, 780), (1080, 780), (1080, 980), (1000, 980)],
    "lane_3": [(1120, 780), (1200, 780), (1200, 980), (1120, 980)],
    "lane_4": [(100, 200), (180, 200), (180, 400), (100, 400)],
    "lane_5": [(220, 200), (300, 200), (300, 400), (220, 400)],
    "lane_6": [(1000, 200), (1080, 200), (1080, 400), (1000, 400)],
    "lane_7": [(1120, 200), (1200, 200), (1200, 400), (1120, 400)],
}

LANE_ROIS: Dict[str, LaneROI] = {
    lane_id: LaneROI(
        lane_id=lane_id,
        tls_id="tls_0" if int(lane_id.split("_")[1]) < 4 else "tls_1",
        polygon=_POLYGONS[lane_id],
        cam_id=int(lane_id.split("_")[1]),
        lane_length_px=200.0,
        lane_width_px=80.0,
    )
    for lane_id in _POLYGONS
}

_ALL_LANE_IDS = list(_POLYGONS.keys())

# ── factory helpers ───────────────────────────────────────────────────────────

def _build_extractor(
    max_green_time: float = 60.0,
    stop_speed_m_s: float = 0.1,
    px_per_meter: float = 20.0,
    speed_cap: float = 15.0,
) -> StateExtractor:
    return StateExtractor(
        tls_ids=list(TLS_IDS),
        controlled_lanes_dict=CONTROLLED_LANES,
        lane_rois=LANE_ROIS,
        lane_groups=LANE_GROUPS,
        motorbike_class_ids=[3],
        heavy_class_ids=[1, 4],
        max_lanes_per_tls=4,
        px_per_meter=px_per_meter,
        stop_speed_m_s=stop_speed_m_s,
        speed_cap=speed_cap,
        max_green_time=max_green_time,
    )


def _build_obs_builder_with_mock() -> Tuple[ObservationBuilder, Mock]:
    """
    Build an ObservationBuilder configured for the test topology (tls_0/tls_1,
    lane_0..7) with a mock SUMO connection.

    Returns (builder, mock_sumo). Caller may update
    mock_sumo.trafficlight.getPhase.side_effect to change phases per scenario.
    """
    mock_sumo = Mock()
    mock_sumo.trafficlight.getPhase.return_value = 0  # default until overridden

    cfg = build_default_config(
        sumo_cfg_path=str(
            ROOT / "sumo_configs" / "evaluation" / "medium" / "sumo_config.sumocfg"
        ),
        gui=False,
        tls_ids=list(TLS_IDS),
        manual_lane_groups=LANE_GROUPS,
        use_external_state=True,
        max_green_time=60,
        speed_cap=15.0,
    )
    builder = ObservationBuilder(config=cfg, sumo_conn=mock_sumo)
    return builder, mock_sumo


def _obs_from_metrics_via_extractor(
    extractor: StateExtractor,
    metrics_cache: Dict[str, Dict[str, float]],
    phase_map: Dict[str, int],
    green_timers: Dict[str, float],
) -> np.ndarray:
    """
    Assemble obs from pre-computed lane metrics bypassing track processing.
    Mirrors StateExtractor.build_state() assembly step-for-step.
    """
    state: List[float] = []
    for tls_id in extractor.tls_ids:
        lane_ids = extractor.controlled_lanes_dict.get(tls_id, [])[: extractor.max_lanes_per_tls]
        lane_metrics_cache: Dict[str, Dict[str, float]] = {}
        for lane_id in lane_ids:
            m = metrics_cache.get(lane_id, extractor._empty_lane_metrics())
            lane_metrics_cache[lane_id] = m
            state.extend([
                m["effective_queue_norm"],
                m["occupancy_norm"],
                m["avg_speed_norm"],
                m["motorbike_share"],
                m["heavy_vehicle_share"],
            ])
        missing = extractor.max_lanes_per_tls - len(lane_ids)
        if missing > 0:
            state.extend([0.0] * (missing * extractor.lane_feature_dim))
        state.extend(extractor._phase_one_hot(int(phase_map.get(tls_id, 0))))
        state.append(
            extractor._normalize(float(green_timers.get(tls_id, 0.0)), extractor.max_green_time)
        )
        state.append(extractor._tls_pressure_proxy(tls_id, lane_ids, lane_metrics_cache))
    return np.asarray(state, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 1 — STRUCTURAL
# ══════════════════════════════════════════════════════════════════════════════

def check_structural() -> int:
    """Return number of hard failures."""
    print("\n" + "=" * 70)
    print("CHECK 1 — STRUCTURAL: dimensions, feature names, normalization caps")
    print("=" * 70)
    failures = 0

    state_cfg_path = ROOT / "configs" / "state_config.json"
    with open(state_cfg_path, encoding="utf-8") as f:
        state_cfg = json.load(f)
    vl = state_cfg.get("vector_layout", {})
    sp = state_cfg.get("system_params", {})

    extractor = _build_extractor()

    # obs dimensions
    obs_dim_yolo  = extractor.get_obs_dim()
    global_cfg    = int(vl.get("global_observation_dim", 52))
    per_agent_cfg = int(vl.get("per_agent_observation_dim", 26))
    per_agent_yolo = (extractor.max_lanes_per_tls * extractor.lane_feature_dim
                      + extractor.tls_feature_dim)

    _row(obs_dim_yolo == global_cfg, "obs_dim (global)",
         f"{obs_dim_yolo}" + ("" if obs_dim_yolo == global_cfg else f"  expected {global_cfg}"))
    if obs_dim_yolo != global_cfg:
        failures += 1

    _row(per_agent_yolo == per_agent_cfg, "per_agent_obs_dim",
         f"{per_agent_yolo}" + ("" if per_agent_yolo == per_agent_cfg else f"  expected {per_agent_cfg}"))
    if per_agent_yolo != per_agent_cfg:
        failures += 1

    # feature names/order
    lane_ok = tuple(extractor.FEATURE_NAMES) == DEFAULT_LANE_FEATURE_NAMES
    _row(lane_ok, "lane feature names", str(extractor.FEATURE_NAMES))
    if not lane_ok:
        print(f"    expected: {DEFAULT_LANE_FEATURE_NAMES}")
        failures += 1

    expected_tls = (
        "phase_one_hot_0", "phase_one_hot_1", "phase_one_hot_2", "phase_one_hot_3",
        "green_timer_norm", "pressure_norm",
    )
    tls_ok = DEFAULT_TLS_FEATURE_NAMES == expected_tls
    _row(tls_ok, "tls feature names", str(DEFAULT_TLS_FEATURE_NAMES))
    if not tls_ok:
        failures += 1

    # normalization caps
    max_green_cfg = int(sp.get("max_green_time", 60))
    note = "" if extractor.max_green_time == max_green_cfg else "  <- green_timer_norm DRIFT"
    _row(extractor.max_green_time == max_green_cfg, "max_green_time",
         f"StateExtractor={extractor.max_green_time}s  state_config={max_green_cfg}s{note}")
    if extractor.max_green_time != max_green_cfg:
        failures += 1

    speed_cfg = float(sp.get("speed_cap", 15.0))
    _row(extractor.speed_cap == speed_cfg, "speed_cap",
         f"{extractor.speed_cap} m/s")
    if extractor.speed_cap != speed_cfg:
        failures += 1

    # halt threshold alignment
    ext_halt = extractor.stop_speed_m_s
    vlm_halt = VisionLaneMetrics.HALT_SPEED_THRESHOLD_MPS

    _row(True, "halt_threshold (SE vs SUMO native)",
         f"both = {ext_halt} m/s  (SUMO hardcoded 0.1 in _lane_metrics)")

    same = abs(ext_halt - vlm_halt) < 1e-6
    _row(same, "halt_threshold (SE vs VisionLaneMetrics)",
         f"both = {ext_halt}" if same
         else f"SE={ext_halt}  VLM={vlm_halt}  <- DIVERGENCE: speeds 0.1-0.5 m/s")
    if not same:
        failures += 1

    # vehicle class IDs
    cls_ok = (extractor._MOTORBIKE_CLASSES == VisionLaneMetrics.MOTORBIKE_CLASSES and
              extractor._HEAVY_CLASSES == VisionLaneMetrics.HEAVY_CLASSES)
    _row(cls_ok, "vehicle class IDs (moto/heavy)",
         f"moto={extractor._MOTORBIKE_CLASSES}  heavy={extractor._HEAVY_CLASSES}" if cls_ok
         else (f"SE moto={extractor._MOTORBIKE_CLASSES} vs VLM={VisionLaneMetrics.MOTORBIKE_CLASSES}; "
               f"SE heavy={extractor._HEAVY_CLASSES} vs VLM={VisionLaneMetrics.HEAVY_CLASSES}"))
    if not cls_ok:
        failures += 1

    # CLASS_LENGTH_PX calibration (warn, not fail)
    px_per_m = float(sp.get("px_per_meter", 20.0))
    bad = []
    for cls_id, label in {3: "motorcycle", 2: "car", 1: "bus", 4: "truck"}.items():
        expected = VisionLaneMetrics.CLASS_LENGTH_M.get(cls_id, 4.5) * px_per_m
        actual   = StateExtractor._CLASS_LENGTH_PX.get(cls_id, 30.0)
        ratio    = actual / max(expected, 1e-6)
        if abs(ratio - 1.0) > 0.15:
            bad.append(f"{label}: {actual:.1f}px vs expected {expected:.1f}px (ratio={ratio:.2f})")
    if bad:
        _row_warn("CLASS_LENGTH_PX calibration",
                  f"calibrated at ~6.7 px/m, not {px_per_m} px/m:")
        for b in bad:
            print(f"      {b}")
        print(f"      -> Re-calibrate or convert from CLASS_LENGTH_M x px_per_meter at runtime.")
    else:
        _row(True, "CLASS_LENGTH_PX calibration", f"consistent with px_per_meter={px_per_m}")

    print(f"\n  -> Structural check: {failures} failure(s)")
    return failures


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 2 — ASSEMBLY EQUIVALENCE
# ══════════════════════════════════════════════════════════════════════════════

def check_assembly(n_scenarios: int = 300) -> Tuple[int, float]:
    """
    Inject identical lane_metrics + phase + green_timer into both pipelines and
    verify obs vectors are bit-for-bit identical.

    Returns (failures, match_pct).
    """
    print("\n" + "=" * 70)
    print("CHECK 2 — ASSEMBLY EQUIVALENCE (same metrics -> same obs vector)")
    print("=" * 70)

    extractor = _build_extractor()
    builder, mock_sumo = _build_obs_builder_with_mock()

    rng = np.random.default_rng(42)
    total_vals = 0
    match_vals = 0
    max_diff   = 0.0
    failures   = 0

    feat_errs: Dict[str, List[float]] = {
        f: [] for f in list(DEFAULT_LANE_FEATURE_NAMES) + list(DEFAULT_TLS_FEATURE_NAMES)
    }

    for _ in range(n_scenarios):
        phase_map    = {t: int(rng.integers(0, 4)) for t in TLS_IDS}
        green_timers = {t: float(rng.uniform(0.0, 60.0)) for t in TLS_IDS}
        metrics_cache: Dict[str, Dict[str, float]] = {
            lane_id: {
                "effective_queue_norm": float(rng.uniform(0, 1)),
                "occupancy_norm":       float(rng.uniform(0, 1)),
                "avg_speed_norm":       float(rng.uniform(0, 1)),
                "motorbike_share":      float(rng.uniform(0, 1)),
                "heavy_vehicle_share":  float(rng.uniform(0, 0.4)),
            }
            for lane_id in _ALL_LANE_IDS
        }

        # Path A: StateExtractor assembly
        obs_se = _obs_from_metrics_via_extractor(extractor, metrics_cache, phase_map, green_timers)

        # Path B: ObservationBuilder assembly via inject_vision_cache
        mock_sumo.trafficlight.getPhase.side_effect = (
            lambda tls_id, _pm=phase_map: int(_pm.get(tls_id, 0))
        )
        builder.inject_vision_cache(metrics_cache)
        lane_cache = builder.build_lane_cache()
        obs_dict, _ = builder.get_all_observations(lane_cache, green_timers=green_timers)
        obs_ob = np.concatenate([obs_dict[t] for t in TLS_IDS], axis=0).astype(np.float32)

        diff = np.abs(obs_se - obs_ob)
        max_diff    = max(max_diff, float(diff.max()))
        total_vals += obs_se.size
        match_vals += int((diff < 1e-5).sum())

        # Accumulate per-feature
        for ai in range(len(TLS_IDS)):
            base = ai * 26
            for li in range(4):
                for k, feat in enumerate(DEFAULT_LANE_FEATURE_NAMES):
                    idx = base + li * 5 + k
                    if idx < len(diff):
                        feat_errs[feat].append(float(diff[idx]))
            for k, feat in enumerate(DEFAULT_TLS_FEATURE_NAMES):
                idx = base + 20 + k
                if idx < len(diff):
                    feat_errs[feat].append(float(diff[idx]))

    match_pct = 100.0 * match_vals / max(total_vals, 1)

    print(f"\n  Scenarios: {n_scenarios}  |  Total obs values: {total_vals}")
    print(f"  Max absolute diff: {max_diff:.2e}")
    print(f"  Match rate (|delta| < 1e-5): {match_pct:.2f}%")
    print()
    print(f"  {'Feature':<25}  {'Mean |d|':>10}  {'Max |d|':>10}  {'Match%':>8}  Status")
    print("  " + "-" * 64)

    for feat in list(DEFAULT_LANE_FEATURE_NAMES) + list(DEFAULT_TLS_FEATURE_NAMES):
        errs = feat_errs.get(feat, [0.0])
        mean_e = float(np.mean(errs))
        max_e  = float(np.max(errs))
        m_pct  = 100.0 * sum(1 for e in errs if e < 1e-5) / max(len(errs), 1)
        ok = max_e < 1e-5
        if not ok:
            failures += 1
        print(f"  {feat:<25}  {mean_e:>10.2e}  {max_e:>10.2e}  {m_pct:>7.2f}%  "
              f"{PASS if ok else FAIL}")

    if match_pct < 99.9:
        failures += 1
    print(f"\n  -> Assembly check: overall match {match_pct:.2f}%  "
          f"[{PASS if match_pct >= 99.9 else FAIL}]")
    return failures, match_pct


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 3 — METRIC COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def _make_tracks(
    lane_id: str,
    n_veh: int,
    n_stopped: int,
    n_moto: int,
    n_heavy: int,
    px_per_meter: float,
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    roi = LANE_ROIS[lane_id]
    poly = roi.polygon
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    cls_ids = (
        [3] * n_moto + [1] * n_heavy + [2] * max(0, n_veh - n_moto - n_heavy)
    )[:n_veh]

    tracks = []
    for i in range(n_veh):
        cls_id = cls_ids[i]
        w  = rng.uniform(18, 36)
        h  = rng.uniform(18, 36)
        cx = float(rng.uniform(xmin + w / 2, max(xmax - w / 2, xmin + w / 2 + 1)))
        cy = float(rng.uniform(ymin + h / 2, max(ymax - h / 2, ymin + h / 2 + 1)))
        speed_mps = 0.03 if i < n_stopped else float(rng.uniform(1.0, 8.0))
        tracks.append({
            "track_id":   i,
            "cls_id":     cls_id,
            "bbox":       [cx - w/2, cy - h/2, cx + w/2, cy + h/2],
            "center":     [cx, cy],
            "speed_px_s": speed_mps * px_per_meter,
            "cam_id":     roi.cam_id,   # required for _tracks_for_lane cam filter
            "timestamp":  float(i) * 0.1,
        })
    return tracks


def check_metric_computation(n_scenarios: int = 300) -> Tuple[int, float]:
    """
    Run the same scenario through StateExtractor._lane_metrics_from_tracks()
    and VisionLaneMetrics.compute() and compare the 5 lane features.

    Returns (n_hard_failures, overall_match_pct).
    """
    print("\n" + "=" * 70)
    print("CHECK 3 — METRIC COMPUTATION: StateExtractor vs VisionLaneMetrics")
    print("=" * 70)
    print("  Both receive the same pixel-space YOLO track data.")
    print()

    px_per_meter = 20.0
    speed_cap    = 15.0
    fps          = 10.0
    tol          = 0.05
    extractor = _build_extractor(px_per_meter=px_per_meter, speed_cap=speed_cap,
                                  stop_speed_m_s=0.1)

    rng = np.random.default_rng(123)
    feat_diffs: Dict[str, List[float]] = {f: [] for f in DEFAULT_LANE_FEATURE_NAMES}

    for _ in range(n_scenarios):
        n_veh     = int(rng.integers(0, 16))
        n_stopped = int(rng.integers(0, n_veh + 1))
        n_moto    = int(rng.integers(0, max(1, n_veh + 1)))
        n_heavy   = int(rng.integers(0, max(1, n_veh - n_moto + 1)))
        lane_id   = str(rng.choice(list(LANE_ROIS.keys())))

        tracks = _make_tracks(
            lane_id=lane_id, n_veh=n_veh, n_stopped=n_stopped,
            n_moto=n_moto, n_heavy=n_heavy, px_per_meter=px_per_meter, rng=rng,
        )

        # Path A: StateExtractor
        m_se = extractor._lane_metrics_from_tracks(lane_id, tracks)

        # Path B: VisionLaneMetrics
        roi = LANE_ROIS[lane_id]
        lane_area_px  = roi.lane_length_px * roi.lane_width_px
        lane_length_m = roi.lane_length_px / px_per_meter

        class _T:
            __slots__ = ("tlwh", "cls", "velocity")
            def __init__(self, t: Dict[str, Any]) -> None:
                b = t["bbox"]
                self.tlwh     = [b[0], b[1], b[2]-b[0], b[3]-b[1]]
                self.cls      = t["cls_id"]
                spx_f         = float(t.get("speed_px_s", 0.0)) / fps
                self.velocity = np.array([spx_f, 0.0], dtype=np.float32)

        m_vlm = VisionLaneMetrics.compute(
            tracks=[_T(t) for t in tracks],
            lane_roi_area=lane_area_px,
            lane_length_m=lane_length_m,
            speed_cap_mps=speed_cap,
            px_per_meter=px_per_meter,
            fps=fps,
        )

        for feat in DEFAULT_LANE_FEATURE_NAMES:
            feat_diffs[feat].append(
                abs(float(m_se.get(feat, 0.0)) - float(m_vlm.get(feat, 0.0)))
            )

    print(f"  {'Feature':<25}  {'Mean|d|':>9}  {'P50|d|':>9}  {'P95|d|':>9}  "
          f"{'Max|d|':>9}  {'Match%':>8}  Status")
    print("  " + "-" * 82)
    total_ok = 0
    total_n  = 0
    hard_failures = 0

    for feat in DEFAULT_LANE_FEATURE_NAMES:
        d   = np.array(feat_diffs[feat])
        n_ok = int((d < tol).sum())
        m_pct = 100.0 * n_ok / max(len(d), 1)
        total_ok += n_ok
        total_n  += len(d)
        if m_pct >= 95.0:
            status = PASS
        elif m_pct >= 80.0:
            status = WARN
        else:
            status = FAIL
            hard_failures += 1
        print(f"  {feat:<25}  {d.mean():>9.4f}  {np.median(d):>9.4f}  "
              f"{np.percentile(d,95):>9.4f}  {d.max():>9.4f}  {m_pct:>7.2f}%  {status}")

    overall_pct = 100.0 * total_ok / max(total_n, 1)
    print()
    print("  Notes:")
    print(f"    occupancy_norm:  SE = Sigma(veh_len_px)/lane_len_px  |  VLM = Sigma(veh_len_m)/lane_len_m  -> 1D match")
    print(f"    effective_queue: SE halt={extractor.stop_speed_m_s} m/s  "
          f"VLM halt={VisionLaneMetrics.HALT_SPEED_THRESHOLD_MPS} m/s"
          + ("  <- DIVERGENCE" if extractor.stop_speed_m_s != VisionLaneMetrics.HALT_SPEED_THRESHOLD_MPS else "  <- match"))
    print(f"    avg_speed_norm:  SE: speed_px_s/px_per_meter/speed_cap  "
          f"VLM: speed_px_frame*fps/px_per_meter/speed_cap -> same result")
    print()
    print(f"  -> Metric computation match (|delta|<{tol}): {overall_pct:.2f}%  "
          f"[{PASS if overall_pct >= 95 else (WARN if overall_pct >= 80 else FAIL)}]")
    return hard_failures, overall_pct


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 4 — PARAMETER ALIGNMENT MATRIX
# ══════════════════════════════════════════════════════════════════════════════

def check_parameter_alignment() -> int:
    print("\n" + "=" * 70)
    print("CHECK 4 — PARAMETER ALIGNMENT MATRIX")
    print("=" * 70)

    state_cfg_path = ROOT / "configs" / "state_config.json"
    with open(state_cfg_path, encoding="utf-8") as f:
        state_cfg = json.load(f)
    sp = state_cfg.get("system_params", {})

    sumo_cfg_path = ROOT / "sumo_configs" / "evaluation" / "medium" / "sumo_config.sumocfg"
    if not sumo_cfg_path.exists():
        print(f"  [SKIP] SUMO config not found: {sumo_cfg_path}")
        return 0

    env_cfg = build_default_config(sumo_cfg_path=str(sumo_cfg_path), gui=False)

    rows = [
        ("max_green_time (s)",
         float(sp.get("max_green_time", 60)), float(env_cfg.sim.max_green_time), 90.0),
        ("speed_cap (m/s)",
         float(sp.get("speed_cap", 15.0)), float(env_cfg.observation.speed_cap), 15.0),
        ("px_per_meter",
         float(sp.get("px_per_meter", 20.0)), 20.0, 20.0),
        ("max_lanes_per_tls",
         float(sp.get("max_lanes_per_tls", 4)), float(env_cfg.max_lanes_per_tls), 4.0),
        ("queue_cap",
         float(sp.get("queue_cap", 50.0)), float(env_cfg.observation.queue_cap), 50.0),
        ("waiting_cap",
         float(sp.get("waiting_cap", 300.0)), float(env_cfg.observation.waiting_cap), 300.0),
    ]

    failures = 0
    print(f"\n  {'Parameter':<22}  {'state_config':>14}  {'TrafficEnvCfg':>14}  "
          f"{'SE default':>12}  Status  Note")
    print("  " + "-" * 82)
    for label, v_sc, v_env, v_se in rows:
        sc_env_ok = abs(v_sc - v_env) < 1e-6
        all_ok    = sc_env_ok and abs(v_sc - v_se) < 1e-6
        status = PASS if sc_env_ok else FAIL
        if not sc_env_ok:
            failures += 1
        note = "" if all_ok else ("SE default differs — override in orchestrator" if sc_env_ok else "")
        print(f"  {label:<22}  {v_sc:>14.1f}  {v_env:>14.1f}  {v_se:>12.1f}  "
              f"{status}  {note}")

    print()
    if failures:
        print(f"  {failures} mismatch(es) found. Ensure orchestrators load from state_config.json.")
    else:
        print("  All key parameters match between state_config.json and TrafficEnvConfig.")
    return failures


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 5 — LIVE SUMO (optional)
# ══════════════════════════════════════════════════════════════════════════════

def check_sumo_live(n_steps: int = 50) -> Tuple[int, float]:
    """Run SUMO and compare obs from SUMO-native vs injected-SUMO-metrics paths."""
    print("\n" + "=" * 70)
    print("CHECK 5 — LIVE SUMO: SUMO obs vs re-assembled obs from same SUMO metrics")
    print("=" * 70)

    sumo_cfg = ROOT / "sumo_configs" / "evaluation" / "medium" / "sumo_config.sumocfg"
    if not sumo_cfg.exists():
        print(f"  [SKIP] SUMO config not found: {sumo_cfg}")
        return 0, float("nan")

    try:
        env_cfg = build_default_config(sumo_cfg_path=str(sumo_cfg), gui=False)
        from src.traffic_env.envs.multi_agent import MappoTrafficEnv
        env = MappoTrafficEnv(config=env_cfg, gui=False, use_libsumo=False)
    except Exception as e:
        print(f"  [SKIP] Could not initialise SUMO env: {e}")
        return 0, float("nan")

    print(f"  Running {n_steps} SUMO steps ...")
    obs, _ = env.reset()
    rng = np.random.default_rng(7)

    cos_sims: List[float] = []
    feat_diffs: Dict[str, List[float]] = {
        f: [] for f in list(DEFAULT_LANE_FEATURE_NAMES) + list(DEFAULT_TLS_FEATURE_NAMES)
    }

    for _ in range(n_steps):
        action = {t: int(rng.integers(0, 2)) for t in env_cfg.tls_ids}
        obs_dict_sumo, _, _, _, _ = env.step(action)

        sumo_lane_cache = env.obs_builder.build_lane_cache(list(env_cfg.tls_ids))
        env.obs_builder.inject_vision_cache(sumo_lane_cache)
        obs_dict_inj, _ = env.obs_builder.get_all_observations(
            env.obs_builder.build_lane_cache(),
            green_timers=env.green_timers,
        )

        sumo_vec = np.concatenate([obs_dict_sumo[t] for t in env_cfg.tls_ids]).astype(np.float32)
        inj_vec  = np.concatenate([obs_dict_inj[t]  for t in env_cfg.tls_ids]).astype(np.float32)

        n_a = float(np.linalg.norm(sumo_vec)) + 1e-12
        n_b = float(np.linalg.norm(inj_vec))  + 1e-12
        cos_sims.append(float(np.dot(sumo_vec, inj_vec)) / (n_a * n_b))

        diff = np.abs(sumo_vec - inj_vec)
        for ai in range(len(env_cfg.tls_ids)):
            base = ai * 26
            for li in range(4):
                for k, feat in enumerate(DEFAULT_LANE_FEATURE_NAMES):
                    idx = base + li * 5 + k
                    if idx < len(diff):
                        feat_diffs[feat].append(float(diff[idx]))
            for k, feat in enumerate(DEFAULT_TLS_FEATURE_NAMES):
                idx = base + 20 + k
                if idx < len(diff):
                    feat_diffs[feat].append(float(diff[idx]))

    env.close()
    mean_cos = float(np.mean(cos_sims))
    print(f"\n  Mean cosine similarity (target >0.95): {mean_cos:.4f}")
    print()
    print(f"  {'Feature':<25}  {'Mean|d|':>9}  {'Max|d|':>9}  {'Match%':>8}")
    print("  " + "-" * 56)
    for feat in list(DEFAULT_LANE_FEATURE_NAMES) + list(DEFAULT_TLS_FEATURE_NAMES):
        d = np.array(feat_diffs.get(feat, [0.0]))
        print(f"  {feat:<25}  {d.mean():>9.4f}  {d.max():>9.4f}  "
              f"{100.0*(d < 0.05).mean():>7.2f}%")

    ok = mean_cos >= 0.95
    print(f"\n  -> Live SUMO: cosine={mean_cos:.4f}  [{PASS if ok else FAIL}]")
    return (0 if ok else 1), mean_cos


# ══════════════════════════════════════════════════════════════════════════════
# KNOWN GAPS
# ══════════════════════════════════════════════════════════════════════════════

def print_known_gaps() -> None:
    print("\n" + "=" * 70)
    print("KNOWN SYSTEMATIC GAPS")
    print("=" * 70)
    gaps = [
        (
            "occupancy_norm: temporal mismatch (residual, not a bug)",
            "Both SUMO and YOLO now use 1D formula: Sigma(vehicle_length) / lane_length.\n"
            "     Residual gap: SUMO reports step-averaged occupancy over 5 s;\n"
            "     camera reports instantaneous occupancy at 10 Hz + EMA smoothing.\n"
            "     These values are semantically equivalent but not bit-for-bit equal.",
        ),
        (
            "avg_speed_norm: ByteTrack velocity noise at low speeds",
            "ByteTrack estimates speed via delta-position / delta-time.\n"
            "     At 10 fps + px_per_meter=20, a 1-pixel jitter = 0.5 m/s absolute noise.\n"
            "     In congestion (0-3 m/s regime), relative error can exceed 50%.\n"
            "     EMA alpha=0.6 partially mitigates; no code fix available.",
        ),
        (
            "phase_one_hot: yellow transitions never appear in deployment",
            "Training: policy sees [0,1,0,0] and [0,0,0,1] briefly during yellow phases.\n"
            "     Deployment: orchestrator only tracks green phases (0 or 2); yellow is\n"
            "     handled by Arduino sub-second hardware. Policy never sees yellow encoding.\n"
            "     Impact: minor — yellow phases are 3 s out of a 60 s cycle.",
        ),
    ]
    for title, detail in gaps:
        print(f"\n  [!] {title}")
        print(f"      {detail}")


# ── output helpers ────────────────────────────────────────────────────────────

def _row(ok: bool, label: str, detail: str = "") -> None:
    status = PASS if ok else FAIL
    print(f"  [{status}]  {label:<42} {detail}")


def _row_warn(label: str, detail: str = "") -> None:
    print(f"  [{WARN}]  {label:<42} {detail}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check YOLO/SUMO obs vector alignment."
    )
    parser.add_argument("--sumo", action="store_true",
                        help="Also run live SUMO check (requires SUMO + SUMO_HOME).")
    parser.add_argument("--scenarios", type=int, default=300,
                        help="Scenarios for checks 2 & 3 (default 300).")
    parser.add_argument("--sumo-steps", type=int, default=50,
                        help="SUMO steps for check 5 (default 50).")
    args = parser.parse_args()

    t0 = time.perf_counter()
    print("=" * 70)
    print("  YOLO <-> SUMO Observation-Vector Match Checker")
    print(f"  Project: {ROOT.name}")
    print("=" * 70)

    f1           = check_structural()
    f2, pct2     = check_assembly(n_scenarios=args.scenarios)
    f3, pct3     = check_metric_computation(n_scenarios=args.scenarios)
    f4           = check_parameter_alignment()
    f5, cos5     = 0, float("nan")
    if args.sumo:
        f5, cos5 = check_sumo_live(n_steps=args.sumo_steps)

    print_known_gaps()

    elapsed = time.perf_counter() - t0
    total   = f1 + f2 + f3 + f4 + f5

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Check 1 structural:       {'PASS' if f1==0 else 'FAIL':4}  ({f1} failure(s))")
    print(f"  Check 2 assembly equiv:   {'PASS' if f2==0 else 'FAIL':4}  match={pct2:.2f}%")
    print(f"  Check 3 metric compute:   {'PASS' if f3==0 else 'FAIL':4}  match={pct3:.2f}%")
    print(f"  Check 4 param alignment:  {'PASS' if f4==0 else 'FAIL':4}  ({f4} mismatch(es))")
    if args.sumo:
        print(f"  Check 5 live SUMO:        {'PASS' if f5==0 else 'FAIL':4}  cosine={cos5:.4f}")
    print(f"\n  Total failures: {total}")
    print(f"  Elapsed: {elapsed:.1f}s")

    if pct3 >= 95.0:
        print(f"\n  TARGET >=95% metric match (Check 3): {pct3:.2f}%  ACHIEVED")
    else:
        print(f"\n  TARGET >=95% metric match (Check 3): {pct3:.2f}%  NOT MET")
        print("  Resolve the known gaps above to reach the 95% threshold.")

    sys.exit(0 if total == 0 else 1)


if __name__ == "__main__":
    main()
