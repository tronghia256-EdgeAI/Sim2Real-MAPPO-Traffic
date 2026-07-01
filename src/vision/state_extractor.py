from __future__ import annotations

"""
state_extractor.py  (audit-fixed v3 — schema 1.1.0)
======================================
Convert YOLO + ByteTrack semantic outputs into the exact same observation layout
that the current SUMO-based PPO / MAPPO policy was trained on.

SCHEMA 1.1.0 (S1/S2):
  S2  pressure_norm is now SIGNED: 0.5*(1 + mean_q(group_a) - mean_q(group_b)),
      clipped to [0,1]; 0.5 = balanced. Matches observations.py exactly.
  S1  (training side) each obs slot is now one APPROACH (lanes of one incoming
      edge, aggregated). Deployment is unchanged structurally: each camera ROI
      already covers one approach, so slot k = ROI k as before — but ROI order
      in controlled_lanes_dict MUST match the training approach order
      (getControlledLanes edge order). Verify via check_obs_match.py --sumo.

AUDIT FIXES applied (see AUDIT_REPORT.md for full analysis):
  FIX-1  FEATURE_NAMES reduced from 7 → 5 to match observations.py.
         build_state() loop now writes exactly those 5 features per lane slot.
         observation_dim is recomputed correctly.
  FIX-2  Heavy-vehicle COCO class IDs corrected: {1,4} → {5,7}
         (bicycle, airplane → bus, truck).
  FIX-3  Pressure normalisation corrected:
         Now uses effective_queue_norm (∈[0,1] ratio) / max_lanes_per_tls,
         matching the _tls_pressure_proxy() in observations.py exactly.
  FIX-4  observation_dim recomputed as n_tls × (max_lanes×5 + 6).

Canonical observation layout (per TLS, 26 dims for max_lanes=4):
─────────────────────────────────────────────────────────────────
  lane_0 : [effective_queue_norm, occupancy_norm, avg_speed_norm,
             motorbike_share, heavy_vehicle_share]      (5 dims)
  lane_1 :  … same 5 features …
  lane_2 :  … same 5 features …
  lane_3 :  … same 5 features …
  TLS    : [phase_0, phase_1, phase_2, phase_3,
             green_timer_norm, pressure_norm]           (6 dims)
─────────────────────────────────────────────────────────────────
Total = n_tls × (max_lanes_per_tls × 5 + 6)

Feature definitions (must match observations.py exactly):
  effective_queue_norm  = halted_vehicle_length_px / lane_length_px   ∈ [0,1]
  occupancy_norm        = Σ(vehicle_length_px) / lane_length_px       ∈ [0,1]
  avg_speed_norm        = mean_speed / speed_cap                       ∈ [0,1]
  motorbike_share       = motorbike_count / vehicle_count             ∈ [0,1]
  heavy_vehicle_share   = (bus+truck) count / vehicle_count           ∈ [0,1]
  phase_k               = one-hot for current phase ∈ {0,1,2,3}
  green_timer_norm      = elapsed_green_s / max_green_time            ∈ [0,1]
  pressure_norm         = 0.5·(1 + mean_q(group_a) − mean_q(group_b))  ∈ [0,1]
                          (signed; 0.5 = balanced — schema 1.1.0)

Temporal smoothing
------------------
This module produces raw, instantaneous per-frame estimates.
All temporal smoothing (EMA / median) is handled exclusively by VisionBuffer.
Do NOT add smoothing here — it would create double-smoothing when VisionBuffer
is active, which was absent from the SUMO training loop.

Custom YOLO vehicle class mapping
----------------------------------
  1 = bus            (heavy)
  2 = car            (light)
  3 = motorcycle     (motorbike)
  4 = truck          (heavy)

Input track format
------------------
Each track in the frame's "tracks" list should contain:
  track_id    : int
  cls_id      : int    (COCO class id)
  bbox        : [x1, y1, x2, y2]  image pixels
  speed_px_s  : float  (optional; pixels per second; 0 if unavailable)
  center      : [cx, cy]  (optional; computed from bbox if absent)
  timestamp   : float  (optional)
  cam_id      : str    (optional; used for per-camera track memory)
"""

from dataclasses import dataclass
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import math
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# LaneROI
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LaneROI:
    """
    Region of interest for a single traffic lane in camera pixel coordinates.

    Parameters
    ----------
    lane_id : str
        Unique identifier matching keys in controlled_lanes_dict.
    tls_id : str
        Parent traffic-light system.
    polygon : list of (x, y) float pairs
        Closed polygon in pixel coordinates.
        Used for bbox-centroid point-in-polygon assignment.
    cam_id : int or str, optional
        Camera id filter.  If given, only tracks from this camera are
        assigned to the lane.
    lane_length_px : float, optional
        Physical length of the lane in pixels.
        If None, estimated from the polygon bounding box extent.
    lane_width_px : float, optional
        Physical width of the lane in pixels.
        If None, estimated as polygon_area / lane_length_px.
    """

    lane_id:        str
    tls_id:         str
    polygon:        List[Tuple[float, float]]
    cam_id:         Optional[Union[int, str]] = None
    lane_length_px: Optional[float]           = None
    lane_width_px:  Optional[float]           = None


# ─────────────────────────────────────────────────────────────────────────────
# StateExtractor
# ─────────────────────────────────────────────────────────────────────────────

class StateExtractor:
    """
    Convert YOLO / ByteTrack detection snapshots to a PPO observation vector
    that is dimensionally and semantically identical to the SUMO training obs.

    Parameters
    ----------
    tls_ids : sequence of str
        Ordered list of TLS ids.  Order determines obs layout.
    controlled_lanes_dict : {tls_id → [lane_id, …]}
        Lane ids per TLS in the same order used during training.
    lane_rois : {lane_id → LaneROI}
        Camera-space ROI for each lane.
    lane_groups : {tls_id → ([group_a], [group_b])}, optional
        Opposing lane groups for pressure computation.
        If absent, lanes are split by list order (first half vs second half).
    max_lanes_per_tls : int
        Fixed obs width per TLS.  Zero-padded if fewer lanes are active.
    speed_cap : float
        Normalisation cap for avg_speed.  Must match training config.
        Same units as track["speed_px_s"].
        Calibrate: speed_cap_mps × px_per_meter / fps.
    max_green_time : float
        Normalisation cap for green timer (seconds).
    stop_speed_px_s : float
        Speed threshold for "halted" classification.
        Calibrate: 0.1 m/s × px_per_meter / fps
        (HALT_SPEED_THRESHOLD_MPS=0.1 from VisionLaneMetrics; B1/B1b fix —
        training, SUMO _lane_metrics, and state_config all use 0.1 m/s).
    """

    # ── FIX-1: 5 lane features matching DEFAULT_LANE_FEATURE_NAMES ───────────
    FEATURE_NAMES: Tuple[str, ...] = (
        "effective_queue_norm",   # halted_length_px / lane_length_px
        "occupancy_norm",         # Σbbox_area / lane_roi_area_px
        "avg_speed_norm",         # avg_speed / speed_cap
        "motorbike_share",        # motorcycle count / total
        "heavy_vehicle_share",    # (bus+truck) count / total
    )

    # Custom YOLO class IDs: 0=accident,1=bus,2=car,3=motorcycle,4=truck
    _MOTORBIKE_CLASSES: frozenset = frozenset({3})         # motorcycle
    _HEAVY_CLASSES:     frozenset = frozenset({1, 4})      # bus, truck

    # Physical vehicle lengths in metres. Used to derive pixel lengths at init.
    # Matches VisionLaneMetrics.CLASS_LENGTH_M so both pipelines agree.
    # Uses custom YOLO class IDs: 1=bus, 2=car, 3=motorcycle, 4=truck
    _CLASS_LENGTH_M: Dict[int, float] = {3: 1.8, 2: 4.5, 1: 12.0, 4: 8.0}
    _DEFAULT_LENGTH_M: float = 4.5
    # Pixel lengths are derived at __init__ from _CLASS_LENGTH_M × px_per_meter.
    # The old hardcoded values (car=30px) were calibrated for ~6.7 px/m, not 20 px/m.
    _CLASS_LENGTH_PX: Dict[int, float] = {3: 10.0, 2: 30.0, 1: 80.0, 4: 55.0}
    _DEFAULT_LENGTH_PX: float = 30.0

    def __init__(
        self,
        tls_ids:               Sequence[str],
        controlled_lanes_dict: Dict[str, List[str]],
        lane_rois:             Dict[str, LaneROI],
        lane_groups:           Optional[Dict[str, Tuple[List[str], List[str]]]] = None,
        motorbike_class_ids: Optional[Sequence[int]] = None,
        heavy_class_ids:     Optional[Sequence[int]] = None,
        px_per_meter:          float = 20.0,
        stop_speed_m_s:        float = 0.1,
        upstream_phase_obs:    bool  = False,
        upstream_tls_map:      Optional[Dict[str, List[str]]] = None,
        max_lanes_per_tls:     int   = 4,
        speed_cap:             float = 15.0,
        max_green_time:        float = 60.0,
        # Diagnostic caps (not used in 5 policy features):
        waiting_cap:           float = 300.0,
        vehicle_cap:           float = 20.0,
        queue_cap:             float = 50.0,
        fps:                   float = 10.0,
    ) -> None:
        self.tls_ids               = list(tls_ids)
        self.controlled_lanes_dict = {k: list(v) for k, v in controlled_lanes_dict.items()}
        self.lane_rois             = dict(lane_rois)
        self.lane_groups           = lane_groups or {}
        self.max_lanes_per_tls     = int(max_lanes_per_tls)
        # corridor coordination: must match training upstream_phase_obs. The
        # orchestrator supplies upstream_tls_map (which TLS feed each TLS) and the
        # full phase_map at inference, so the neighbour phases are known without
        # any camera. Keep formula-identical to ObservationBuilder._upstream_phase_vector.
        self.upstream_phase_obs    = bool(upstream_phase_obs)
        self.upstream_tls_map      = {k: list(v) for k, v in (upstream_tls_map or {}).items()}

        if self.max_lanes_per_tls <= 0:
            raise ValueError("max_lanes_per_tls must be > 0")
        if px_per_meter <= 0:
            raise ValueError("px_per_meter must be > 0")
        if stop_speed_m_s < 0:
            raise ValueError("stop_speed_m_s must be >= 0")
        if motorbike_class_ids is not None:                            
            self._MOTORBIKE_CLASSES = frozenset(motorbike_class_ids)   
        if heavy_class_ids is not None:                              
            self._HEAVY_CLASSES = frozenset(heavy_class_ids)  
        self.px_per_meter   = float(px_per_meter)
        self.stop_speed_m_s = float(stop_speed_m_s)
        self.speed_cap       = float(speed_cap)
        self.max_green_time  = float(max_green_time)
        self.waiting_cap     = float(waiting_cap)
        self.vehicle_cap     = float(vehicle_cap)
        self.queue_cap       = float(queue_cap)
        self.default_dt      = 1.0 / max(float(fps), 1e-6)

        # Derive pixel vehicle lengths from physical lengths at the configured
        # px_per_meter so effective_queue_norm matches VisionLaneMetrics.
        self._CLASS_LENGTH_PX = {
            cls_id: length_m * self.px_per_meter
            for cls_id, length_m in self._CLASS_LENGTH_M.items()
        }
        self._DEFAULT_LENGTH_PX = self._DEFAULT_LENGTH_M * self.px_per_meter

        # ── FIX-4: recomputed with lane_feature_dim = 5 ──────────────────────
        self.lane_feature_dim = len(self.FEATURE_NAMES)          # 5
        # 4-hot + green + pressure (+4 upstream one-hot when upstream_phase_obs)
        self.tls_feature_dim  = 6 + (4 if self.upstream_phase_obs else 0)
        self.observation_dim  = len(self.tls_ids) * (
            self.max_lanes_per_tls * self.lane_feature_dim + self.tls_feature_dim
        )

        # Per-camera, per-track waiting time memory
        self.track_memory: Dict[Any, Dict[int, Dict[str, float]]] = defaultdict(dict)

        self.last_lane_metrics:   Dict[str, Dict[str, float]] = {}
        self.last_accident_found: bool  = False
        self.last_accident_conf:  float = 0.0

        for tls_id in self.tls_ids:
            if tls_id not in self.controlled_lanes_dict:
                raise ValueError(f"Missing controlled lanes for tls_id={tls_id}")
            lane_ids = self.controlled_lanes_dict[tls_id][: self.max_lanes_per_tls]
            for lane_id in lane_ids:
                if lane_id not in self.lane_rois:
                    raise ValueError(f"Lane '{lane_id}' missing from lane_rois")

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def get_obs_dim(self) -> int:
        return int(self.observation_dim)

    def reset(self, cam_id: Optional[Union[int, str]] = None) -> None:
        if cam_id is None:
            self.track_memory.clear()
        else:
            self.track_memory.pop(cam_id, None)
        self.last_lane_metrics.clear()
        self.last_accident_found = False
        self.last_accident_conf  = 0.0

    def build_state(
        self,
        parsed_frames: Union[Dict[Union[int, str], Dict[str, Any]], List[Any], Any],
        phase_map:    Dict[str, int],
        green_timers: Dict[str, float],
        return_info:  bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Build the PPO observation from one or multiple camera detections.

        Accepted snapshot shapes:
          (a) list of track dicts / ParsedFrame objects
          (b) single frame dict: {"tracks": [...], "accident_found": bool}
          (c) multi-camera dict: {cam_id: frame_dict, ...}
        """
        frame_list = self._normalize_frames_input(parsed_frames)
        tracks     = self._collect_tracks(frame_list)

        self.last_accident_found = False
        self.last_accident_conf  = 0.0
        for fr in frame_list:
            fd = self._as_dict(fr)
            if bool(fd.get("accident_found", False)):
                self.last_accident_found = True
                self.last_accident_conf  = max(
                    self.last_accident_conf, float(fd.get("accident_conf", 0.0))
                )

        state: List[float] = []
        self.last_lane_metrics = {}

        for tls_id in self.tls_ids:
            lane_ids = self.controlled_lanes_dict.get(tls_id, [])[: self.max_lanes_per_tls]
            lane_metrics_cache: Dict[str, Dict[str, float]] = {}

            for lane_id in lane_ids:
                m = self._lane_metrics_from_tracks(lane_id, tracks)
                lane_metrics_cache[lane_id] = m
                self.last_lane_metrics[lane_id] = m

                # ── FIX-1: exactly 5 policy features ─────────────────────────
                # !! DO NOT add extra features without full PPO retraining !!
                state.extend([
                    m["effective_queue_norm"],
                    m["occupancy_norm"],
                    m["avg_speed_norm"],
                    m["motorbike_share"],
                    m["heavy_vehicle_share"],
                ])

            # Zero-pad missing lane slots (same as training)
            missing = self.max_lanes_per_tls - len(lane_ids)
            if missing > 0:
                state.extend([0.0] * (missing * self.lane_feature_dim))

            # 6 TLS-level features
            state.extend(self._phase_one_hot(int(phase_map.get(tls_id, 0))))
            state.append(self._normalize(float(green_timers.get(tls_id, 0.0)), self.max_green_time))
            state.append(self._tls_pressure_proxy(tls_id, lane_ids, lane_metrics_cache))

            # +4 upstream neighbour phase one-hot (corridor coordination)
            if self.upstream_phase_obs:
                state.extend(self._upstream_phase_vector(tls_id, phase_map))

        obs = np.asarray(state, dtype=np.float32)
        if obs.shape != (self.observation_dim,):
            raise RuntimeError(
                f"Observation shape mismatch for tls_ids={self.tls_ids}: "
                f"got {obs.shape}, expected {(self.observation_dim,)}. "
                f"Verify controlled_lanes_dict and max_lanes_per_tls match training config."
            )

        if return_info:
            return obs, {
                "accident_found": self.last_accident_found,
                "accident_conf":  self.last_accident_conf,
                "lane_metrics":   self.last_lane_metrics,
            }
        return obs

    def build_semantic_state(
        self,
        parsed_frames: Union[Dict[Union[int, str], Dict[str, Any]], List[Any], Any],
    ) -> Dict[str, Any]:
        frame_list = self._normalize_frames_input(parsed_frames)
        tracks     = self._collect_tracks(frame_list)

        out: Dict[str, Any] = {
            "accident_found": False, "accident_conf": 0.0,
            "num_cameras": len(frame_list),
            "tracks_by_lane": {}, "lane_metrics": {},
        }
        for fr in frame_list:
            fd = self._as_dict(fr)
            if bool(fd.get("accident_found", False)):
                out["accident_found"] = True
                out["accident_conf"]  = max(float(out["accident_conf"]),
                                            float(fd.get("accident_conf", 0.0)))

        for lane_id in self.lane_rois:
            out["lane_metrics"][lane_id]   = self._lane_metrics_from_tracks(lane_id, tracks)
            out["tracks_by_lane"][lane_id] = self._tracks_for_lane(lane_id, tracks)
        return out

    # ──────────────────────────────────────────────────────────────────────────
    # Lane metrics
    # ──────────────────────────────────────────────────────────────────────────

    def _empty_lane_metrics(self) -> Dict[str, float]:
        return {
            "effective_queue_norm": 0.0, "occupancy_norm": 0.0,
            "avg_speed_norm": 0.0, "motorbike_share": 0.0, "heavy_vehicle_share": 0.0,
            "queue_norm": 0.0, "waiting_norm": 0.0, "vehicle_count_norm": 0.0,
            "stopped_ratio": 0.0, "raw_waiting": 0.0, "raw_speed": 0.0,
            "raw_vehicle_count": 0.0, "raw_halted_length_px": 0.0,
            "lane_length_px": 1.0, "lane_area_px": 1.0,
        }

    def _lane_metrics_from_tracks(
        self, lane_id: str, tracks: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        roi = self.lane_rois.get(lane_id)
        if roi is None:
            return self._empty_lane_metrics()

        lane_length_px, _, lane_area_px = self._lane_geometry(lane_id)
        lane_tracks = self._tracks_for_lane(lane_id, tracks)

        if not lane_tracks:
            return {**self._empty_lane_metrics(),
                    "lane_length_px": float(lane_length_px),
                    "lane_area_px":   float(lane_area_px)}

        total_speed             = 0.0
        total_vehicle_count     = 0.0
        halted_length_px        = 0.0
        total_vehicle_length_px = 0.0
        motorbike_count         = 0.0
        heavy_vehicle_count     = 0.0
        queue_count             = 0.0
        waiting_time            = 0.0

        fallback_ts  = float(lane_tracks[0].get("timestamp", 0.0))
        fallback_cam = lane_tracks[0].get("cam_id", "cam0")

        for t in lane_tracks:
            bbox = t.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            cls_id = int(t.get("cls_id", -1))
            if cls_id == 0:          # skip accident class
                continue

            track_id = t.get("track_id")
            speed_px_s = float(t.get("speed_px_s", 0.0))
            if not math.isfinite(speed_px_s):
                speed_px_s = 0.0
            speed_m_s = speed_px_s / self.px_per_meter

            total_vehicle_count     += 1.0
            total_speed             += speed_m_s
            veh_len_px = self._CLASS_LENGTH_PX.get(cls_id, self._DEFAULT_LENGTH_PX)
            total_vehicle_length_px += veh_len_px

            if cls_id in self._MOTORBIKE_CLASSES:
                motorbike_count     += 1.0
            if cls_id in self._HEAVY_CLASSES:       # custom YOLO: {1,4}
                heavy_vehicle_count += 1.0

            stopped = speed_m_s <= self.stop_speed_m_s
            if stopped:
                queue_count      += 1.0
                halted_length_px += veh_len_px

            if track_id is not None:
                waiting_time += self._update_track_stop_memory(
                    cam_id    = t.get("cam_id", fallback_cam),
                    track_id  = int(track_id),
                    timestamp = float(t.get("timestamp", fallback_ts)),
                    stopped   = stopped,
                )
            else:
                waiting_time += self.default_dt if stopped else 0.0

        n = max(total_vehicle_count, 1.0)

        return {
            # ── Policy features (5) ───────────────────────────────────────────
            "effective_queue_norm": float(np.clip(halted_length_px   / max(lane_length_px, 1.0), 0.0, 1.0)),
            "occupancy_norm":       float(np.clip(total_vehicle_length_px / max(lane_length_px, 1.0), 0.0, 1.0)),
            "avg_speed_norm":       self._normalize(total_speed / n,  self.speed_cap),
            "motorbike_share":      float(np.clip(motorbike_count    / n, 0.0, 1.0)),
            "heavy_vehicle_share":  float(np.clip(heavy_vehicle_count/ n, 0.0, 1.0)),
            # ── Diagnostics (not in policy obs) ──────────────────────────────
            "queue_norm":           self._normalize(queue_count,          self.queue_cap),
            "waiting_norm":         self._normalize(waiting_time,         self.waiting_cap),
            "vehicle_count_norm":   self._normalize(total_vehicle_count,  self.vehicle_cap),
            "stopped_ratio":        float(np.clip(queue_count / n, 0.0, 1.0)),
            "raw_waiting":          float(waiting_time),
            "raw_speed":            float(total_speed / n),
            "raw_vehicle_count":    float(total_vehicle_count),
            "raw_halted_length_px": float(halted_length_px),
            "lane_length_px":       float(lane_length_px),
            "lane_area_px":         float(lane_area_px),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # TLS features
    # ──────────────────────────────────────────────────────────────────────────

    def _phase_one_hot(self, phase: int) -> List[float]:
        # phase is a raw SUMO phase index (0-3). Encode directly so the
        # one-hot is semantically identical to ObservationBuilder._phase_one_hot().
        phase = int(np.clip(int(phase), 0, 3))
        vec = [0.0, 0.0, 0.0, 0.0]
        vec[phase] = 1.0
        return vec

    def _tls_pressure_proxy(
        self,
        tls_id:             str,
        lane_ids:           List[str],
        lane_metrics_cache: Dict[str, Dict[str, float]],
    ) -> float:
        """
        S2 FIX (schema 1.1.0): SIGNED pressure feature in [0,1].

        pressure_norm = clip(0.5 * (1 + mean_q(group_a) - mean_q(group_b)), 0, 1)

        0.5 = balanced; >0.5 = group A (phase-0 lanes) more queued. Group MEANS
        keep the value invariant to group size. Must stay formula-identical to
        ObservationBuilder._tls_pressure_proxy() (training side) — verified by
        scripts/check_obs_match.py CHECK 2.
        """
        empty = self._empty_lane_metrics()

        def eff_q(lid: str) -> float:
            return float(lane_metrics_cache.get(lid, empty).get("effective_queue_norm", 0.0))

        if tls_id in self.lane_groups:
            raw_a, raw_b = self.lane_groups[tls_id]
            group_a = [l for l in raw_a if l in lane_ids]
            group_b = [l for l in raw_b if l in lane_ids]
        else:
            if len(lane_ids) < 2:
                return 0.5  # no group info -> report balanced
            mid = len(lane_ids) // 2
            group_a = list(lane_ids[:mid])
            group_b = list(lane_ids[mid:])

        qa = sum(eff_q(l) for l in group_a) / max(len(group_a), 1)
        qb = sum(eff_q(l) for l in group_b) / max(len(group_b), 1)
        return float(np.clip(0.5 * (1.0 + qa - qb), 0.0, 1.0))

    def _upstream_phase_vector(self, tls_id: str, phase_map: Dict[str, int]) -> List[float]:
        """Mean phase one-hot over this TLS's upstream neighbours (4-d, in [0,1]).

        Formula-identical to ObservationBuilder._upstream_phase_vector(): the
        neighbour set comes from upstream_tls_map (topology, supplied by the
        orchestrator) and each neighbour's phase from the shared phase_map.
        All-zero when a TLS has no upstream neighbour (corridor endpoint).
        """
        ups = self.upstream_tls_map.get(tls_id, [])
        acc = [0.0, 0.0, 0.0, 0.0]
        if not ups:
            return acc
        for y in ups:
            one_hot = self._phase_one_hot(int(phase_map.get(y, 0)))
            for i in range(4):
                acc[i] += one_hot[i]
        n = float(len(ups))
        return [a / n for a in acc]

    # ──────────────────────────────────────────────────────────────────────────
    # Input helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _normalize_frames_input(self, parsed_frames: Any) -> List[Any]:
        if parsed_frames is None:
            return []
        if isinstance(parsed_frames, list):
            return parsed_frames
        if isinstance(parsed_frames, dict):
            if "tracks" in parsed_frames or "accident_found" in parsed_frames:
                return [parsed_frames]
            return list(parsed_frames.values())
        return [parsed_frames]

    def _as_dict(self, frame: Any) -> Dict[str, Any]:
        if isinstance(frame, dict):
            return frame
        if hasattr(frame, "__dict__"):
            return frame.__dict__
        raise TypeError(f"Unsupported frame type: {type(frame)}")

    def _collect_tracks(self, frame_list: List[Any]) -> List[Dict[str, Any]]:
        tracks: List[Dict[str, Any]] = []
        for fr in frame_list:
            d         = self._as_dict(fr)
            cam_id    = d.get("cam_id", "cam0")
            timestamp = float(d.get("timestamp", 0.0))
            for t in d.get("tracks", []):
                td = dict(t)
                td.setdefault("cam_id",    cam_id)
                td.setdefault("timestamp", timestamp)
                tracks.append(td)
        return tracks

    # ──────────────────────────────────────────────────────────────────────────
    # Geometry helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(value: float, cap: float) -> float:
        return float(np.clip(float(value) / max(float(cap), 1e-6), 0.0, 1.0))

    @staticmethod
    def _bbox_wh(bbox: Sequence[float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = map(float, bbox)
        return max(0.0, x2 - x1), max(0.0, y2 - y1)

    @staticmethod
    def _bbox_center(bbox: Sequence[float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = map(float, bbox)
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @staticmethod
    def _polygon_area(poly: Sequence[Tuple[float, float]]) -> float:
        if len(poly) < 3:
            return 0.0
        s, n = 0.0, len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            s += x1 * y2 - x2 * y1
        return abs(s) * 0.5

    @staticmethod
    def _point_in_polygon(x: float, y: float, poly: Sequence[Tuple[float, float]]) -> bool:
        if len(poly) < 3:
            return False
        inside, j = False, len(poly) - 1
        for i in range(len(poly)):
            xi, yi = poly[i]; xj, yj = poly[j]
            denom = yj - yi
            if abs(denom) < 1e-9:
                denom = 1e-9
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / denom + xi):
                inside = not inside
            j = i
        return inside

    def _lane_geometry(self, lane_id: str) -> Tuple[float, float, float]:
        roi = self.lane_rois.get(lane_id)
        if roi is None or not roi.polygon:
            return 100.0, 3.5, 350.0

        poly      = roi.polygon
        poly_area = max(self._polygon_area(poly), 1.0)

        lane_length_px = (
            float(roi.lane_length_px) if roi.lane_length_px and roi.lane_length_px > 0
            else float(max(max(p[0] for p in poly) - min(p[0] for p in poly),
                          max(p[1] for p in poly) - min(p[1] for p in poly), 1.0))
        )
        lane_width_px = (
            float(roi.lane_width_px) if roi.lane_width_px and roi.lane_width_px > 0
            else float(max(poly_area / max(lane_length_px, 1.0), 1.0))
        )
        lane_area_px = max(poly_area, lane_length_px * lane_width_px, 1.0)
        return lane_length_px, lane_width_px, lane_area_px

    def _tracks_for_lane(
        self, lane_id: str, tracks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        roi = self.lane_rois.get(lane_id)
        if roi is None:
            return []
        cam_filter = roi.cam_id
        out: List[Dict[str, Any]] = []
        for t in tracks:
            if cam_filter is not None and not self._cam_ids_match(t.get("cam_id"), cam_filter):
                continue
            bbox = t.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            center = t.get("center") or self._bbox_center(bbox)
            if self._point_in_polygon(float(center[0]), float(center[1]), roi.polygon):
                out.append(t)
        return out

    @staticmethod
    def _cam_ids_match(track_cam_id: Any, lane_cam_id: Any) -> bool:
        if track_cam_id == lane_cam_id:
            return True
        t = str(track_cam_id)
        l = str(lane_cam_id)
        if t == l:
            return True
        t_num = "".join(ch for ch in t if ch.isdigit())
        l_num = "".join(ch for ch in l if ch.isdigit())
        return bool(t_num and l_num and t_num == l_num)

    # ──────────────────────────────────────────────────────────────────────────
    # Waiting-time memory
    # ──────────────────────────────────────────────────────────────────────────

    def _update_track_stop_memory(
        self,
        cam_id:    Union[int, str],
        track_id:  int,
        timestamp: float,
        stopped:   bool,
    ) -> float:
        mem  = self.track_memory[cam_id]
        prev = mem.get(track_id)

        if prev is None:
            mem[track_id] = {"last_ts": float(timestamp), "stopped_s": 0.0}
            return 0.0

        dt        = max(float(timestamp) - float(prev["last_ts"]), self.default_dt)
        stopped_s = float(prev["stopped_s"])
        stopped_s = (stopped_s + dt) if stopped else 0.0   # hard reset on movement

        mem[track_id] = {"last_ts": float(timestamp), "stopped_s": stopped_s}
        return stopped_s if stopped else 0.0


# ─────────────────────────────────────────────────────────────────────────────
YoloStateExtractor = StateExtractor   # backward-compatible alias


def build_state_from_detector_snapshot(
    extractor:    StateExtractor,
    snapshot:     Union[Dict[Union[int, str], Dict[str, Any]], List[Any], Any],
    phase_map:    Dict[str, int],
    green_timers: Dict[str, float],
) -> np.ndarray:
    return extractor.build_state(snapshot, phase_map, green_timers)