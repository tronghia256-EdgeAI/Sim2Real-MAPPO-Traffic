from __future__ import annotations

"""
observation builder for a mappo-ready sumo traffic environment.

PATCH NOTES (audit v3 — schema 1.1.0):
- S1 FIX: build_local_obs previously sliced getControlledLanes()[:4], leaving
  4 of the 8 approach lanes per junction unobserved. Lanes are now grouped by
  parent edge into approaches and aggregated, so each of the 4 obs slots covers
  one full approach. In vision mode each ROI id is already approach-level and
  maps to its own slot (unchanged behaviour, now matching training semantics).
- S2 FIX: pressure_norm was abs(sum_qa - sum_qb)/max_lanes — unsigned, so the
  policy could not tell WHICH group was congested. Now signed and group-mean
  based: 0.5*(1 + mean_q(group_a) - mean_q(group_b)), clipped to [0,1].
  0.5 = balanced; >0.5 = group A (phase-0 lanes) more queued.

PATCH NOTES (audit v2):
- FIX: pressure_norm now normalized by max_lanes_per_tls (was queue_cap=50,
  causing pressure to be effectively 0 for all traffic states).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# subscription variable ids (shared by traci and libsumo). If the constants
# module is unavailable the builder transparently uses the legacy per-vehicle
# query path.
try:
    import traci.constants as tc  # type: ignore
except Exception:  # pragma: no cover
    tc = None  # type: ignore[assignment]

from src.traffic_env.config import (
    DEFAULT_LANE_FEATURE_NAMES,
    DEFAULT_MANUAL_LANE_GROUPS,
    DEFAULT_TLS_FEATURE_NAMES,
    TrafficEnvConfig,
)


LaneMetrics = Dict[str, float]
ObsDict = Dict[str, np.ndarray]


# ---------------------------------------------------------------------------
# Vision adapter
# ---------------------------------------------------------------------------

class VisionLaneMetrics:
    """
    Maps one camera frame of YOLO detections + ByteTrack output to the
    same LaneMetrics dict produced by ObservationBuilder._lane_metrics().

    This class has NO SUMO dependency and is safe to instantiate in
    deployment without traci/libsumo.

    COCO class indices used:
        2  = car
        3  = motorcycle  (motorbike)
        5  = bus         (heavy)
        7  = truck       (heavy)

    Call signature
    --------------
    metrics = VisionLaneMetrics.compute(
        tracks          = bytetrack_output_for_this_lane_roi,
        lane_roi_area   = pixel area of the lane polygon,
        lane_length_m   = physical length from calibration map,
        speed_cap_mps   = config.observation.speed_cap,
        px_per_meter    = camera calibration factor,
        fps             = camera frame rate,
    )
    """

    MOTORBIKE_CLASSES: frozenset = frozenset({3})
    HEAVY_CLASSES: frozenset = frozenset({1, 4})      # custom YOLO: 1=bus, 4=truck
    CLASS_LENGTH_M: Dict[int, float] = {3: 1.8, 2: 4.5, 1: 12.0, 4: 8.0}
    DEFAULT_LENGTH_M: float = 4.5
    HALT_SPEED_THRESHOLD_MPS: float = 0.1  # must match StateExtractor.stop_speed_m_s and SUMO _lane_metrics hardcoded 0.1

    @classmethod
    def compute(
        cls,
        tracks: list,
        lane_roi_area: float,
        lane_length_m: float,
        speed_cap_mps: float,
        px_per_meter: float,
        fps: float,
    ) -> LaneMetrics:
        """
        Parameters
        ----------
        tracks : list of ByteTrack STrack objects (or duck-typed dicts)
            Each track must expose:
                .tlwh       -> (x, y, w, h) in image pixels
                .cls        -> int COCO class id
                .velocity   -> np.ndarray([vx, vy]) in px/frame
        lane_roi_area : float
            Pixel area of the lane's region-of-interest polygon.
        lane_length_m : float
            Physical lane length in metres (from prior calibration).
        speed_cap_mps : float
            Normalisation cap for speed (config.observation.speed_cap).
        px_per_meter : float
            Pixels per metre (camera intrinsic calibration).
        fps : float
            Camera acquisition frame rate.
        """
        empty: LaneMetrics = {
            "effective_queue_norm": 0.0,
            "occupancy_norm": 0.0,
            "avg_speed_norm": 0.0,
            "motorbike_share": 0.0,
            "heavy_vehicle_share": 0.0,
            "raw_speed": 0.0,
            "raw_vehicle_count": 0.0,
            "raw_halted_length": 0.0,
            "lane_length": lane_length_m,
            "co2_mg_per_s": 0.0,   # not available from vision
            "fuel_ml_per_s": 0.0,  # not available from vision
        }
        if not tracks:
            return empty

        halted_length          = 0.0
        total_vehicle_length_m = 0.0
        total_speed_mps        = 0.0
        motorbike_count        = 0
        heavy_count            = 0

        for t in tracks:
            try:
                cls_id = int(t.cls if hasattr(t, "cls") else t["cls"])
                vx = float(t.velocity[0] if hasattr(t, "velocity") else t["velocity"][0])
                vy = float(t.velocity[1] if hasattr(t, "velocity") else t["velocity"][1])
            except Exception:
                continue

            speed_px_per_frame = (vx ** 2 + vy ** 2) ** 0.5
            speed_mps = speed_px_per_frame * fps / max(px_per_meter, 1.0)
            total_speed_mps += speed_mps

            veh_length = cls.CLASS_LENGTH_M.get(cls_id, cls.DEFAULT_LENGTH_M)
            total_vehicle_length_m += veh_length
            if speed_mps < cls.HALT_SPEED_THRESHOLD_MPS:
                halted_length += veh_length

            if cls_id in cls.MOTORBIKE_CLASSES:
                motorbike_count += 1
            if cls_id in cls.HEAVY_CLASSES:
                heavy_count += 1

        n = len(tracks)
        avg_speed = total_speed_mps / n
        occupancy = float(np.clip(total_vehicle_length_m / max(lane_length_m, 1.0), 0.0, 1.0))
        effective_queue_norm = float(np.clip(halted_length / max(lane_length_m, 1.0), 0.0, 1.0))

        return {
            "effective_queue_norm": effective_queue_norm,
            "occupancy_norm": occupancy,
            "avg_speed_norm": float(np.clip(avg_speed / max(speed_cap_mps, 1.0), 0.0, 1.0)),
            "motorbike_share": motorbike_count / n,
            "heavy_vehicle_share": heavy_count / n,
            "raw_speed": avg_speed,
            "raw_vehicle_count": float(n),
            "raw_halted_length": halted_length,
            "lane_length": lane_length_m,
            "co2_mg_per_s": 0.0,
            "fuel_ml_per_s": 0.0,
        }


# ---------------------------------------------------------------------------
# Observation builder
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ObservationBuilder:
    """build local observations and global states from sumo lane statistics."""

    config: TrafficEnvConfig
    sumo_conn: Any

    _lane_feature_names: Tuple[str, ...] = field(init=False)
    _tls_feature_names: Tuple[str, ...] = field(init=False)
    _lane_feature_index: Dict[str, int] = field(init=False)
    _tls_feature_index: Dict[str, int] = field(init=False)
    _vehicle_profiles: Dict[str, Tuple[float, float]] = field(init=False)

    # External vision cache: populated by inject_vision_cache()
    _external_lane_cache: Optional[Dict[str, LaneMetrics]] = field(default=None, init=False)

    # fast-metrics state (bulk vehicle subscriptions; see _build_lane_cache_fast)
    _fast_metrics_enabled: bool = field(default=True, init=False)
    _sub_vars: Tuple[int, ...] = field(default=(), init=False)
    _lane_length_cache: Dict[str, float] = field(default_factory=dict, init=False)
    # W5-9: emit PRIVILEGED_EXTRA_LANE_FEATURE_NAMES from exact SUMO state
    _privileged: bool = field(default=False, init=False)
    # VI-F ablation: reproduce the pre-S1 blind-spot obs (raw lanes, no approach aggregation)
    _lane_truncated: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.config.validate()
        self._lane_feature_names = tuple(self.config.observation.lane_feature_names)
        self._tls_feature_names = tuple(self.config.observation.tls_feature_names)
        self._lane_feature_index = {name: idx for idx, name in enumerate(self._lane_feature_names)}
        self._tls_feature_index = {name: idx for idx, name in enumerate(self._tls_feature_names)}
        self._vehicle_profiles = {
            "motorbike": (1.8, 0.7),
            "car": (4.5, 1.8),
            "bus": (12.0, 2.5),
            "truck": (8.0, 2.5),
        }
        self._external_lane_cache = None
        self._privileged = self.config.observation.obs_mode == "privileged"
        self._lane_truncated = bool(getattr(self.config.observation, "lane_truncated", False))
        self._fast_metrics_enabled = tc is not None
        self._lane_length_cache = {}
        if tc is not None:
            self._sub_vars = (
                tc.VAR_SPEED,
                tc.VAR_LENGTH,
                tc.VAR_TYPE,
                tc.VAR_CO2EMISSION,
                tc.VAR_FUELCONSUMPTION,
            )

    # ------------------------------------------------------------------
    # public api
    # ------------------------------------------------------------------
    def set_sumo_conn(self, sumo_conn: Any) -> None:
        """replace the current traci or libsumo connection."""
        self.sumo_conn = sumo_conn
        # new simulation -> static caches invalid, subscriptions cleared by SUMO
        self._lane_length_cache = {}
        self._fast_metrics_enabled = tc is not None

    def inject_vision_cache(self, lane_cache: Dict[str, LaneMetrics]) -> None:
        """
        Inject an externally computed lane cache (from YOLO+ByteTrack).
        After calling this, build_lane_cache() will return this dict instead
        of querying SUMO, as long as use_external_state=True in config.

        Call once per environment step from the vision pipeline, BEFORE calling
        get_all_observations().
        """
        self._external_lane_cache = {k: dict(v) for k, v in lane_cache.items()}

    def build_lane_cache(self, tls_ids: Optional[Sequence[str]] = None) -> Dict[str, LaneMetrics]:
        """
        Collect lane-level statistics once and reuse them across all tls ids.

        When config.observation.use_external_state is True AND inject_vision_cache()
        has been called this step, the external cache is returned directly without
        any SUMO API calls.
        """
        if self.config.observation.use_external_state and self._external_lane_cache is not None:
            return dict(self._external_lane_cache)

        lane_ids: List[str] = []
        for tls_id in self._resolve_tls_ids(tls_ids):
            lane_ids.extend(self._safe_lane_ids_for_tls(tls_id))

        ordered_lane_ids = list(dict.fromkeys(lane_ids))

        # fast path: bulk vehicle subscriptions replace the 5-calls-per-vehicle
        # loop (the dominant wall-clock cost at scale). Falls back permanently
        # to the legacy path on the first failure.
        if self._fast_metrics_enabled:
            try:
                return self._build_lane_cache_fast(ordered_lane_ids)
            except Exception:
                logger.exception(
                    "fast lane metrics failed — falling back to legacy per-vehicle queries"
                )
                self._fast_metrics_enabled = False

        return {lane_id: self._lane_metrics(lane_id) for lane_id in ordered_lane_ids}

    def _build_lane_cache_fast(self, lane_ids: Sequence[str]) -> Dict[str, LaneMetrics]:
        """Subscription-based lane cache, value-identical to _lane_metrics().

        Per env step this issues:  1x getAllSubscriptionResults  +  per lane
        (getLastStepVehicleIDs, getLastStepOccupancy)  +  direct queries ONLY
        for vehicles seen for the first time (which are simultaneously
        subscribed so every later step serves them from the bulk result).
        Lane lengths are static and cached for the connection lifetime.
        """
        conn = self.sumo_conn
        results: Mapping[str, Mapping[int, Any]] = conn.vehicle.getAllSubscriptionResults() or {}
        return {lane_id: self._lane_metrics_from_subs(lane_id, results) for lane_id in lane_ids}

    def _lane_metrics_from_subs(
        self,
        lane_id: str,
        results: Mapping[str, Mapping[int, Any]],
    ) -> LaneMetrics:
        """compute the exact _lane_metrics() output from subscription data."""
        conn = self.sumo_conn

        lane_length = self._lane_length_cache.get(lane_id)
        if lane_length is None:
            try:
                lane_length = max(float(conn.lane.getLength(lane_id)), 1.0)
            except Exception:
                lane_length = 1.0
            self._lane_length_cache[lane_id] = lane_length

        try:
            vehicle_ids: Sequence[str] = list(conn.lane.getLastStepVehicleIDs(lane_id))
        except Exception:
            vehicle_ids = []

        occupancy_raw = 0.0
        try:
            occupancy_raw = float(conn.lane.getLastStepOccupancy(lane_id))
        except Exception:
            pass
        occupancy = occupancy_raw / 100.0 if occupancy_raw > 1.0 else occupancy_raw
        occupancy = self._clamp01(occupancy)

        total_speed = 0.0
        vehicle_count = 0.0
        halted_effective_length = 0.0
        halted_count = 0.0
        motorbike_count = 0.0
        heavy_vehicle_count = 0.0
        co2_total = 0.0
        fuel_total = 0.0

        for vehicle_id in vehicle_ids:
            data = results.get(vehicle_id)
            if data is not None:
                type_id = str(data.get(tc.VAR_TYPE, "") or "")
                length = self._safe_float(data.get(tc.VAR_LENGTH, 0.0))
                speed = self._safe_float(data.get(tc.VAR_SPEED, 0.0))
                co2 = self._safe_float(data.get(tc.VAR_CO2EMISSION, 0.0))
                fuel = self._safe_float(data.get(tc.VAR_FUELCONSUMPTION, 0.0))
            else:
                # first sighting: query directly this step, subscribe for the next
                try:
                    conn.vehicle.subscribe(vehicle_id, list(self._sub_vars))
                except Exception:
                    pass
                try:
                    type_id = str(conn.vehicle.getTypeID(vehicle_id))
                except Exception:
                    type_id = "unknown"
                try:
                    length = float(conn.vehicle.getLength(vehicle_id))
                except Exception:
                    length = 0.0
                try:
                    speed = float(conn.vehicle.getSpeed(vehicle_id))
                except Exception:
                    speed = 0.0
                try:
                    co2 = float(conn.vehicle.getCO2Emission(vehicle_id))
                except Exception:
                    co2 = 0.0
                try:
                    fuel = float(conn.vehicle.getFuelConsumption(vehicle_id))
                except Exception:
                    fuel = 0.0

            category = self._classify_vehicle_type(type_id, length=length)
            if length <= 0:
                length = self._vehicle_profiles[category][0]

            vehicle_count += 1.0
            total_speed += speed
            co2_total += co2
            fuel_total += fuel
            if category == "motorbike":
                motorbike_count += 1.0
            if category in ("bus", "truck"):
                heavy_vehicle_count += 1.0
            if speed <= 0.1:
                halted_effective_length += length
                halted_count += 1.0

        avg_speed = total_speed / max(vehicle_count, 1.0)
        metrics = {
            "effective_queue_norm": self._clamp01(halted_effective_length / lane_length),
            "occupancy_norm": occupancy,
            "avg_speed_norm": self._normalize(avg_speed, self.config.observation.speed_cap),
            "motorbike_share": self._clamp01(motorbike_count / max(vehicle_count, 1.0)),
            "heavy_vehicle_share": self._clamp01(heavy_vehicle_count / max(vehicle_count, 1.0)),
            "raw_speed": float(avg_speed),
            "raw_vehicle_count": float(vehicle_count),
            "raw_halted_length": float(halted_effective_length),
            "lane_length": float(lane_length),
            "co2_mg_per_s": float(co2_total),
            "fuel_ml_per_s": float(fuel_total),
            # 1.2.0: consumed by RewardCalculator local throughput (training only)
            "vehicle_ids": tuple(str(v) for v in vehicle_ids),
        }
        if self._privileged:
            metrics.update(
                self._privileged_lane_features(lane_id, halted_count, vehicle_count)
            )
        return metrics

    def _privileged_lane_features(
        self, lane_id: str, halted_count: float, vehicle_count: float
    ) -> Dict[str, float]:
        """Exact-SUMO per-lane features for the MAPPO-privileged arm (W5-9).

        Namespaced ``privileged_*`` so the RewardCalculator never consumes them
        (the reward must be identical across the privileged and proxy arms — only
        the observation differs). ``getWaitingTime`` is the lane-summed accumulated
        waiting time, queried only in privileged mode to avoid extra TraCI calls on
        the deployable path.
        """
        obs_cfg = self.config.observation
        qcap = max(float(obs_cfg.queue_cap), 1.0)
        wcap = max(float(obs_cfg.waiting_cap), 1.0)
        try:
            waiting = float(self.sumo_conn.lane.getWaitingTime(lane_id))
        except Exception:
            waiting = 0.0
        return {
            "privileged_halt_count_norm": self._clamp01(halted_count / qcap),
            "privileged_waiting_norm": self._clamp01(waiting / wcap),
            "privileged_vehicle_count_norm": self._clamp01(vehicle_count / qcap),
        }

    def build_local_obs(
        self,
        tls_id: str,
        lane_cache: Mapping[str, Mapping[str, float]],
        green_timers: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """build a fixed-size local observation for one traffic light system.

        S1: each obs slot is one APPROACH (all lanes of one incoming edge,
        aggregated), not one raw lane — so all controlled lanes contribute.
        """
        obs: List[float] = []
        approaches = self._approach_groups_for_tls(tls_id)[: self.config.max_lanes_per_tls]

        for _approach_key, group_lane_ids in approaches:
            agg = self._aggregate_approach_metrics(group_lane_ids, lane_cache)
            for feature_name in self._lane_feature_names:
                obs.append(agg.get(feature_name, 0.0))

        missing_lanes = self.config.max_lanes_per_tls - len(approaches)
        if missing_lanes > 0:
            obs.extend([0.0] * (missing_lanes * self.config.lane_feature_dim))
        obs.extend(self._tls_features(tls_id, lane_cache, green_timers))

        arr = np.asarray(obs, dtype=np.float32)
        expected = self.config.local_obs_dim
        if arr.shape != (expected,):
            raise RuntimeError(
                f"local observation shape mismatch for tls_id={tls_id!r}: "
                f"got {arr.shape}, expected {(expected,)}"
            )
        return arr

    def build_global_state(self, obs_dict: Mapping[str, np.ndarray]) -> np.ndarray:
        """concatenate all local observations into a single critic state."""
        chunks: List[np.ndarray] = []
        for tls_id in self._resolve_tls_ids(None):
            local_obs = obs_dict.get(tls_id)
            if local_obs is None:
                chunks.append(np.zeros(self.config.local_obs_dim, dtype=np.float32))
                continue

            arr = np.asarray(local_obs, dtype=np.float32).reshape(-1)
            if arr.size != self.config.local_obs_dim:
                padded = np.zeros(self.config.local_obs_dim, dtype=np.float32)
                limit = min(arr.size, self.config.local_obs_dim)
                padded[:limit] = arr[:limit]
                arr = padded
            chunks.append(arr)

        if not chunks:
            return np.zeros(self.config.global_state_dim, dtype=np.float32)

        state = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
        expected = self.config.global_state_dim
        if state.size != expected:
            padded = np.zeros(expected, dtype=np.float32)
            limit = min(state.size, expected)
            padded[:limit] = state[:limit]
            state = padded
        return state

    def get_all_observations(
        self,
        lane_cache: Mapping[str, Mapping[str, float]],
        green_timers: Optional[Dict[str, float]] = None
    ) -> Tuple[ObsDict, np.ndarray]:
        obs_dict: ObsDict = {}
        for tls_id in self._resolve_tls_ids(None):
            obs_dict[tls_id] = self.build_local_obs(tls_id, lane_cache, green_timers)
        global_state = self.build_global_state(obs_dict)
        return obs_dict, global_state

    # ------------------------------------------------------------------
    # lane and tls helpers
    # ------------------------------------------------------------------
    def _resolve_tls_ids(self, tls_ids: Optional[Sequence[str]]) -> Tuple[str, ...]:
        if tls_ids is not None:
            return tuple(tls_ids)
        return tuple(self.config.tls_ids)

    def _safe_lane_ids_for_tls(self, tls_id: str) -> List[str]:
        """return controlled lanes in sumo order, without internal lane ids."""
        if self.config.observation.use_external_state and self._external_lane_cache is not None:
            # In vision mode, derive lane list from manual lane groups in config.
            groups = self.config.get_lane_groups()
            if tls_id in groups:
                group_a, group_b = groups[tls_id]
                combined = list(dict.fromkeys(group_a + group_b))
                return combined
            return []

        try:
            lanes = self.sumo_conn.trafficlight.getControlledLanes(tls_id)
        except Exception:
            lanes = []
        cleaned = [lane_id for lane_id in dict.fromkeys(lanes) if lane_id and not str(lane_id).startswith(":")]
        return list(cleaned)

    def _approach_groups_for_tls(self, tls_id: str) -> List[Tuple[str, List[str]]]:
        """group controlled lanes into approaches, one group per incoming edge.

        SUMO lane ids follow '<edge_id>_<lane_index>'; lanes sharing a parent
        edge form one approach. In external/vision mode each cache key is
        already an approach-level ROI, so every id maps to its own group.
        """
        lane_ids = self._safe_lane_ids_for_tls(tls_id)
        if self.config.observation.use_external_state and self._external_lane_cache is not None:
            return [(lane_id, [lane_id]) for lane_id in lane_ids]

        # VI-F blind-spot ablation: raw lanes as individual slots (no approach
        # aggregation), so build_local_obs sees only the first max_lanes lanes.
        if self._lane_truncated:
            return [(lane_id, [lane_id]) for lane_id in lane_ids[: self.config.max_lanes_per_tls]]

        groups: Dict[str, List[str]] = {}
        order: List[str] = []
        for lane_id in lane_ids:
            edge_key = lane_id.rsplit("_", 1)[0] if "_" in lane_id else lane_id
            if edge_key not in groups:
                groups[edge_key] = []
                order.append(edge_key)
            groups[edge_key].append(lane_id)
        return [(edge_key, groups[edge_key]) for edge_key in order]

    def _aggregate_approach_metrics(
        self,
        lane_ids: Sequence[str],
        lane_cache: Mapping[str, Mapping[str, float]],
    ) -> Dict[str, float]:
        """aggregate per-lane metrics into one approach-level feature set.

        queue/occupancy are length-ratio features -> unweighted mean across
        lanes. speed and class shares are per-vehicle statistics -> vehicle-
        count-weighted mean, so empty lanes do not dilute them.
        """
        if not lane_ids:
            return {name: 0.0 for name in self._lane_feature_names}
        if len(lane_ids) == 1:
            metrics = lane_cache.get(lane_ids[0], {})
            return {
                name: self._safe_float(metrics.get(name, 0.0))
                for name in self._lane_feature_names
            }

        per_lane = [lane_cache.get(lane_id, {}) for lane_id in lane_ids]
        counts = [
            max(self._safe_float(m.get("raw_vehicle_count", 0.0)), 0.0)
            for m in per_lane
        ]
        total_count = sum(counts)
        count_weighted = {"avg_speed_norm", "motorbike_share", "heavy_vehicle_share"}

        agg: Dict[str, float] = {}
        for name in self._lane_feature_names:
            vals = [self._safe_float(m.get(name, 0.0)) for m in per_lane]
            if name in count_weighted and total_count > 0:
                value = sum(v * c for v, c in zip(vals, counts)) / total_count
            else:
                value = float(np.mean(vals))
            agg[name] = self._clamp01(value)
        return agg

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return float(default)
            return float(value)
        except Exception:
            return float(default)

    def _normalize(self, value: float, cap: float) -> float:
        if cap <= 0:
            return 0.0
        return float(np.clip(value / cap, 0.0, 1.0))

    def _clamp01(self, value: float) -> float:
        return float(np.clip(value, 0.0, 1.0))

    def _phase_is_green(self, state: str) -> bool:
        return any(c in state for c in "Gg") and ("y" not in state.lower())

    def _phase_is_yellow(self, state: str) -> bool:
        return "y" in state.lower()

    def _get_lane_direction_class(self, lane_id: str) -> str:
        """classify a lane as ns or ew from its geometry."""
        try:
            shape = self.sumo_conn.lane.getShape(lane_id)
            if len(shape) < 2:
                return "EW"
            x1, y1 = shape[0]
            x2, y2 = shape[-1]
            dx = x2 - x1
            dy = y2 - y1
            if abs(dx) + abs(dy) < 1e-6:
                return "EW"
            angle = float(np.degrees(np.arctan2(dy, dx)) % 360.0)
            if 45.0 <= angle < 135.0 or 225.0 <= angle < 315.0:
                return "NS"
            return "EW"
        except Exception:
            return "EW"

    def _get_tls_group_lanes(self, tls_id: str) -> Tuple[List[str], List[str]]:
        """return two opposing lane groups for a tls."""
        manual_groups = self.config.manual_lane_groups or DEFAULT_MANUAL_LANE_GROUPS
        if tls_id in manual_groups:
            group_a, group_b = manual_groups[tls_id]
            return list(group_a), list(group_b)

        lane_ids = self._safe_lane_ids_for_tls(tls_id)
        ns_lanes: List[str] = []
        ew_lanes: List[str] = []
        for lane_id in lane_ids:
            if self._get_lane_direction_class(lane_id) == "NS":
                ns_lanes.append(lane_id)
            else:
                ew_lanes.append(lane_id)

        if ns_lanes and ew_lanes:
            return ns_lanes, ew_lanes

        groups: Dict[str, List[str]] = {}
        for lane_id in lane_ids:
            edge_key = lane_id.split("_")[0] if "_" in lane_id else lane_id
            groups.setdefault(edge_key, []).append(lane_id)

        ordered_groups = list(groups.values())
        if not ordered_groups:
            return [], []
        if len(ordered_groups) == 1:
            return ordered_groups[0], []
        return ordered_groups[0], ordered_groups[1]

    def _build_signal_maps(self, tls_id: str) -> Tuple[List[int], Dict[int, int]]:
        """build green-phase and yellow-phase mapping for a tls."""
        fallback_phase_map = [0, 2]
        fallback_yellow_map = {0: 1, 2: 3}

        try:
            logics = self.sumo_conn.trafficlight.getAllProgramLogics(tls_id)
            if not logics:
                return fallback_phase_map, fallback_yellow_map

            phases = list(logics[0].phases)
            if not phases:
                return fallback_phase_map, fallback_yellow_map

            green_candidates: List[int] = []
            green_masks: Dict[int, Tuple[bool, ...]] = {}
            for idx, phase in enumerate(phases):
                if self._phase_is_green(phase.state):
                    green_candidates.append(idx)
                    green_masks[idx] = tuple(c in "Gg" for c in phase.state)

            if len(green_candidates) < 2:
                return fallback_phase_map, fallback_yellow_map

            best_pair = (green_candidates[0], green_candidates[1])
            best_score = -1
            for i in range(len(green_candidates)):
                for j in range(i + 1, len(green_candidates)):
                    a = green_candidates[i]
                    b = green_candidates[j]
                    mask_a = green_masks[a]
                    mask_b = green_masks[b]
                    n = min(len(mask_a), len(mask_b))
                    dist = sum(1 for k in range(n) if mask_a[k] != mask_b[k]) + abs(len(mask_a) - len(mask_b))
                    green_strength = sum(mask_a) + sum(mask_b)
                    score = dist * 1000 + green_strength
                    if score > best_score:
                        best_score = score
                        best_pair = (a, b)

            phase_map = [best_pair[0], best_pair[1]]
            yellow_map: Dict[int, int] = {}
            for green_idx in phase_map:
                yellow_idx: Optional[int] = None
                for idx in range(green_idx + 1, len(phases)):
                    if self._phase_is_yellow(phases[idx].state):
                        yellow_idx = idx
                        break
                    if self._phase_is_green(phases[idx].state):
                        break
                if yellow_idx is None:
                    if green_idx + 1 < len(phases) and self._phase_is_yellow(phases[green_idx + 1].state):
                        yellow_idx = green_idx + 1
                    else:
                        yellow_idx = (green_idx + 1) % len(phases)
                yellow_map[green_idx] = yellow_idx

            return phase_map, yellow_map
        except Exception:
            return fallback_phase_map, fallback_yellow_map

    def _phase_one_hot(self, tls_id: str) -> List[float]:
        """encode the current tls phase as a 4-d one-hot vector."""
        try:
            phase = int(self.sumo_conn.trafficlight.getPhase(tls_id))
        except Exception:
            phase = 0
        phase = int(np.clip(phase, 0, 3))
        vec = [0.0, 0.0, 0.0, 0.0]
        vec[phase] = 1.0
        return vec

    def _get_vehicle_geometry(self, vehicle_id: str) -> Tuple[float, float, str]:
        """retrieve vehicle geometry and classify it into a traffic class."""
        type_id = "unknown"
        try:
            type_id = str(self.sumo_conn.vehicle.getTypeID(vehicle_id))
        except Exception:
            pass

        length = 0.0
        width = 0.0
        try:
            length = float(self.sumo_conn.vehicle.getLength(vehicle_id))
        except Exception:
            pass
        try:
            width = float(self.sumo_conn.vehicle.getWidth(vehicle_id))
        except Exception:
            pass

        category = self._classify_vehicle_type(type_id, length=length)
        default_length, default_width = self._vehicle_profiles[category]
        if length <= 0:
            length = default_length
        if width <= 0:
            width = default_width
        return length, width, category

    def _classify_vehicle_type(self, type_id: str, length: float = 0.0) -> str:
        """classify a vehicle into a supported vietnamese traffic class."""
        t = (type_id or "").lower()
        if any(k in t for k in ("motor", "bike", "moto", "scooter", "twowheel", "two_wheel")):
            return "motorbike"
        if any(k in t for k in ("bus", "coach")):
            return "bus"
        if any(k in t for k in ("truck", "lorry", "heavy")):
            return "truck"
        if any(k in t for k in ("car", "sedan", "taxi", "van", "passenger", "auto")):
            return "car"

        if length > 0:
            if length <= 2.2:
                return "motorbike"
            if length <= 5.5:
                return "car"
            if length <= 9.0:
                return "truck"
            return "bus"
        return "car"

    def _lane_metrics(self, lane_id: str) -> LaneMetrics:
        """compute normalized lane statistics with safe fallbacks."""
        lane_length = 1.0
        try:
            lane_length = max(float(self.sumo_conn.lane.getLength(lane_id)), 1.0)
        except Exception:
            pass

        vehicle_ids: Sequence[str]
        try:
            vehicle_ids = list(self.sumo_conn.lane.getLastStepVehicleIDs(lane_id))
        except Exception:
            vehicle_ids = []

        occupancy_raw = 0.0
        try:
            occupancy_raw = float(self.sumo_conn.lane.getLastStepOccupancy(lane_id))
        except Exception:
            pass
        occupancy = occupancy_raw / 100.0 if occupancy_raw > 1.0 else occupancy_raw
        occupancy = self._clamp01(occupancy)

        total_speed = 0.0
        vehicle_count = 0.0
        halted_effective_length = 0.0
        halted_count = 0.0
        motorbike_count = 0.0
        heavy_vehicle_count = 0.0
        co2_total = 0.0
        fuel_total = 0.0

        for vehicle_id in vehicle_ids:
            length, _width, category = self._get_vehicle_geometry(vehicle_id)
            vehicle_count += 1.0

            try:
                speed = float(self.sumo_conn.vehicle.getSpeed(vehicle_id))
            except Exception:
                speed = 0.0
            total_speed += speed

            # CO2 (mg/s) and fuel (ml/s) per vehicle
            try:
                co2_total += float(self.sumo_conn.vehicle.getCO2Emission(vehicle_id))
            except Exception:
                pass
            try:
                fuel_total += float(self.sumo_conn.vehicle.getFuelConsumption(vehicle_id))
            except Exception:
                pass

            if category == "motorbike":
                motorbike_count += 1.0
            if category in ("bus", "truck"):
                heavy_vehicle_count += 1.0
            if speed <= 0.1:
                halted_effective_length += length
                halted_count += 1.0

        avg_speed = total_speed / max(vehicle_count, 1.0)
        effective_queue_norm = self._clamp01(halted_effective_length / lane_length)
        motorbike_share = self._clamp01(motorbike_count / max(vehicle_count, 1.0))
        heavy_vehicle_share = self._clamp01(heavy_vehicle_count / max(vehicle_count, 1.0))

        metrics = {
            "effective_queue_norm": effective_queue_norm,
            "occupancy_norm": occupancy,
            "avg_speed_norm": self._normalize(avg_speed, self.config.observation.speed_cap),
            "motorbike_share": motorbike_share,
            "heavy_vehicle_share": heavy_vehicle_share,
            "raw_speed": float(avg_speed),
            "raw_vehicle_count": float(vehicle_count),
            "raw_halted_length": float(halted_effective_length),
            "lane_length": float(lane_length),
            "co2_mg_per_s": float(co2_total),
            "fuel_ml_per_s": float(fuel_total),
            # 1.2.0: consumed by RewardCalculator local throughput (training only)
            "vehicle_ids": tuple(str(v) for v in vehicle_ids),
        }
        if self._privileged:
            metrics.update(
                self._privileged_lane_features(lane_id, halted_count, vehicle_count)
            )
        return metrics

    def _tls_pressure_proxy(self, tls_id: str, lane_cache: Mapping[str, Mapping[str, float]]) -> float:
        """signed pressure feature in [0,1] (S2 fix).

        p = mean_q(group_a) - mean_q(group_b) in [-1, 1]; feature = (p+1)/2.
        0.5 = balanced; >0.5 = group A (phase-0 lanes) more queued. Group MEANS
        (not sums) keep the value invariant to group size. Must stay formula-
        identical to StateExtractor._tls_pressure_proxy (deployment side).
        """
        group_a, group_b = self._get_tls_group_lanes(tls_id)
        qa = sum(
            self._safe_float(lane_cache.get(lane_id, {}).get("effective_queue_norm", 0.0))
            for lane_id in group_a
        ) / max(len(group_a), 1)
        qb = sum(
            self._safe_float(lane_cache.get(lane_id, {}).get("effective_queue_norm", 0.0))
            for lane_id in group_b
        ) / max(len(group_b), 1)
        return float(np.clip(0.5 * (1.0 + qa - qb), 0.0, 1.0))

    def _tls_group_queues(self, tls_id: str, lane_cache: Mapping[str, Mapping[str, float]]) -> Tuple[float, float]:
        """return normalized queue totals for the two opposing groups."""
        group_a, group_b = self._get_tls_group_lanes(tls_id)
        qa = sum(self._safe_float(lane_cache.get(lane_id, {}).get("effective_queue_norm", 0.0)) for lane_id in group_a)
        qb = sum(self._safe_float(lane_cache.get(lane_id, {}).get("effective_queue_norm", 0.0)) for lane_id in group_b)
        return float(qa), float(qb)

    def _tls_features(
        self,
        tls_id: str,
        lane_cache: Mapping[str, Mapping[str, float]],
        green_timers: Optional[Dict[str, float]] = None
    ) -> List[float]:
        """build tls-level features in the exact config-defined order."""
        features = [0.0] * self.config.tls_feature_dim
        feature_map: Dict[str, float] = {name: 0.0 for name in self._tls_feature_names}

        phase_vec = self._phase_one_hot(tls_id)
        for idx, value in enumerate(phase_vec):
            key = f"phase_one_hot_{idx}"
            if key in feature_map:
                feature_map[key] = float(value)

        green_timer = 0.0
        if green_timers is not None:
            green_timer = float(green_timers.get(tls_id, 0.0))

        feature_map["green_timer_norm"] = self._normalize(green_timer, self.config.sim.max_green_time)

        # S2: signed pressure proxy already returns a [0,1] feature
        # (0.5 = balanced) — no further cap normalisation.
        feature_map["pressure_norm"] = self._tls_pressure_proxy(tls_id, lane_cache)

        for name, idx in self._tls_feature_index.items():
            if idx < len(features):
                features[idx] = self._safe_float(feature_map.get(name, 0.0))
        return features


__all__ = ["ObservationBuilder", "VisionLaneMetrics", "LaneMetrics", "ObsDict"]