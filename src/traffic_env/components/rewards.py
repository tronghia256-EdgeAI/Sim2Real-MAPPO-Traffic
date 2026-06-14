from __future__ import annotations

"""
rewards.py
==========
Per-agent reward calculator for MAPPO traffic-signal control.

Reward signal (reward revision 1.2.0)
-------------------------------------
R(t) = w_q  · queue_penalty(t)      — mean of SQUARED lane queues    [0, 1]
     + w_pr · pressure_penalty(t)   — SIGNED phase-aware imbalance   [-1, 1]
     + w_t  · throughput_reward(t)  — LOCAL lane-exit count          [0, 1]
     + w_s  · switch_penalty(t)     — phase-switching cost           [0, ∞)
     + w_sp · low_speed_penalty(t)  — mean speed below threshold     [0, thr]
     + w_w  · waiting_penalty(t)    — accumulated delay              [0, 1]

With default weights (queue=-1, pressure=-0.5, throughput=+1, switch=-0.1,
low_speed=-0.2, waiting=-0.3) the raw range is [-1.94, +1.50], matching the
clip range [-2.0, +1.5] in RewardConfig.

1.2.0 changes (W5 campaign revision)
------------------------------------
- queue_penalty: mean(q)^2 -> mean(q^2). Jensen: mean(q^2) >= mean(q)^2 with
  equality only for uniform queues — a single starved/saturated lane can no
  longer hide behind a low average (anti-starvation).
- pressure_penalty: one-sided max(red-green, 0) -> SIGNED
  (sum_red - sum_green) / n_total in [-1, 1]. With a negative weight the
  agent is penalised for holding green on the emptier side AND rewarded for
  serving the more congested side — restoring gradient on correct decisions.
- throughput_reward: global network arrival delta (split across agents,
  high variance, diluted credit) -> LOCAL per-agent lane-exit count from
  per-lane vehicle-ID set differences. This also aligns training exactly
  with the documented vision proxy (ROI exit events). The global-delta path
  is kept as an automatic fallback when lane vehicle IDs are unavailable.

Sim-to-real alignment (SUMO → YOLO + ByteTrack)
------------------------------------------------
queue_penalty      effective_queue_norm  → halted-vehicle fraction in ROI
pressure_penalty   effective_queue_norm  → same, partitioned by phase group
throughput_reward  lane vehicle-ID exits → track IDs exiting ROI per step
switch_penalty     action logic          → no vision dependency
waiting_penalty    waiting_time_norm     → fallback: effective_queue_norm
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.traffic_env.config import TrafficEnvConfig

logger = logging.getLogger(__name__)

LaneMetrics = Mapping[str, Any]
LaneCache   = Mapping[str, LaneMetrics]
ActionInfo  = Mapping[str, Any]
RewardDict  = Dict[str, float]


@dataclass(slots=True)
class RewardCalculator:
    """Compute per-agent rewards from lane-level traffic statistics.

    Parameters
    ----------
    config : TrafficEnvConfig
        Full environment configuration.  ``reward.weights``,
        ``reward.reward_scale``, and ``reward.reward_clip_*`` control shaping.

    Internal state
    --------------
    _prev_passed_total : float
        Running cumulative arrived-vehicle count.  Used only when the caller
        does not supply ``prev_passed`` in ``action_info``.  Reset via
        :meth:`reset` at episode boundaries.
    last_reward_details : dict
        Per-step component breakdown, keyed by ``tls_id``.  Written after
        every :meth:`calculate_rewards` call; useful for TensorBoard logging.
    """

    config: TrafficEnvConfig
    _prev_passed_total: float = 0.0
    last_reward_details: Dict[str, Dict[str, float]] = field(
        default_factory=dict, init=False
    )
    # per-lane vehicle-ID sets from the previous step (local throughput, 1.2.0)
    _prev_lane_vehicle_ids: Dict[str, frozenset] = field(
        default_factory=dict, init=False
    )

    def __post_init__(self) -> None:
        self.config.validate()
        self.reset()

    def reset(self, prev_passed_total: float = 0.0) -> None:
        """Reset internal throughput trackers between episodes."""
        self._prev_passed_total = float(prev_passed_total)
        self.last_reward_details = {}
        self._prev_lane_vehicle_ids = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_rewards(
        self,
        tls_ids: Sequence[str],
        lane_cache: LaneCache,
        action_info: ActionInfo,
    ) -> RewardDict:
        """Calculate a scalar reward for each TLS agent.

        Expected ``action_info`` keys
        -----------------------------
        ``lane_ids_by_tls`` / ``controlled_lanes_dict``
            ``{tls_id: [lane_id, ...]}``  — all controlled lanes per agent.
        ``green_lanes_by_tls``  *(optional)*
            ``{tls_id: [lane_id, ...]}``  — lanes currently showing green.
            Required to enable ``pressure_penalty``; omit to disable it.
        ``current_passed`` / ``passed``
            Cumulative arrived-vehicle count at this step.
        ``prev_passed``  *(optional)*
            Previous cumulative count.  If absent, internal tracker is used.
        ``delta_passed``  *(optional)*
            Pre-computed throughput delta; overrides current/prev when present.
        ``switches_by_tls``
            ``{tls_id: bool|int}``  — phase switches executed this step.
        ``num_switches``
            Shared integer switch count distributed across all agents.
        ``step_length_seconds``  *(optional, unused — kept for compatibility)*
            Previously used for waiting-time scaling; no longer applied.
        """
        tls_ids = tuple(tls_ids)
        if not tls_ids:
            return {}

        lane_ids_by_tls  = self._resolve_lane_ids_by_tls(tls_ids, action_info)
        green_ids_by_tls = self._resolve_green_lane_ids(tls_ids, action_info, lane_ids_by_tls)
        throughput_delta = self._resolve_throughput_delta(action_info)
        switch_counts    = self._resolve_switch_counts(tls_ids, action_info)

        # 1.2.0: snapshot current per-lane vehicle-ID sets (local throughput).
        # has_lane_ids is False when the cache carries no ID data (e.g. unit
        # tests with synthetic metrics) -> global-delta fallback is used.
        current_ids_by_lane: Dict[str, frozenset] = {}
        for tls_id in tls_ids:
            for lane_id in lane_ids_by_tls.get(tls_id, ()):
                if lane_id in current_ids_by_lane:
                    continue
                raw_ids = lane_cache.get(lane_id, {}).get("vehicle_ids")
                if raw_ids is not None:
                    current_ids_by_lane[lane_id] = frozenset(str(v) for v in raw_ids)
        has_lane_ids = bool(current_ids_by_lane)

        rewards: RewardDict = {}
        details: Dict[str, Dict[str, float]] = {}

        for tls_id in tls_ids:
            lane_ids  = lane_ids_by_tls.get(tls_id, ())
            green_ids = green_ids_by_tls.get(tls_id)   # None → pressure disabled
            red_ids   = green_ids_by_tls.get(f"_red_{tls_id}")

            queue_p      = self._compute_queue_penalty(tls_id, lane_ids, lane_cache)
            pressure_p   = self._compute_pressure_penalty(tls_id, green_ids, red_ids, lane_cache)
            if has_lane_ids:
                throughput_r = self._compute_local_throughput(tls_id, lane_ids, current_ids_by_lane)
            else:
                throughput_r = self._compute_throughput_reward(tls_id, throughput_delta, len(tls_ids))
            switch_p     = self._compute_switch_penalty(tls_id, switch_counts)
            low_speed_p  = self._compute_low_speed_penalty(tls_id, lane_ids, lane_cache)
            waiting_p    = self._compute_waiting_penalty(tls_id, lane_ids, lane_cache)

            w = self.config.reward.weights
            raw = (
                self._weight(w, "queue")           * queue_p
                + self._weight(w, "pressure")      * pressure_p
                + self._weight(w, "throughput")    * throughput_r
                + self._weight(w, "switch_penalty")* switch_p
                + self._weight(w, "low_speed_penalty") * low_speed_p
                + self._weight(w, "waiting_time")  * waiting_p
            )

            scaled  = float(raw * self.config.reward.reward_scale)
            clipped = float(
                np.clip(scaled,
                        self.config.reward.reward_clip_low,
                        self.config.reward.reward_clip_high)
            )

            rewards[tls_id] = clipped
            details[tls_id] = {
                "queue_penalty":     queue_p,
                "pressure_penalty":  pressure_p,
                "throughput_reward": throughput_r,
                "switch_penalty":    switch_p,
                "low_speed_penalty": low_speed_p,
                "waiting_penalty":   waiting_p,
                "raw_reward":        raw,
                "scaled_reward":     scaled,
                "clipped_reward":    clipped,
            }
            logger.debug(
                "[%s] r=%.4f  q=%.3f pr=%.3f tp=%.3f sw=%.3f sp=%.3f wt=%.3f",
                tls_id, clipped, queue_p, pressure_p, throughput_r,
                switch_p, low_speed_p, waiting_p,
            )

        if has_lane_ids:
            self._prev_lane_vehicle_ids = current_ids_by_lane

        self.last_reward_details = details
        return rewards

    # ------------------------------------------------------------------
    # Reward components
    # ------------------------------------------------------------------

    def _compute_queue_penalty(
        self,
        tls_id: str,
        lane_ids: Sequence[str],
        lane_cache: LaneCache,
    ) -> float:
        """Nonlinear congestion penalty across all controlled lanes.

        Formula (1.2.0): ``mean(effective_queue_norm ** 2)``  — mean of
        squares, NOT square of the mean. By Jensen's inequality
        mean(q^2) >= mean(q)^2 with equality only when all lanes are equal,
        so one saturated lane among many empty ones is penalised heavily
        instead of being averaged away (anti-starvation guard). Output stays
        in [0, 1] and keeps the near-zero free-flow / strong-congestion
        gradient profile of the previous formulation.

        Returns
        -------
        float in [0, 1]
        """
        if not lane_ids:
            logger.debug("[%s] queue_penalty: no lane_ids → 0.0", tls_id)
            return 0.0

        q_vals = [
            self._safe_metric(lane_cache.get(lid, {}), "effective_queue_norm")
            for lid in lane_ids
        ]
        # 1.2.0 default: mean(q^2) (anti-starvation). Ablation queue_mean_of_squares=False
        # reverts to the pre-1.2.0 mean(q)^2 (single-lane starvation averaged away).
        if self.config.reward.queue_mean_of_squares:
            value = float(np.mean(np.square(q_vals)))
        else:
            value = float(np.square(np.mean(q_vals)))
        result = float(np.clip(value, 0.0, 1.0))
        logger.debug(
            "[%s] queue_penalty=%.4f (mean_q=%.4f)",
            tls_id, result, float(np.mean(q_vals)),
        )
        return result

    def _compute_pressure_penalty(
        self,
        tls_id: str,
        green_lane_ids: Optional[Sequence[str]],
        red_lane_ids: Optional[Sequence[str]],
        lane_cache: LaneCache,
    ) -> float:
        """SIGNED phase-aware queue-imbalance term (1.2.0, PRESSLIGHT-inspired).

        Pressure = (Σ queue_norm over red lanes − Σ queue_norm over green lanes)
                   / total_lanes              ∈ [-1, 1]

        Positive  → the currently-red lanes carry more queue than the green
                    ones: green is being held on the wrong direction. With
                    the negative weight (-0.5) this is a penalty.
        Negative  → green is correctly serving the more congested side; the
                    negative weight turns this into a positive reward, giving
                    gradient on CORRECT decisions too (the previous one-sided
                    max(·, 0) form was silent whenever allocation was right).

        Returns 0.0 (neutral) when phase-lane information is absent from
        ``action_info``.

        Returns
        -------
        float in [-1, 1]
        """
        if not green_lane_ids or not red_lane_ids:
            logger.debug("[%s] pressure_penalty: phase lanes unavailable → 0.0", tls_id)
            return 0.0

        sum_green = sum(
            self._safe_metric(lane_cache.get(lid, {}), "effective_queue_norm")
            for lid in green_lane_ids
        )
        sum_red = sum(
            self._safe_metric(lane_cache.get(lid, {}), "effective_queue_norm")
            for lid in red_lane_ids
        )
        n_total = max(len(green_lane_ids) + len(red_lane_ids), 1)
        raw = (sum_red - sum_green) / n_total
        # 1.2.0 default: SIGNED in [-1,1] (gradient on correct decisions too).
        # Ablation pressure_signed=False reverts to the pre-1.2.0 one-sided
        # max(red-green, 0) (zero gradient whenever allocation is already correct).
        if not self.config.reward.pressure_signed:
            raw = max(raw, 0.0)
        result = float(np.clip(raw, -1.0, 1.0))
        logger.debug(
            "[%s] pressure_penalty=%.4f (red=%.3f green=%.3f n=%d signed=%s)",
            tls_id, result, sum_red, sum_green, n_total, self.config.reward.pressure_signed,
        )
        return result

    def _compute_local_throughput(
        self,
        tls_id: str,
        lane_ids: Sequence[str],
        current_ids_by_lane: Mapping[str, frozenset],
    ) -> float:
        """LOCAL per-agent throughput from lane vehicle-ID exits (1.2.0).

        exits = Σ over this agent's lanes of |prev_step_ids \\ current_ids| —
        vehicles that left the approach lane since the last decision step
        (crossed the stop line, modulo lane-change noise). Normalised by
        ``throughput_norm_divisor`` and clipped to [0, 1]. No cross-agent
        splitting: each agent is credited only for traffic it served, which
        is also exactly the deployable vision proxy (ByteTrack ROI exits).

        Returns
        -------
        float in [0, 1]
        """
        exits = 0
        for lane_id in lane_ids:
            prev = self._prev_lane_vehicle_ids.get(lane_id)
            if prev is None:
                continue
            cur = current_ids_by_lane.get(lane_id, frozenset())
            exits += len(prev - cur)

        norm = exits / max(self.config.reward.throughput_norm_divisor, 1.0)
        result = float(np.clip(norm, 0.0, 1.0))
        logger.debug("[%s] local_throughput=%.4f (exits=%d)", tls_id, result, exits)
        return result

    def _compute_throughput_reward(
        self,
        tls_id: str,
        throughput_delta: float,
        num_tls: int,
    ) -> float:
        """FALLBACK global cleared-vehicles reward (pre-1.2.0 behaviour).

        Used only when the lane cache carries no per-lane vehicle IDs (e.g.
        synthetic test fixtures or external callers). The global throughput
        increment is split equally across agents — high variance and diluted
        credit; prefer the local path. Normalised by
        ``config.reward.throughput_norm_divisor``.

        Returns
        -------
        float in [0, 1]
        """
        if num_tls <= 0:
            logger.debug("[%s] throughput_reward: num_tls=0 → 0.0", tls_id)
            return 0.0

        norm   = throughput_delta / max(self.config.reward.throughput_norm_divisor, 1.0)
        result = float(np.clip(norm, 0.0, 1.0)) / float(num_tls)
        logger.debug("[%s] throughput_reward=%.4f (delta=%.1f)", tls_id, result, throughput_delta)
        return result

    def _compute_switch_penalty(
        self,
        tls_id: str,
        switch_counts: Mapping[str, float],
    ) -> float:
        """Phase-switching cost for this agent.

        Returns the raw switch count (typically 0 or 1 per step); the scale is
        controlled by the corresponding weight in ``RewardConfig.weights``.

        Returns
        -------
        float >= 0
        """
        result = max(self._safe_float(switch_counts.get(tls_id, 0.0)), 0.0)
        logger.debug("[%s] switch_penalty=%.4f", tls_id, result)
        return result

    def _compute_low_speed_penalty(
        self,
        tls_id: str,
        lane_ids: Sequence[str],
        lane_cache: LaneCache,
    ) -> float:
        """Penalise mean lane speed below the configured threshold.

        Uses ``avg_speed_norm`` (speed / speed_cap) from lane_cache.  In vision
        mode this is estimated from ByteTrack pixel displacement and is noisier
        than SUMO — consider reducing the weight or disabling via weight=0 when
        deploying with YOLO + ByteTrack.

        Returns
        -------
        float in [0, config.reward.low_speed_threshold]
        """
        if not lane_ids:
            logger.debug("[%s] low_speed_penalty: no lane_ids → 0.0", tls_id)
            return 0.0

        speed_vals = [
            self._safe_metric(lane_cache.get(lid, {}), "avg_speed_norm")
            for lid in lane_ids
        ]
        mean_speed = float(np.mean(speed_vals))
        threshold  = float(self.config.reward.low_speed_threshold)
        result     = max(threshold - mean_speed, 0.0)
        logger.debug("[%s] low_speed_penalty=%.4f (mean_spd=%.4f)", tls_id, result, mean_speed)
        return result

    def _compute_waiting_penalty(
        self,
        tls_id: str,
        lane_ids: Sequence[str],
        lane_cache: LaneCache,
    ) -> float:
        """Accumulated-delay penalty, vision-aligned.

        Reads ``waiting_time_norm`` (SUMO per-vehicle timers / waiting_cap) when
        available.  Falls back to ``effective_queue_norm`` as a proportional
        proxy; the halted-vehicle fraction is directly measurable from
        ByteTrack without per-vehicle counters.

        The previous implementation multiplied by ``step_length / max_green_time``
        which capped the output at ≤ 0.083 and rendered the term negligible.
        That scaling is removed — both paths now return a value in [0, 1].

        Returns
        -------
        float in [0, 1]
        """
        if not lane_ids:
            logger.debug("[%s] waiting_penalty: no lane_ids → 0.0", tls_id)
            return 0.0

        scores: list[float] = []
        for lid in lane_ids:
            m  = lane_cache.get(lid, {})
            wt = self._safe_metric(m, "waiting_time_norm", default=-1.0)
            scores.append(
                wt if wt >= 0.0
                else self._safe_metric(m, "effective_queue_norm")
            )

        result = float(np.clip(np.mean(scores), 0.0, 1.0))
        logger.debug("[%s] waiting_penalty=%.4f", tls_id, result)
        return result

    # ------------------------------------------------------------------
    # Resolution helpers
    # ------------------------------------------------------------------

    def _resolve_lane_ids_by_tls(
        self,
        tls_ids: Sequence[str],
        action_info: ActionInfo,
    ) -> Dict[str, Tuple[str, ...]]:
        """Resolve controlled lane IDs for each TLS from action_info."""
        mapping: Any = (
            action_info.get("lane_ids_by_tls")
            or action_info.get("controlled_lanes_dict")
        )
        if isinstance(mapping, Mapping):
            result: Dict[str, Tuple[str, ...]] = {}
            for tls_id in tls_ids:
                raw = mapping.get(tls_id)
                if isinstance(raw, (list, tuple)):
                    result[tls_id] = tuple(str(x) for x in raw)
                elif raw is not None:
                    result[tls_id] = (str(raw),)
                else:
                    result[tls_id] = ()
            return result

        fallback = action_info.get("lane_ids")
        if isinstance(fallback, (list, tuple)):
            shared = tuple(str(x) for x in fallback)
            return {tls_id: shared for tls_id in tls_ids}

        return {tls_id: () for tls_id in tls_ids}

    def _resolve_green_lane_ids(
        self,
        tls_ids: Sequence[str],
        action_info: ActionInfo,
        lane_ids_by_tls: Dict[str, Tuple[str, ...]],
    ) -> Dict[str, Optional[Tuple[str, ...]]]:
        """Resolve green lanes and derive red lanes (complement) per TLS.

        Returns a dict that also stores red lanes under the key
        ``_red_{tls_id}`` so both sets are available to the caller without
        an extra pass.

        If ``green_lanes_by_tls`` is absent from action_info, all entries
        are ``None`` and pressure_penalty returns 0 for every agent.
        """
        result: Dict[str, Optional[Tuple[str, ...]]] = {}
        green_info = action_info.get("green_lanes_by_tls")

        if not isinstance(green_info, Mapping):
            for tls_id in tls_ids:
                result[tls_id] = None
                result[f"_red_{tls_id}"] = None
            return result

        for tls_id in tls_ids:
            raw = green_info.get(tls_id)
            if isinstance(raw, (list, tuple)):
                green = tuple(str(x) for x in raw)
                all_lanes = set(lane_ids_by_tls.get(tls_id, ()))
                red = tuple(all_lanes - set(green))
                result[tls_id]              = green
                result[f"_red_{tls_id}"]   = red
            else:
                result[tls_id]              = None
                result[f"_red_{tls_id}"]   = None

        return result

    def _resolve_throughput_delta(self, action_info: ActionInfo) -> float:
        """Compute per-step throughput increment.

        Precedence:
        1. ``delta_passed`` — pre-computed; used as-is.
        2. ``current_passed`` − ``prev_passed`` — caller-managed tracking.
        3. ``current_passed`` − ``_prev_passed_total`` — internal tracking.

        The internal ``_prev_passed_total`` is updated **only** when the caller
        does not supply ``prev_passed``.  This prevents state corruption when
        calls alternate between internal and external tracking.
        """
        if "delta_passed" in action_info:
            return max(self._safe_float(action_info["delta_passed"]), 0.0)

        current = action_info.get("current_passed") or action_info.get("passed")
        if current is None:
            return 0.0

        current_f          = max(self._safe_float(current), 0.0)
        caller_owns_prev   = "prev_passed" in action_info
        prev_f             = max(
            self._safe_float(
                action_info.get("prev_passed", self._prev_passed_total)
            ),
            0.0,
        )
        delta = max(current_f - prev_f, 0.0)

        if not caller_owns_prev:
            self._prev_passed_total = current_f

        return delta

    def _resolve_switch_counts(
        self,
        tls_ids: Sequence[str],
        action_info: ActionInfo,
    ) -> Dict[str, float]:
        """Resolve per-TLS phase-switch counts from action_info."""
        switches_by_tls = action_info.get("switches_by_tls")
        if isinstance(switches_by_tls, Mapping):
            return {
                tls_id: max(self._safe_float(switches_by_tls.get(tls_id, 0.0)), 0.0)
                for tls_id in tls_ids
            }

        num_switches = action_info.get("num_switches")
        if num_switches is not None and tls_ids:
            shared = max(self._safe_float(num_switches), 0.0) / float(len(tls_ids))
            return {tls_id: shared for tls_id in tls_ids}

        switched_tls = action_info.get("switched_tls")
        if isinstance(switched_tls, Mapping):
            return {
                tls_id: 1.0 if switched_tls.get(tls_id, False) else 0.0
                for tls_id in tls_ids
            }

        return {tls_id: 0.0 for tls_id in tls_ids}

    # ------------------------------------------------------------------
    # Micro-utilities
    # ------------------------------------------------------------------

    def _weight(self, weights: Mapping[str, float], key: str) -> float:
        try:
            return float(weights.get(key, 0.0))
        except Exception:
            return 0.0

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(default if value is None else value)
        except Exception:
            return float(default)

    def _safe_metric(
        self, metrics: LaneMetrics, key: str, default: float = 0.0
    ) -> float:
        try:
            return self._safe_float(metrics.get(key, default), default=default)
        except Exception:
            return float(default)


__all__ = ["RewardCalculator", "LaneCache", "LaneMetrics", "ActionInfo", "RewardDict"]
