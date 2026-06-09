from __future__ import annotations

"""
PATCH NOTES (audit v2):
- FIX: info_dict now includes co2_mg_per_s, fuel_ml_per_s aggregated from lane_cache.
- FIX: info_dict now includes current_passed (throughput counter) so the training
  loop can track episode throughput.
- FIX: info_dict now includes avg_waiting_proxy (mean effective_queue_norm × step_length).
- FIX: green_timers are capped at max_green_time to prevent unbounded growth.
"""

from typing import Any, Dict, Optional, Tuple, List
import numpy as np

from src.traffic_env.envs.base_sumo import BaseSumoEnv
from src.traffic_env.config import TrafficEnvConfig
from src.traffic_env.components.observations import ObservationBuilder
from src.traffic_env.components.rewards import RewardCalculator

ObsDict = Dict[str, np.ndarray]
RewardDict = Dict[str, float]
DoneDict = Dict[str, bool]
InfoDict = Dict[str, Dict[str, Any]]


class MappoTrafficEnv(BaseSumoEnv):

    def __init__(self, config: TrafficEnvConfig, **sumo_kwargs: Any) -> None:
        super().__init__(sumo_cfg_path=config.sim.sumo_cfg_path, **sumo_kwargs)

        self.config = config
        self.tls_ids = list(config.tls_ids)

        self.obs_builder = ObservationBuilder(config=config, sumo_conn=None)
        self.reward_calculator = RewardCalculator(config=config)

        self.green_timers = {tls_id: 0.0 for tls_id in self.tls_ids}
        self.last_actions = {tls_id: 0 for tls_id in self.tls_ids}

        # dynamic phase maps (auto-detect on reset)
        self._green_phase_map: Dict[str, List[int]] = {}
        self._yellow_phase_map: Dict[str, Dict[int, int]] = {}

        # action_dim inferred after first reset; default 2 for binary phase control
        self.action_dim: int = 2

        self._step_count = 0

    # =========================================================
    # RESET
    # =========================================================
    def reset(self, seed: Optional[int] = None):
        sumo_conn = self.start_simulation(seed=seed)
        self.obs_builder.set_sumo_conn(sumo_conn)

        self.green_timers = {tls_id: 0.0 for tls_id in self.tls_ids}
        self.last_actions = {tls_id: 0 for tls_id in self.tls_ids}
        self.reward_calculator.reset()
        self._step_count = 0

        # build phase maps from SUMO program logic
        self._build_phase_maps()

        # update action_dim from detected green phases
        if self._green_phase_map:
            first_tls = self.tls_ids[0]
            self.action_dim = len(self._green_phase_map.get(first_tls, [0, 2]))

        self.sim_step(1)

        lane_cache = self.obs_builder.build_lane_cache(self.tls_ids)
        obs_dict, global_state = self.obs_builder.get_all_observations(
            lane_cache, green_timers=self.green_timers
        )

        info_dict = {
            tls_id: {"global_state": global_state} for tls_id in self.tls_ids
        }

        return obs_dict, info_dict

    # =========================================================
    # STEP
    # =========================================================
    def step(self, action_dict: Dict[str, int]):

        prev_passed = self._get_total_passed()

        self._step_count += 1
        is_done = self._step_count >= self.config.sim.max_steps

        switches = {}
        target_phases = {}

        # -----------------------------------------
        # 1. ACTION → GREEN PHASE
        # -----------------------------------------
        for tls_id, action in action_dict.items():
            current = self._get_phase(tls_id)

            action_idx = int(action)
            action_idx = max(0, min(action_idx, len(self._green_phase_map[tls_id]) - 1))
            target = self._green_phase_map[tls_id][action_idx]

            # enforce min green time
            if target != current and self.green_timers[tls_id] < self.config.sim.min_green_time:
                target = current
                switches[tls_id] = False
            else:
                switches[tls_id] = (target != current)

            target_phases[tls_id] = target

        # -----------------------------------------
        # 2. APPLY YELLOW PER TLS
        # -----------------------------------------
        yellow_duration = int(self.config.sim.yellow_time)

        for tls_id in self.tls_ids:
            if switches.get(tls_id, False):
                yellow = self._yellow_phase_map[tls_id].get(
                    self._get_phase(tls_id),
                    self._get_phase(tls_id)
                )
                self._set_phase(tls_id, yellow)

        if any(switches.values()):
            self.sim_step(yellow_duration)

        # -----------------------------------------
        # 3. APPLY GREEN
        # -----------------------------------------
        for tls_id in self.tls_ids:
            self._set_phase(tls_id, target_phases[tls_id])

        remaining = int(
            self.config.sim.step_length - (yellow_duration if any(switches.values()) else 0)
        )

        if remaining > 0:
            self.sim_step(remaining)

        # -----------------------------------------
        # 4. UPDATE TIMERS  (FIX: cap at max_green_time)
        # -----------------------------------------
        max_green = float(self.config.sim.max_green_time)
        for tls_id in self.tls_ids:
            if switches.get(tls_id, False):
                self.green_timers[tls_id] = float(max(remaining, 0))
            else:
                self.green_timers[tls_id] = min(
                    self.green_timers[tls_id] + float(self.config.sim.step_length),
                    max_green,
                )

            self.last_actions[tls_id] = int(action_dict.get(tls_id, 0))

        # -----------------------------------------
        # 5. OBS
        # -----------------------------------------
        lane_cache = self.obs_builder.build_lane_cache(self.tls_ids)
        obs_dict, global_state = self.obs_builder.get_all_observations(
            lane_cache,
            green_timers=self.green_timers
        )

        current_passed = self._get_total_passed()

        # -----------------------------------------
        # 6. REWARD
        # -----------------------------------------
        action_info = {
            "lane_ids_by_tls": {
                tls_id: self._get_lanes(tls_id) for tls_id in self.tls_ids
            },
            "current_passed": current_passed,
            "prev_passed": prev_passed,
            "switches_by_tls": switches,
        }

        reward_dict = self.reward_calculator.calculate_rewards(
            self.tls_ids, lane_cache, action_info
        )

        # -----------------------------------------
        # 7. DONE
        # -----------------------------------------
        terminated = {tls_id: False for tls_id in self.tls_ids}
        truncated = {tls_id: is_done for tls_id in self.tls_ids}

        # -----------------------------------------
        # 8. INFO  (FIX: add co2, fuel, throughput, waiting)
        # -----------------------------------------
        info_dict = {}
        for tls_id in self.tls_ids:
            lanes = self._get_lanes(tls_id)

            queue = 0.0
            co2 = 0.0
            fuel = 0.0
            avg_wait = 0.0
            if lanes:
                queue = sum(
                    float(self.sumo_conn.lane.getLastStepHaltingNumber(l))
                    for l in lanes
                )
                # aggregate emissions and waiting proxy from lane_cache
                for l in lanes:
                    m = lane_cache.get(l, {})
                    co2 += float(m.get("co2_mg_per_s", 0.0))
                    fuel += float(m.get("fuel_ml_per_s", 0.0))
                    avg_wait += float(m.get("effective_queue_norm", 0.0))
                avg_wait = (avg_wait / len(lanes)) * self.config.sim.step_length

            info_dict[tls_id] = {
                "global_state": global_state,
                "queue_total_proxy": queue,
                "current_passed": current_passed,          # FIX: throughput counter
                "co2_mg_per_s": co2,                       # FIX: per-tls emission
                "fuel_ml_per_s": fuel,                     # FIX: per-tls fuel
                "avg_waiting_proxy": avg_wait,             # FIX: waiting time proxy
                "reward_components": self.reward_calculator.last_reward_details.get(tls_id, {}),
            }

        return obs_dict, reward_dict, terminated, truncated, info_dict

    # =========================================================
    # PHASE MAP BUILDER (AUTO-DETECT)
    # =========================================================
    def _build_phase_maps(self) -> None:
        for tls_id in self.tls_ids:
            try:
                logic = self.sumo_conn.trafficlight.getAllProgramLogics(tls_id)[0]
                phases = logic.getPhases()

                green_phases = []
                yellow_map = {}

                for i, p in enumerate(phases):
                    state = p.state.lower()

                    if "g" in state:  # green phase
                        green_phases.append(i)

                    if "y" in state:
                        prev = (i - 1) % len(phases)
                        yellow_map[prev] = i

                if len(green_phases) >= 2:
                    self._green_phase_map[tls_id] = green_phases[:2]
                else:
                    self._green_phase_map[tls_id] = [0, 2]

                self._yellow_phase_map[tls_id] = yellow_map

            except Exception:
                # safe fallback for standard 4-phase SUMO signal
                self._green_phase_map[tls_id] = [0, 2]
                self._yellow_phase_map[tls_id] = {0: 1, 2: 3}

    # =========================================================
    # HELPERS
    # =========================================================
    def _get_phase(self, tls_id: str) -> int:
        try:
            return int(self.sumo_conn.trafficlight.getPhase(tls_id))
        except Exception:
            return 0

    def _set_phase(self, tls_id: str, phase: int) -> None:
        try:
            self.sumo_conn.trafficlight.setPhase(tls_id, int(phase))
        except Exception:
            pass

    def _get_lanes(self, tls_id: str) -> List[str]:
        try:
            return list(self.sumo_conn.trafficlight.getControlledLanes(tls_id))
        except Exception:
            return []

    def _get_total_passed(self) -> float:
        try:
            return float(self.sumo_conn.simulation.getArrivedNumber())
        except Exception:
            return 0.0


__all__ = ["MappoTrafficEnv"]