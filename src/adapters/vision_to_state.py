from __future__ import annotations

"""
vision_to_state.py

Bridge module between the vision stack and the PPO policy.

Pipeline
--------
DetectorManager.get_latest_snapshot()
    -> VisionToState.build_packet(...)
        -> StateExtractor.build_state(...)
            -> PPO observation vector

Design goals
------------
1) Keep PPO input identical to the SUMO-trained observation layout.
2) Keep the adapter thin and deterministic.
3) Expose diagnostics and accident flags in metadata, not in the policy obs.
4) Leave a clean insertion point for the future vision_buffer stage.

This module intentionally does NOT invent new state semantics.
It only translates detector snapshots into the exact semantic bundle expected
by the current StateExtractor.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import time

import numpy as np

try:
    from src.vision.state_extractor import StateExtractor, LaneROI  
except Exception:  
    try:
        from src.vision.state_extractor import StateExtractor, LaneROI 
    except Exception as exc: 
        raise ImportError(
            "Unable to import StateExtractor/LaneROI. Ensure state_extractor.py is on PYTHONPATH."
        ) from exc


SnapshotType = Union[Dict[Union[int, str], Dict[str, Any]], List[Any], Any]


@dataclass
class VisionStatePacket:
    """Full output of the vision-to-state bridge."""
    obs: np.ndarray
    info: Dict[str, Any] = field(default_factory=dict)
    semantic_state: Dict[str, Any] = field(default_factory=dict)
    raw_snapshot: Any = None
    timestamp: float = 0.0


class VisionToState:
    """
    Convert detector snapshots into PPO-ready observations.

    The class is deliberately thin: it delegates lane aggregation and feature
    construction to StateExtractor, and only adds validation, camera ordering,
    and a stable high-level API.
    """

    def __init__(
        self,
        state_extractor: StateExtractor,
        camera_order: Optional[Sequence[Union[int, str]]] = None,
        strict_shape_check: bool = True,
        attach_timestamp: bool = True,
    ) -> None:
        self.state_extractor = state_extractor
        self.camera_order = list(camera_order) if camera_order is not None else None
        self.strict_shape_check = bool(strict_shape_check)
        self.attach_timestamp = bool(attach_timestamp)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self, cam_id: Optional[Union[int, str]] = None) -> None:
        """Reset internal state memory (currently delegated to StateExtractor)."""
        self.state_extractor.reset(cam_id=cam_id)

    def build_packet(
        self,
        detector_snapshot: SnapshotType,
        phase_map: Dict[str, int],
        green_timers: Dict[str, float],
        return_semantic: bool = True,
    ) -> VisionStatePacket:
        """
        Build a full packet from the detector snapshot.

        Parameters
        ----------
        detector_snapshot:
            Output of DetectorManager.get_latest_snapshot(), or any equivalent
            structure accepted by StateExtractor.
        phase_map:
            {tls_id: current phase index}
        green_timers:
            {tls_id: green time in seconds}
        return_semantic:
            If True, populate semantic_state for debugging / logging.
        """
        snapshot = self._ordered_snapshot(detector_snapshot)
        ts = time.time() if self.attach_timestamp else 0.0

        obs, info = self.state_extractor.build_state(
            snapshot,
            phase_map=phase_map,
            green_timers=green_timers,
            return_info=True,
        )

        if self.strict_shape_check:
            expected = self.state_extractor.get_obs_dim()
            if obs.shape != (expected,):
                raise RuntimeError(f"VisionToState produced obs shape {obs.shape}, expected {(expected,)}")

        semantic_state: Dict[str, Any] = {}
        if return_semantic:
            semantic_state = self.state_extractor.build_semantic_state(snapshot)
            semantic_state["source"] = "vision_to_state"
            semantic_state["timestamp"] = ts

        # Preserve top-level accident summary for orchestration / safety logic.
        semantic_state.setdefault("accident_found", bool(info.get("accident_found", False)))
        semantic_state.setdefault("accident_conf", float(info.get("accident_conf", 0.0)))

        return VisionStatePacket(
            obs=np.asarray(obs, dtype=np.float32),
            info=info,
            semantic_state=semantic_state,
            raw_snapshot=snapshot,
            timestamp=ts,
        )

    def build_obs(
        self,
        detector_snapshot: SnapshotType,
        phase_map: Dict[str, int],
        green_timers: Dict[str, float],
    ) -> np.ndarray:
        """Convenience method when only the PPO observation is needed."""
        packet = self.build_packet(detector_snapshot, phase_map, green_timers, return_semantic=False)
        return packet.obs

    def build_semantic_state(
        self,
        detector_snapshot: SnapshotType,
        phase_map: Dict[str, int],
        green_timers: Dict[str, float],
    ) -> Dict[str, Any]:
        """Convenience method for downstream debugging / dashboards / rules."""
        packet = self.build_packet(detector_snapshot, phase_map, green_timers, return_semantic=True)
        return packet.semantic_state

    def build_from_detector_manager(
        self,
        detector_manager: Any,
        phase_map: Dict[str, int],
        green_timers: Dict[str, float],
        return_semantic: bool = True,
    ) -> VisionStatePacket:
        """
        Pull the latest snapshot directly from DetectorManager.

        This keeps the PPO integration layer small:
            packet = bridge.build_from_detector_manager(det_mgr, phase_map, green_timers)
        """
        if not hasattr(detector_manager, "get_latest_snapshot"):
            raise TypeError("detector_manager must expose get_latest_snapshot().")

        snapshot = detector_manager.get_latest_snapshot()
        return self.build_packet(snapshot, phase_map, green_timers, return_semantic=return_semantic)

    # ------------------------------------------------------------------
    # Ordering / normalization
    # ------------------------------------------------------------------
    def _ordered_snapshot(self, detector_snapshot: SnapshotType) -> SnapshotType:
        """
        Ensure camera iteration order is stable.

        StateExtractor itself is lane-order driven, but keeping camera order
        deterministic helps debugging and future buffer fusion.
        """
        if self.camera_order is None:
            return detector_snapshot

        # Single frame-like dict should pass through untouched.
        if isinstance(detector_snapshot, dict):
            if "tracks" in detector_snapshot or "accident_found" in detector_snapshot:
                return detector_snapshot

            ordered: Dict[Union[int, str], Dict[str, Any]] = {}
            for cam_id in self.camera_order:
                if cam_id in detector_snapshot:
                    ordered[cam_id] = detector_snapshot[cam_id]
            for cam_id, item in detector_snapshot.items():
                if cam_id not in ordered:
                    ordered[cam_id] = item
            return ordered

        if isinstance(detector_snapshot, list):
            # Lists are kept as-is; if you need strict ordering, sort upstream.
            return detector_snapshot

        return detector_snapshot


class VisionToStateFactory:
    """
    Small helper for constructing the bridge from a config dictionary.

    This is useful when wiring the pipeline from JSON/YAML config files.
    """

    @staticmethod
    def create(
        tls_ids: Sequence[str],
        controlled_lanes_dict: Dict[str, List[str]],
        lane_rois: Dict[str, LaneROI],
        lane_groups: Optional[Dict[str, Tuple[List[str], List[str]]]] = None,
        max_lanes_per_tls: int = 4,
        queue_cap: float = 50.0,
        waiting_cap: float = 300.0,
        speed_cap: float = 15.0,
        vehicle_cap: float = 20.0,
        max_green_time: float = 90.0,
        stop_speed_m_s: float = 0.5,
        fps: float = 10.0,
        px_per_meter: float = 20.0,
        camera_order: Optional[Sequence[Union[int, str]]] = None,
        motorbike_class_ids: Optional[List[int]] = None,
        heavy_class_ids: Optional[List[int]] = None,
    ) -> VisionToState:
        extractor = StateExtractor(
            tls_ids=tls_ids,
            controlled_lanes_dict=controlled_lanes_dict,
            lane_rois=lane_rois,
            lane_groups=lane_groups,
            max_lanes_per_tls=max_lanes_per_tls,
            queue_cap=queue_cap,
            waiting_cap=waiting_cap,
            speed_cap=speed_cap,
            vehicle_cap=vehicle_cap,
            max_green_time=max_green_time,
            stop_speed_m_s=stop_speed_m_s,
            fps=fps,
            px_per_meter=px_per_meter,
            motorbike_class_ids=motorbike_class_ids,
            heavy_class_ids=heavy_class_ids,
        )
        return VisionToState(
            state_extractor=extractor,
            camera_order=camera_order,
            strict_shape_check=True,
            attach_timestamp=True,
        )


def build_vision_state_from_detector(
    bridge: VisionToState,
    detector_snapshot: SnapshotType,
    phase_map: Dict[str, int],
    green_timers: Dict[str, float],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Minimal convenience wrapper for PPO environments.
    Returns (obs, info) in the same spirit as gymnasium step/reset.
    """
    packet = bridge.build_packet(detector_snapshot, phase_map, green_timers, return_semantic=True)
    return packet.obs, packet.info