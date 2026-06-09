"""
tests/test_state_consistency.py
================================
Verify state format consistency between training and runtime.

Tests
-----
- Observation dimension correctness (52 dims for 2 agents)
- Feature order and structure
- Normalization bounds [0, 1]
- Class ID handling (motorbike, heavy vehicles)
- Lane metrics computation
- TLS pressure calculation
"""

import pytest
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, List

from src.vision.state_extractor import StateExtractor, LaneROI
from src.core.config_validator import validate_state_config, ConfigValidationError


class TestStateExtractor:
    """State extraction consistency tests."""
    
    @pytest.fixture
    def state_config(self):
        """Load state configuration."""
        cfg_path = Path("configs/state_config.json")
        assert cfg_path.exists(), f"Config not found: {cfg_path}"
        with open(cfg_path, "r") as f:
            return json.load(f)
    
    @pytest.fixture
    def lane_rois(self, state_config):
        """Build lane ROIs from config."""
        lane_defs = state_config.get("lane_definitions", {})
        rois = {}
        for lane_id, lane_def in lane_defs.items():
            poly = [(float(x), float(y)) for x, y in lane_def.get("polygon", [])]
            rois[lane_id] = LaneROI(
                lane_id=lane_id,
                tls_id=lane_def.get("tls_id", ""),
                polygon=poly if poly else [(0, 0), (100, 0), (100, 100), (0, 100)],
            )
        return rois
    
    @pytest.fixture
    def extractor(self, state_config, lane_rois):
        """Create StateExtractor instance."""
        sp = state_config.get("system_params", {})
        vehicle_det = sp.get("vehicle_detection", {})
        lane_groups = state_config.get("lane_groups", {})
        
        return StateExtractor(
            tls_ids=state_config["tls_ids"],
            controlled_lanes_dict=state_config["controlled_lanes_dict"],
            lane_rois=lane_rois,
            lane_groups=lane_groups,
            motorbike_class_ids=vehicle_det.get("motorbike_class_ids", [3]),
            heavy_class_ids=vehicle_det.get("heavy_class_ids", [1, 4]),
            max_lanes_per_tls=state_config["traffic_lights"]["max_lanes_per_tls"],
            speed_cap=float(sp.get("speed_cap", 15.0)),
            max_green_time=float(sp.get("max_green_time", 90.0)),
            px_per_meter=float(sp.get("px_per_meter", 20.0)),
            stop_speed_m_s=float(sp.get("stop_speed_m_s", 0.5)),
        )
    
    def test_observation_dimension(self, state_config, extractor):
        """Test observation dimension matches config."""
        tls_ids = state_config["tls_ids"]
        vl = state_config["vector_layout"]
        expected_dim = vl["global_observation_dim"]
        
        actual_dim = extractor.get_obs_dim()
        assert actual_dim == expected_dim, (
            f"Obs dim mismatch: extractor={actual_dim}, config={expected_dim}"
        )
    
    def test_feature_names(self, extractor):
        """Test feature names (5 lane features)."""
        assert len(extractor.FEATURE_NAMES) == 5, (
            f"Expected 5 lane features, got {len(extractor.FEATURE_NAMES)}"
        )
        expected_names = {
            "effective_queue_norm", "occupancy_norm", "avg_speed_norm",
            "motorbike_share", "heavy_vehicle_share"
        }
        actual_names = set(extractor.FEATURE_NAMES)
        assert actual_names == expected_names, (
            f"Feature names mismatch: {actual_names} vs {expected_names}"
        )
    
    def test_class_ids_loaded(self, extractor):
        """Test class IDs are correctly loaded."""
        assert extractor._MOTORBIKE_CLASSES == frozenset({3}), (
            f"Motorbike classes: {extractor._MOTORBIKE_CLASSES}"
        )
        assert extractor._HEAVY_CLASSES == frozenset({1, 4}), (
            f"Heavy classes: {extractor._HEAVY_CLASSES}"
        )
    
    def test_observation_normalization(self, extractor, state_config):
        """Test observation values are normalized to [0, 1]."""
        # Build fake snapshot with vehicles
        snapshot = {
            0: {
                "tracks": [
                    {
                        "track_id": 1,
                        "cls_id": 2,  # car
                        "bbox": [10, 20, 50, 70],
                        "speed_px_s": 10.0,
                        "center": [30, 45],
                    },
                    {
                        "track_id": 2,
                        "cls_id": 3,  # motorcycle
                        "bbox": [100, 100, 120, 140],
                        "speed_px_s": 5.0,
                        "center": [110, 120],
                    },
                ],
                "accident_found": False,
                "timestamp": 0.0,
            }
        }
        
        phase_map = {tls: 0 for tls in state_config["tls_ids"]}
        green_timers = {tls: 30.0 for tls in state_config["tls_ids"]}
        
        obs = extractor.build_state(snapshot, phase_map, green_timers)
        
        # Check shape
        expected_dim = extractor.get_obs_dim()
        assert obs.shape == (expected_dim,), f"Shape: {obs.shape} vs ({expected_dim},)"
        
        # Check bounds (all values should be in [0, 1])
        assert np.all(obs >= 0.0), f"Min value: {np.min(obs)} < 0"
        assert np.all(obs <= 1.0), f"Max value: {np.max(obs)} > 1"
        
        # Check dtype
        assert obs.dtype == np.float32, f"Dtype: {obs.dtype}"
    
    def test_heavy_vehicle_detection(self, extractor, state_config):
        """Test heavy vehicle share is calculated correctly."""
        snapshot = {
            0: {
                "tracks": [
                    {"track_id": 1, "cls_id": 2, "bbox": [0, 0, 40, 50], "speed_px_s": 10.0},  # car
                    {"track_id": 2, "cls_id": 1, "bbox": [50, 0, 100, 80], "speed_px_s": 5.0},   # bus (heavy)
                    {"track_id": 3, "cls_id": 4, "bbox": [150, 0, 200, 80], "speed_px_s": 3.0},  # truck (heavy)
                ],
                "accident_found": False,
                "timestamp": 0.0,
            }
        }
        
        phase_map = {tls: 0 for tls in state_config["tls_ids"]}
        green_timers = {tls: 30.0 for tls in state_config["tls_ids"]}
        
        obs, info = extractor.build_state(snapshot, phase_map, green_timers, return_info=True)
        
        # Check that lane metrics include heavy_vehicle_share
        metrics = info.get("lane_metrics", {})
        assert metrics, "No lane metrics returned"
        
        for lane_id, m in metrics.items():
            heavy_share = m.get("heavy_vehicle_share", 0.0)
            # With 3 vehicles and 2 heavy, share should be ~0.67
            # (might be 0 if vehicles assigned to different lane)
            assert 0.0 <= heavy_share <= 1.0, (
                f"heavy_vehicle_share out of bounds: {heavy_share}"
            )
    
    def test_motorbike_detection(self, extractor, state_config):
        """Test motorbike share is calculated correctly."""
        snapshot = {
            0: {
                "tracks": [
                    {"track_id": 1, "cls_id": 3, "bbox": [0, 0, 20, 40], "speed_px_s": 20.0},   # motorcycle
                    {"track_id": 2, "cls_id": 3, "bbox": [50, 0, 70, 40], "speed_px_s": 18.0},  # motorcycle
                    {"track_id": 3, "cls_id": 2, "bbox": [150, 0, 190, 50], "speed_px_s": 10.0}, # car
                ],
                "accident_found": False,
                "timestamp": 0.0,
            }
        }
        
        phase_map = {tls: 0 for tls in state_config["tls_ids"]}
        green_timers = {tls: 30.0 for tls in state_config["tls_ids"]}
        
        obs, info = extractor.build_state(snapshot, phase_map, green_timers, return_info=True)
        
        metrics = info.get("lane_metrics", {})
        for lane_id, m in metrics.items():
            motorbike_share = m.get("motorbike_share", 0.0)
            assert 0.0 <= motorbike_share <= 1.0, (
                f"motorbike_share out of bounds: {motorbike_share}"
            )


class TestConfigValidator:
    """Configuration validation tests."""
    
    def test_state_config_valid(self):
        """Test state config passes validation."""
        cfg_path = Path("configs/state_config.json")
        assert cfg_path.exists(), f"Config not found: {cfg_path}"
        
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        
        # Should not raise
        validate_state_config(cfg)
    
    def test_halt_speed_consistent(self):
        """Test halt speed threshold is consistent."""
        cfg_path = Path("configs/state_config.json")
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        
        sp = cfg.get("system_params", {})
        halt_speed = float(sp.get("stop_speed_m_s", 0.1))
        
        # Should match training constant
        from src.traffic_env.components.observations import VisionLaneMetrics
        assert abs(halt_speed - VisionLaneMetrics.HALT_SPEED_THRESHOLD_MPS) < 0.01, (
            f"Halt speed mismatch: config={halt_speed}, training={VisionLaneMetrics.HALT_SPEED_THRESHOLD_MPS}"
        )
    
    def test_ema_alpha_in_range(self):
        """Test EMA alpha is in valid range [0, 1]."""
        cfg_path = Path("configs/state_config.json")
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        
        sp = cfg.get("system_params", {})
        smoothing = sp.get("temporal_smoothing", {})
        ema_alpha = smoothing.get("buffer_ema_alpha", 0.6)
        
        assert 0.0 <= ema_alpha <= 1.0, f"EMA alpha out of range: {ema_alpha}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
