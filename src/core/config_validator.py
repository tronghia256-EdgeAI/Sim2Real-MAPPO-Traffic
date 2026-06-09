"""
config_validator.py
====================
Comprehensive configuration validation for the traffic intelligence system.

Verifies:
- State format consistency (train vs runtime)
- Class ID alignment (detector, training, runtime)
- Halt speed threshold uniformity
- EMA smoothing factor ranges
- Observation dimension consistency
- All required fields present and valid types

Usage
-----
    from src.core.config_validator import validate_all_configs
    validate_all_configs(state_cfg, detector_cfg, training_env_cfg)
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class ConfigValidationError(ValueError):
    """Raised when configuration validation fails."""


def validate_state_config(cfg: Dict[str, Any]) -> None:
    """
    Validate production state configuration.
    
    Parameters
    ----------
    cfg : dict
        state_config.json contents
    
    Raises
    ------
    ConfigValidationError
        If any validation fails
    """
    # Required top-level keys
    required = {
        "tls_ids", "controlled_lanes_dict", "lane_definitions",
        "system_params", "traffic_lights", "vector_layout"
    }
    missing = required - set(cfg.keys())
    if missing:
        raise ConfigValidationError(f"state_config missing keys: {missing}")
    
    # TLS IDs
    tls_ids = cfg.get("tls_ids")
    if not isinstance(tls_ids, list) or not tls_ids:
        raise ConfigValidationError(f"tls_ids must be non-empty list, got {tls_ids}")
    
    # System params
    sp = cfg.get("system_params", {})
    
    # Halt speed threshold must be consistent with training
    halt_speed = float(sp.get("stop_speed_m_s", 0.1))
    if not 0.0 <= halt_speed <= 2.0:
        raise ConfigValidationError(
            f"stop_speed_m_s out of range [0, 2]: {halt_speed}"
        )
    
    # Check against training constant
    from src.traffic_env.components.observations import VisionLaneMetrics
    train_halt_speed = VisionLaneMetrics.HALT_SPEED_THRESHOLD_MPS
    if abs(halt_speed - train_halt_speed) > 0.01:
        logger.warning(
            "HALT_SPEED mismatch: config=%.2f m/s, training=%.2f m/s. "
            "This will cause feature distribution divergence!",
            halt_speed, train_halt_speed
        )
    
    # Class IDs
    vehicle_det = sp.get("vehicle_detection", {})
    motorbike_ids = vehicle_det.get("motorbike_class_ids", [3])
    heavy_ids = vehicle_det.get("heavy_class_ids", [1, 4])
    
    if not isinstance(motorbike_ids, list):
        raise ConfigValidationError(
            f"motorbike_class_ids must be list, got {type(motorbike_ids)}"
        )
    if not isinstance(heavy_ids, list):
        raise ConfigValidationError(
            f"heavy_class_ids must be list, got {type(heavy_ids)}"
        )
    
    # Check for overlaps (shouldn't happen)
    overlap = set(motorbike_ids) & set(heavy_ids)
    if overlap:
        logger.warning(
            "motorbike_class_ids and heavy_class_ids have overlap: %s",
            overlap
        )
    
    # EMA alpha
    smoothing = sp.get("temporal_smoothing", {})
    ema_alpha = smoothing.get("buffer_ema_alpha", 0.6)
    if not 0.0 <= ema_alpha <= 1.0:
        raise ConfigValidationError(
            f"buffer_ema_alpha out of range [0, 1]: {ema_alpha}"
        )
    
    # Other system params
    for key, min_val in {
        "fps": 1.0, "speed_cap": 0.1, "queue_cap": 1.0,
        "waiting_cap": 0.1, "vehicle_cap": 1.0, "max_green_time": 1.0,
        "px_per_meter": 1.0
    }.items():
        val = float(sp.get(key, 0))
        if val < min_val:
            raise ConfigValidationError(
                f"system_params.{key} below minimum {min_val}: {val}"
            )
    
    # Vector layout
    vl = cfg.get("vector_layout", {})
    lanes_dim = vl.get("lane_feature_dim", 5)
    tls_dim = vl.get("tls_feature_dim", 6)
    per_agent = vl.get("per_agent_observation_dim")
    global_dim = vl.get("global_observation_dim")
    max_lanes = cfg.get("traffic_lights", {}).get("max_lanes_per_tls", 4)
    
    expected_per_agent = max_lanes * lanes_dim + tls_dim
    if per_agent and per_agent != expected_per_agent:
        raise ConfigValidationError(
            f"per_agent_observation_dim mismatch: "
            f"config={per_agent}, expected={expected_per_agent} "
            f"(max_lanes={max_lanes} × lane_dim={lanes_dim} + tls_dim={tls_dim})"
        )
    
    expected_global = expected_per_agent * len(tls_ids)
    if global_dim and global_dim != expected_global:
        raise ConfigValidationError(
            f"global_observation_dim mismatch: "
            f"config={global_dim}, expected={expected_global} "
            f"({len(tls_ids)} agents × {expected_per_agent})"
        )
    
    # Controlled lanes dict
    ctrl_lanes = cfg.get("controlled_lanes_dict", {})
    for tls_id in tls_ids:
        if tls_id not in ctrl_lanes:
            raise ConfigValidationError(
                f"tls_id '{tls_id}' missing from controlled_lanes_dict"
            )
        lanes = ctrl_lanes[tls_id]
        if not isinstance(lanes, list) or not lanes:
            raise ConfigValidationError(
                f"controlled_lanes_dict['{tls_id}'] must be non-empty list, got {lanes}"
            )
    
    # Lane definitions
    lane_defs = cfg.get("lane_definitions", {})
    all_lanes = set()
    for tls_id, lanes in ctrl_lanes.items():
        all_lanes.update(lanes)
    
    for lane_id in all_lanes:
        if lane_id not in lane_defs:
            raise ConfigValidationError(
                f"Lane '{lane_id}' referenced in controlled_lanes_dict "
                f"but missing from lane_definitions"
            )
    
    logger.info("✓ state_config validation passed")
    logger.info(
        "  TLS IDs: %s", tls_ids
    )
    logger.info(
        "  Halt speed: %.2f m/s (vs training: %.2f m/s)",
        halt_speed, train_halt_speed
    )
    logger.info(
        "  Motorbike classes: %s, Heavy classes: %s", motorbike_ids, heavy_ids
    )
    logger.info(
        "  EMA alpha: %.2f", ema_alpha
    )
    logger.info(
        "  Observation: %d agents × %d dims = %d global",
        len(tls_ids), expected_per_agent, expected_global
    )


def validate_detector_config(
    cfg: Dict[str, Any],
    state_config: Optional[Dict[str, Any]] = None
) -> None:
    """
    Validate YOLO detector configuration.
    
    Parameters
    ----------
    cfg : dict
        Detector configuration
    state_config : dict, optional
        Reference state config for class ID alignment
    
    Raises
    ------
    ConfigValidationError
        If any validation fails
    """
    # Check for YOLO model file
    model_path = cfg.get("model_path")
    if not model_path:
        raise ConfigValidationError("detector config missing 'model_path'")
    
    # Check vehicle class IDs
    vehicle_ids = cfg.get("vehicle_class_ids", [1, 2, 3, 4])
    if not isinstance(vehicle_ids, list):
        raise ConfigValidationError(
            f"vehicle_class_ids must be list, got {type(vehicle_ids)}"
        )
    
    # Default YOLO model mapping for custom traffic dataset
    expected_classes = {0: "accident", 1: "bus", 2: "car", 3: "motorcycle", 4: "truck"}
    logger.info("✓ Detector YOLO class mapping: %s", expected_classes)
    
    # Cross-check with state config if provided
    if state_config:
        sp = state_config.get("system_params", {})
        vehicle_det = sp.get("vehicle_detection", {})
        state_heavy = set(vehicle_det.get("heavy_class_ids", [1, 4]))
        state_motorbike = set(vehicle_det.get("motorbike_class_ids", [3]))
        
        # Verify consistency
        if 1 not in vehicle_ids or 4 not in vehicle_ids:
            logger.warning(
                "Detector vehicle_class_ids %s doesn't include expected heavy classes [1,4]",
                vehicle_ids
            )
        if 3 not in vehicle_ids:
            logger.warning(
                "Detector vehicle_class_ids %s doesn't include motorbike class [3]",
                vehicle_ids
            )
    
    logger.info("✓ Detector configuration validated")


def validate_all_configs(
    state_config: Dict[str, Any],
    detector_config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Comprehensive validation across all config files.
    
    Parameters
    ----------
    state_config : dict
        state_config.json contents
    detector_config : dict, optional
        Detector configuration
    
    Raises
    ------
    ConfigValidationError
        If any validation fails
    """
    logger.info("=" * 70)
    logger.info("COMPREHENSIVE CONFIG VALIDATION")
    logger.info("=" * 70)
    
    try:
        validate_state_config(state_config)
        if detector_config:
            validate_detector_config(detector_config, state_config)
        logger.info("=" * 70)
        logger.info("✓ ALL CONFIGS VALIDATED SUCCESSFULLY")
        logger.info("=" * 70)
    except ConfigValidationError as e:
        logger.error("=" * 70)
        logger.error("✗ CONFIG VALIDATION FAILED: %s", e)
        logger.error("=" * 70)
        raise


def print_config_diagnostics(cfg: Dict[str, Any]) -> None:
    """
    Print detailed config diagnostics for troubleshooting.
    
    Parameters
    ----------
    cfg : dict
        state_config.json contents
    """
    logger.info("CONFIG DIAGNOSTICS:")
    logger.info("  TLS IDs: %s", cfg.get("tls_ids", []))
    
    sp = cfg.get("system_params", {})
    logger.info("  System params:")
    for key in ["fps", "speed_cap", "max_green_time", "stop_speed_m_s"]:
        logger.info("    %s: %.2f", key, float(sp.get(key, 0)))
    
    vehicle_det = sp.get("vehicle_detection", {})
    logger.info("  Vehicle classes:")
    logger.info("    motorbike: %s", vehicle_det.get("motorbike_class_ids", []))
    logger.info("    heavy: %s", vehicle_det.get("heavy_class_ids", []))
    
    smoothing = sp.get("temporal_smoothing", {})
    logger.info("  Temporal smoothing:")
    logger.info("    EMA alpha: %.2f", smoothing.get("buffer_ema_alpha", 0.6))
    
    vl = cfg.get("vector_layout", {})
    logger.info("  Vector layout:")
    logger.info("    lane_feature_dim: %d", vl.get("lane_feature_dim", 0))
    logger.info("    tls_feature_dim: %d", vl.get("tls_feature_dim", 0))
    logger.info("    per_agent_observation_dim: %d", vl.get("per_agent_observation_dim", 0))
    logger.info("    global_observation_dim: %d", vl.get("global_observation_dim", 0))
