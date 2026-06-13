"""
tests/test_obs_mode_and_noise.py
================================
W5-9 campaign infrastructure:
- privileged observation mode (paper VI-C upper-bound arm)
- eval-time sensing-noise model (paper VI-E robustness study)

Dependency-light: exercises config dims/guards and the noise model directly,
no SUMO engine required.
"""

import numpy as np
import pytest

from src.traffic_env.config import (
    DEFAULT_LANE_FEATURE_NAMES,
    PRIVILEGED_EXTRA_LANE_FEATURE_NAMES,
    ObservationConfig,
    build_default_config,
)
from src.traffic_env.components.obs_noise import NoiseConfig, SensingNoiseModel


# ---------------------------------------------------------------------------
# Privileged observation mode
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPrivilegedObsMode:
    def test_proxy_dims_unchanged(self):
        cfg = build_default_config(sumo_cfg_path="x.sumocfg", obs_mode="proxy")
        assert cfg.local_obs_dim == 26
        assert cfg.global_state_dim == 52
        assert cfg.observation.obs_mode == "proxy"

    def test_privileged_enlarges_obs(self):
        cfg = build_default_config(sumo_cfg_path="x.sumocfg", obs_mode="privileged")
        # 8 lane features x 4 approaches + 6 TLS dims
        assert cfg.local_obs_dim == 38
        assert cfg.global_state_dim == 76
        assert cfg.observation.obs_mode == "privileged"

    def test_privileged_appends_namespaced_features(self):
        cfg = build_default_config(sumo_cfg_path="x.sumocfg", obs_mode="privileged")
        names = cfg.observation.lane_feature_names
        # proxy features come first (slice order preserved for the 26-dim prefix)
        assert names[: len(DEFAULT_LANE_FEATURE_NAMES)] == DEFAULT_LANE_FEATURE_NAMES
        for extra in PRIVILEGED_EXTRA_LANE_FEATURE_NAMES:
            assert extra in names
            # namespaced so the RewardCalculator (reads "waiting_time_norm" etc.)
            # never consumes them: reward identical across arms, only obs differs
            assert extra.startswith("privileged_")

    def test_to_dict_records_obs_mode(self):
        cfg = build_default_config(sumo_cfg_path="x.sumocfg", obs_mode="privileged")
        assert cfg.to_dict()["observation"]["obs_mode"] == "privileged"

    def test_unknown_obs_mode_rejected(self):
        with pytest.raises(ValueError):
            build_default_config(sumo_cfg_path="x.sumocfg", obs_mode="cheating")

    def test_proxy_mode_rejects_privileged_feature_leak(self):
        leaked = ObservationConfig(
            lane_feature_names=DEFAULT_LANE_FEATURE_NAMES + PRIVILEGED_EXTRA_LANE_FEATURE_NAMES,
            obs_mode="proxy",
        )
        with pytest.raises(ValueError):
            leaked.validate()

    def test_privileged_mode_requires_features(self):
        missing = ObservationConfig(
            lane_feature_names=DEFAULT_LANE_FEATURE_NAMES,  # no privileged extras
            obs_mode="privileged",
        )
        with pytest.raises(ValueError):
            missing.validate()


# ---------------------------------------------------------------------------
# Sensing-noise model
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_obs():
    rng = np.random.default_rng(0)
    obs = rng.uniform(0.0, 1.0, 26).astype(np.float32)
    obs[20:24] = np.array([1, 0, 0, 0], dtype=np.float32)  # phase one-hot
    obs[24] = 0.5                                           # green timer
    return obs


@pytest.mark.unit
class TestSensingNoiseModel:
    def test_scale_zero_is_identity(self, sample_obs):
        m = SensingNoiseModel(NoiseConfig(scale=0.0), seed=7)
        assert np.allclose(m.apply(sample_obs), sample_obs)

    def test_determinism_same_seed(self, sample_obs):
        a = SensingNoiseModel(NoiseConfig(scale=1.0), seed=7).apply(sample_obs)
        b = SensingNoiseModel(NoiseConfig(scale=1.0), seed=7).apply(sample_obs)
        assert np.array_equal(a, b)

    def test_different_seed_differs(self, sample_obs):
        a = SensingNoiseModel(NoiseConfig(scale=1.0), seed=1).apply(sample_obs)
        b = SensingNoiseModel(NoiseConfig(scale=1.0), seed=2).apply(sample_obs)
        assert not np.array_equal(a, b)

    def test_output_bounds(self, sample_obs):
        out = SensingNoiseModel(NoiseConfig(scale=2.0), seed=3).apply(sample_obs)
        assert out.min() >= 0.0 and out.max() <= 1.0
        assert out.shape == (26,)

    def test_internal_dims_untouched(self, sample_obs):
        # phase one-hot (20..23) and green timer (24) are not camera-derived
        out = SensingNoiseModel(NoiseConfig(scale=2.0), seed=3).apply(sample_obs)
        assert np.array_equal(out[20:25], sample_obs[20:25])

    def test_larger_scale_more_deviation(self, sample_obs):
        base = sample_obs
        d1 = np.abs(SensingNoiseModel(NoiseConfig(scale=0.5), seed=5).apply(base) - base).sum()
        d2 = np.abs(SensingNoiseModel(NoiseConfig(scale=2.0), seed=5).apply(base) - base).sum()
        assert d2 > d1

    def test_camera_dropout_zeros_approach(self, sample_obs):
        m = SensingNoiseModel(
            NoiseConfig(scale=1.0, camera_dropout_approaches=(1,)), seed=7
        )
        out = m.apply(sample_obs)
        assert np.allclose(out[5:10], 0.0)   # approach 1 lane features
        assert not np.allclose(out[0:5], 0.0)  # approach 0 still present

    def test_one_step_delay_returns_previous(self, sample_obs):
        m = SensingNoiseModel(NoiseConfig(scale=1.0, delay_steps=1), seed=7)
        first = m.apply(sample_obs, "tls_0").copy()
        second = m.apply(sample_obs * 0.3, "tls_0")
        assert np.array_equal(second, first)

    def test_delay_buffers_per_agent(self, sample_obs):
        m = SensingNoiseModel(NoiseConfig(scale=1.0, delay_steps=1), seed=7)
        a0 = m.apply(sample_obs, "tls_0").copy()
        b0 = m.apply(sample_obs, "tls_1").copy()
        # second call per agent returns that agent's own stored frame
        a1 = m.apply(sample_obs * 0.1, "tls_0")
        b1 = m.apply(sample_obs * 0.1, "tls_1")
        assert np.array_equal(a1, a0)
        assert np.array_equal(b1, b0)

    def test_reset_clears_delay(self, sample_obs):
        m = SensingNoiseModel(NoiseConfig(scale=1.0, delay_steps=1), seed=7)
        first = m.apply(sample_obs, "tls_0").copy()
        m.reset()
        # after reset there is no stored previous -> returns current noisy frame
        out = m.apply(sample_obs * 0.3, "tls_0")
        assert not np.array_equal(out, first)

    def test_scaled_helper(self):
        c = NoiseConfig(scale=1.0).scaled(2.0)
        assert c.scale == 2.0
        assert c.queue_calib_sigma == 0.10  # other params preserved

    def test_is_calibrated_flag(self):
        # default placeholders -> not calibrated; results must not enter paper
        assert NoiseConfig().is_calibrated is False
        # once the placeholders are replaced with measured stats
        assert NoiseConfig(class_flip_rate=0.07, occupancy_bias_sigma=0.05).is_calibrated

    def test_invalid_delay_rejected(self):
        with pytest.raises(ValueError):
            SensingNoiseModel(NoiseConfig(delay_steps=2), seed=0)


# ---------------------------------------------------------------------------
# Privileged obs assembly (no SUMO — fake connection for lane topology)
# ---------------------------------------------------------------------------

class _FakeTL:
    def __init__(self, lanes):
        self._lanes = lanes

    def getControlledLanes(self, tls_id):
        return self._lanes


class _FakeLane:
    def getLength(self, lane_id):
        return 100.0


class _FakeConn:
    """Minimal connection exposing only what _approach_groups_for_tls needs."""
    def __init__(self, lanes):
        self.trafficlight = _FakeTL(lanes)
        self.lane = _FakeLane()


@pytest.mark.unit
class TestPrivilegedObsAssembly:
    def _builder(self):
        from src.traffic_env.components.observations import ObservationBuilder
        # two approaches, two lanes each: edges E0 and E1
        lanes = ["E0_0", "E0_1", "E1_0", "E1_1"]
        cfg = build_default_config(
            sumo_cfg_path="x.sumocfg", obs_mode="privileged",
            tls_ids=("J0",), max_lanes_per_tls=4,
            manual_lane_groups={"J0": (["E0_0", "E0_1"], ["E1_0", "E1_1"])},
        )
        return ObservationBuilder(config=cfg, sumo_conn=_FakeConn(lanes)), cfg

    def test_privileged_local_obs_is_38_dim(self):
        builder, cfg = self._builder()
        # lane_cache carrying both proxy and privileged keys per lane
        def lane(vals):
            return {
                "effective_queue_norm": vals[0], "occupancy_norm": vals[1],
                "avg_speed_norm": vals[2], "motorbike_share": vals[3],
                "heavy_vehicle_share": vals[4], "raw_vehicle_count": 2.0,
                "privileged_halt_count_norm": vals[5],
                "privileged_waiting_norm": vals[6],
                "privileged_vehicle_count_norm": vals[7],
            }
        cache = {
            "E0_0": lane([0.2, 0.3, 0.5, 0.7, 0.1, 0.4, 0.6, 0.8]),
            "E0_1": lane([0.4, 0.5, 0.5, 0.7, 0.1, 0.6, 0.8, 1.0]),
            "E1_0": lane([0.1, 0.1, 0.9, 0.2, 0.0, 0.2, 0.2, 0.3]),
            "E1_1": lane([0.1, 0.1, 0.9, 0.2, 0.0, 0.2, 0.2, 0.3]),
        }
        obs = builder.build_local_obs("J0", cache, green_timers={"J0": 0.0})
        assert obs.shape == (38,)

        # approach 0 (E0) privileged_halt_count_norm sits at offset 5 of block 0;
        # length-ratio feature -> unweighted mean of the two lanes (0.4, 0.6)
        assert obs[5] == pytest.approx(0.5, abs=1e-6)
        # approach 0 privileged_waiting_norm at offset 6: mean(0.6, 0.8)
        assert obs[6] == pytest.approx(0.7, abs=1e-6)

    def test_proxy_builder_emits_no_privileged_slots(self):
        from src.traffic_env.components.observations import ObservationBuilder
        lanes = ["E0_0", "E0_1", "E1_0", "E1_1"]
        cfg = build_default_config(
            sumo_cfg_path="x.sumocfg", obs_mode="proxy", tls_ids=("J0",),
            manual_lane_groups={"J0": (["E0_0", "E0_1"], ["E1_0", "E1_1"])},
        )
        builder = ObservationBuilder(config=cfg, sumo_conn=_FakeConn(lanes))
        assert builder._privileged is False
        cache = {lid: {"effective_queue_norm": 0.1, "raw_vehicle_count": 1.0}
                 for lid in lanes}
        obs = builder.build_local_obs("J0", cache, green_timers={"J0": 0.0})
        assert obs.shape == (26,)
