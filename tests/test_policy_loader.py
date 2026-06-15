"""
tests/test_policy_loader.py
===========================
Regression guard for the PolicyLoader actor-loading bug (2026-06-16).

ActorNet.state_dict() prepends a "net." prefix, but load_state_dict() used to
forward keys to self.net (a Sequential expecting bare "0.weight") WITHOUT
stripping it. Strict load therefore failed and the strict=False fallback
silently left EVERY weight at random initialisation — so eval_compare's MAPPO
column and the live deployment policy ran an *untrained* network. The
missing/unexpected guard compared prefixed-vs-prefixed keys, hiding it.

These tests assert the loaded weights actually equal the checkpoint, and that
predictions match a direct reference forward pass.
"""
import torch
import torch.nn as nn
import numpy as np
import pytest

from src.core.policy_loader import PolicyLoader

OBS_DIM = 26
ACTION_DIM = 2


def _reference_actor(seed: int = 0) -> nn.Sequential:
    """A net architecturally identical to ActorNet, with non-default weights."""
    torch.manual_seed(seed)
    net = nn.Sequential(
        nn.Linear(OBS_DIM, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, ACTION_DIM),
    )
    for p in net.parameters():
        nn.init.normal_(p, mean=0.0, std=0.5)  # diverge from default init
    net.eval()
    return net


def _save_ckpt(path, actor_sd, obs_dim=OBS_DIM):
    torch.save(
        {
            "actor_state_dict": actor_sd,
            "obs_dim": obs_dim,
            # identity-ish normaliser so predict() argmax matches a raw forward
            "obs_rms": {"mean": [0.0] * obs_dim, "var": [1.0] * obs_dim, "count": 1000.0},
        },
        str(path),
    )


@pytest.mark.unit
def test_loaded_weights_match_checkpoint(tmp_path):
    """THE regression: every actor weight must equal what was saved."""
    ref = _reference_actor()
    # trainer checkpoint keys look like "net.0.weight"
    actor_sd = {f"net.{k}": v for k, v in ref.state_dict().items()}
    ckpt = tmp_path / "best_model.pt"
    _save_ckpt(ckpt, actor_sd)

    pol = PolicyLoader(checkpoint_path=ckpt, obs_dim=OBS_DIM, action_dim=ACTION_DIM).load()

    loaded = pol._model.net.state_dict()
    for k, v in ref.state_dict().items():
        assert torch.allclose(v, loaded[k]), f"weight {k} not loaded (random init?)"
    # obs_rms must be loaded too (else obs are un-normalised at inference)
    assert pol.obs_rms is not None


@pytest.mark.unit
def test_predictions_match_reference(tmp_path):
    """predict() argmax must equal a direct forward pass of the saved net."""
    ref = _reference_actor(seed=3)
    actor_sd = {f"net.{k}": v for k, v in ref.state_dict().items()}
    ckpt = tmp_path / "best_model.pt"
    _save_ckpt(ckpt, actor_sd)

    pol = PolicyLoader(checkpoint_path=ckpt, obs_dim=OBS_DIM, action_dim=ACTION_DIM).load()

    rng = np.random.default_rng(0)
    for _ in range(100):
        obs = rng.random(OBS_DIM).astype(np.float32)
        with torch.no_grad():
            expected = int(torch.argmax(ref(torch.as_tensor(obs).unsqueeze(0)), dim=-1).item())
        got = pol.predict({"tls_0": obs}, deterministic=True)["tls_0"]
        assert got == expected


@pytest.mark.unit
def test_prefixed_checkpoint_keys_still_load(tmp_path):
    """A wrapper prefix (e.g. DataParallel 'module.') must not break loading."""
    ref = _reference_actor(seed=7)
    actor_sd = {f"module.net.{k}": v for k, v in ref.state_dict().items()}
    ckpt = tmp_path / "best_model.pt"
    _save_ckpt(ckpt, actor_sd)

    pol = PolicyLoader(checkpoint_path=ckpt, obs_dim=OBS_DIM, action_dim=ACTION_DIM).load()

    loaded = pol._model.net.state_dict()
    for k, v in ref.state_dict().items():
        assert torch.allclose(v, loaded[k]), f"weight {k} not loaded from prefixed ckpt"
