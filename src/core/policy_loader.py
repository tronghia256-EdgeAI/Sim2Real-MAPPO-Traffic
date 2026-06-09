"""
policy_loader.py
================
Production-grade MAPPO policy loader.

Problems solved
---------------
1. State-dict key mismatch  — strips "net." / "module." / "actor." prefixes
                              automatically before loading.
2. Partial load fallback    — if strict=True fails, retries with strict=False
                              and logs every missing/unexpected key.
3. Architecture mismatch    — validates obs_dim and action_dim before inference.
4. Constant-action bug      — root cause: wrong obs_dim slice fed to a randomly-
                              initialised model.  This loader validates the
                              checkpoint's stored obs_dim if available.
5. Epsilon-greedy           — optional stochastic noise for production robustness.

Usage
-----
    loader = PolicyLoader(
        checkpoint_path="models/mappo/20260418_215140/best_model.pt",
        obs_dim=26,
        action_dim=2,
    )
    policy = loader.load()
    actions = policy.predict({"tls_0": obs_0, "tls_1": obs_1})
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class RunningMeanStd:
    """Welford online algorithm for mean and variance (matches train_ppo.py)."""

    def __init__(self, shape: tuple) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize: (x - mean) / (sqrt(var) + eps), clipped to [-10, 10]."""
        return np.clip(
            (x - self.mean.astype(np.float32)) / (np.sqrt(self.var).astype(np.float32) + 1e-8),
            -10.0, 10.0,
        ).astype(np.float32)

    def state_dict(self) -> Dict[str, Any]:
        return {"mean": self.mean.tolist(), "var": self.var.tolist(), "count": self.count}

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        self.mean = np.array(d["mean"], dtype=np.float64)
        self.var = np.array(d["var"], dtype=np.float64)
        self.count = float(d["count"])


# ─────────────────────────────────────────────────────────────────────────────
# Actor network  (matches train_ppo.py architecture exactly)
# ─────────────────────────────────────────────────────────────────────────────

class ActorNet:
    """Thin wrapper around nn.Sequential to avoid importing torch at module level."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        import torch
        import torch.nn as nn

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Attribute named "net" so checkpoint keys "net.0.*" etc. resolve directly
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def parameters(self):
        return self.net.parameters()

    def eval(self):
        self.net.eval()
        return self

    def state_dict(self):
        return {"net." + k: v for k, v in self.net.state_dict().items()}

    def load_state_dict(self, sd: Dict[str, Any], strict: bool = True):
        self.net.load_state_dict(sd, strict=strict)

    @property
    def _module(self):
        return self.net

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Key normalizer
# ─────────────────────────────────────────────────────────────────────────────

# Prefixes that various RL frameworks add to actor state dicts.
_STRIP_PREFIXES = (
    "net.",
    "actor.",
    "module.",
    "policy.",
    "actor_net.",
    "model.actor.",
    "model.net.",
    "_nn_module.net.",
    "_nn_module.",
)


def normalize_state_dict(raw_sd: Dict[str, Any]) -> Dict[str, Any]:
    """
    Strip common framework prefixes from checkpoint keys so they match the
    bare ``net.{layer}.*`` naming expected by ``ActorNet``.

    Example
    -------
    "actor.net.0.weight"  →  "net.0.weight"   (stripped "actor.")
    "module.net.0.weight" →  "net.0.weight"   (stripped "module.")
    "net.0.weight"        →  "net.0.weight"   (unchanged)
    """
    cleaned: Dict[str, Any] = {}
    for k, v in raw_sd.items():
        new_k = k
        for prefix in _STRIP_PREFIXES:
            if new_k.startswith(prefix):
                new_k = new_k[len(prefix):]
                break
        # Re-add "net." if the key is now bare (e.g. "0.weight" from SB3)
        if new_k and new_k[0].isdigit():
            new_k = "net." + new_k
        cleaned[new_k] = v
    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# Policy loader
# ─────────────────────────────────────────────────────────────────────────────

class PolicyLoader:
    """
    Loads a MAPPO actor from a PyTorch checkpoint with automatic key repair.

    Supported checkpoint formats
    ----------------------------
    1. ``{"actor_state_dict": {...}}``  — produced by this project's train_ppo.py
    2. ``{"state_dict": {...}}``        — generic PyTorch Lightning / SB3 style
    3. ``{...}``                        — raw state dict (keys at top level)

    The loader tries strict=True first.  On failure it strips prefixes and
    retries.  On second failure it falls back to strict=False with a warning.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        obs_dim: int,
        action_dim: int = 2,
        hidden_dim: int = 256,
        device: str = "cpu",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.device = device

    def load(self) -> "LoadedPolicy":
        """Load and return a ready-to-use policy wrapper."""
        import torch

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        ckpt = torch.load(str(self.checkpoint_path), map_location=self.device, weights_only=False)

        # ── 1. Extract the raw state dict ─────────────────────────────────────
        raw_sd = self._extract_state_dict(ckpt)

        # ── 2. Validate stored obs_dim if present ──────────────────────────────
        if isinstance(ckpt, dict) and "obs_dim" in ckpt:
            stored_obs_dim = int(ckpt["obs_dim"])
            if stored_obs_dim != self.obs_dim:
                raise ValueError(
                    f"obs_dim mismatch: checkpoint has {stored_obs_dim}, "
                    f"expected {self.obs_dim}.  Check state_config.json."
                )

        # ── 3. Build model ─────────────────────────────────────────────────────
        model = ActorNet(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
        )

        # ── 4. Try strict load → normalize keys → strict=False fallback ────────
        success = False
        for attempt, (sd, strict) in enumerate(
            [
                (raw_sd, True),                       # attempt 1: as-is, strict
                (normalize_state_dict(raw_sd), True), # attempt 2: normalized, strict
                (normalize_state_dict(raw_sd), False),# attempt 3: normalized, lenient
            ]
        ):
            try:
                model.load_state_dict(sd, strict=strict)
                logger.info(
                    "Policy loaded (attempt %d, strict=%s): %s",
                    attempt + 1, strict, self.checkpoint_path,
                )
                if not strict:
                    model_keys = set(model.state_dict().keys())
                    ckpt_keys  = set(sd.keys())
                    missing    = model_keys - ckpt_keys
                    unexpected = ckpt_keys - model_keys
                    if missing:
                        logger.warning("Missing keys in checkpoint: %s", missing)
                    if unexpected:
                        logger.warning("Unexpected keys in checkpoint: %s", unexpected)
                success = True
                break
            except RuntimeError as exc:
                logger.debug("Load attempt %d failed: %s", attempt + 1, exc)

        if not success:
            raise RuntimeError(
                f"Failed to load policy from {self.checkpoint_path} "
                "after 3 attempts.  Check architecture and obs_dim."
            )

        model.eval()
        
        obs_rms = None
        state_rms = None
        
        if isinstance(ckpt, dict):
            if "obs_rms" in ckpt:
                obs_rms = RunningMeanStd((self.obs_dim,))
                try:
                    obs_rms.load_state_dict(ckpt["obs_rms"])
                    logger.info("Loaded obs_rms from checkpoint")
                except Exception as e:
                    logger.warning("Failed to load obs_rms: %s", e)
                    obs_rms = None
            else:
                logger.warning("obs_rms not found in checkpoint. Observations will not be normalized.")
            
            if "state_rms" in ckpt:
                state_dim = ckpt.get("state_dim", self.obs_dim * 2)
                state_rms = RunningMeanStd((state_dim,))
                try:
                    state_rms.load_state_dict(ckpt["state_rms"])
                    logger.info("Loaded state_rms from checkpoint")
                except Exception as e:
                    logger.warning("Failed to load state_rms: %s", e)
                    state_rms = None
        
        return LoadedPolicy(
            model=model,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            checkpoint_path=str(self.checkpoint_path),
            obs_rms=obs_rms,
            state_rms=state_rms,
        )

    @staticmethod
    def _extract_state_dict(ckpt: Any) -> Dict[str, Any]:
        """Extract state dict from various checkpoint wrapper formats."""
        if isinstance(ckpt, dict):
            for key in ("actor_state_dict", "state_dict", "model_state_dict",
                        "actor", "policy_state_dict"):
                if key in ckpt and isinstance(ckpt[key], dict):
                    return ckpt[key]
            # Heuristic: if all values are tensors, it's a raw state dict
            import torch
            if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
                return ckpt
        raise ValueError(
            f"Cannot extract state dict from checkpoint type {type(ckpt)}. "
            "Expected dict with 'actor_state_dict' key."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Loaded policy (inference interface)
# ─────────────────────────────────────────────────────────────────────────────

class LoadedPolicy:
    """
    Thin wrapper around a loaded ActorNet.

    Provides:
    - Shape-validated predict()
    - Optional epsilon-greedy exploration
    - Observation logging for debugging
    """

    def __init__(
        self,
        model: ActorNet,
        obs_dim: int,
        action_dim: int,
        checkpoint_path: str = "",
        obs_rms: Optional[RunningMeanStd] = None,
        state_rms: Optional[RunningMeanStd] = None,
    ) -> None:
        self._model = model
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.checkpoint_path = checkpoint_path
        self._call_count = 0
        self.obs_rms = obs_rms
        self.state_rms = state_rms

    def predict(
        self,
        per_agent_obs: Dict[str, np.ndarray],
        deterministic: bool = True,
        epsilon: float = 0.0,
        log_obs: bool = False,
    ) -> Dict[str, int]:
        """
        Parameters
        ----------
        per_agent_obs : {tls_id: obs_array (obs_dim,)}
        deterministic : If True, argmax; otherwise sample from Categorical.
        epsilon       : Probability of random action (0 = greedy).
        log_obs       : If True, logs min/max/mean of each agent's obs (debug).

        Returns
        -------
        {tls_id: action_int}
        """
        import torch
        import random

        actions: Dict[str, int] = {}
        self._call_count += 1
        debug_every = 100

        for tls_id, obs_raw in per_agent_obs.items():
            obs = np.asarray(obs_raw, dtype=np.float32).reshape(-1)

            # ── Shape assertion ────────────────────────────────────────────────
            if obs.shape[0] != self.obs_dim:
                raise ValueError(
                    f"[{tls_id}] obs shape {obs.shape[0]} ≠ expected {self.obs_dim}. "
                    "Check per_agent_dim slicing in ControlThread."
                )

            # ── Debug logging ──────────────────────────────────────────────────
            if log_obs or (self._call_count % debug_every == 0):
                logger.debug(
                    "[%s] obs: min=%.4f max=%.4f mean=%.4f std=%.4f",
                    tls_id, obs.min(), obs.max(), obs.mean(), obs.std(),
                )

            # ── Epsilon-greedy ─────────────────────────────────────────────────
            if epsilon > 0.0 and random.random() < epsilon:
                action = random.randint(0, self.action_dim - 1)
                actions[tls_id] = action
                continue

            # ── Observation normalization ─────────────────────────────────────
            if self.obs_rms is not None:
                obs_normalized = self.obs_rms.normalize(obs)
            else:
                obs_normalized = obs

            # ── Policy inference ───────────────────────────────────────────────
            t = torch.as_tensor(obs_normalized, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = self._model.forward(t)
                if deterministic:
                    action_t = torch.argmax(logits, dim=-1)
                else:
                    from torch.distributions import Categorical
                    action_t = Categorical(logits=logits).sample()

            action = int(action_t.item())
            if not (0 <= action < self.action_dim):
                logger.error(
                    "Invalid action %d for %s; clamping to 0", action, tls_id
                )
                action = 0
            actions[tls_id] = action

        return actions

    def __repr__(self) -> str:
        return (
            f"LoadedPolicy(obs_dim={self.obs_dim}, action_dim={self.action_dim}, "
            f"path={self.checkpoint_path!r})"
        )
