#!/usr/bin/env python3
"""
run_real_deployment.py
======================
End-to-end deployment monitoring harness for Traffic-Guard-AI MAPPO controller.

Generates synthetic observations and runs the MAPPO policy inference loop,
monitoring for latency spikes, NaN values, and action flickering.

Telemetry output (printed every --telemetry-every steps):
  FPS: 8.3 | Step: 042/500 | Mode: synthetic(uniform)
  Latency: infer=12ms  pp=3ms  total=15ms  avg=14ms [OK]
  Actions: tls_0=0(stable 18.2s)  tls_1=1(stable 7.1s) | Flicker: 0 | NaN: 0

Alerts (printed immediately):
  [SPIKE]    step=042  latency=134ms  (threshold=100ms)
  [CRITICAL] step=007  NaN in obs  tls_0 dims=[3,7]  tls_1 dims=[]
  [FLICKER]  step=055  tls_0  0->1  (phase held only 2.3s < min_green=5s)

Usage
-----
  python scripts/run_real_deployment.py --policy models/mappo/20260418_215140/best_model.pt
  python scripts/run_real_deployment.py --policy models/mappo/20260418_215140/best_model.pt --n-steps 500
  python scripts/run_real_deployment.py --policy models/mappo/20260418_215140/best_model.pt \\
      --obs-mode zeros --latency-threshold-ms 50
  python scripts/run_real_deployment.py --help
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Project root on sys.path
# ─────────────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# ANSI colour helpers (gracefully degrade when stdout is not a TTY)
# ─────────────────────────────────────────────────────────────────────────────
_USE_COLOR = sys.stdout.isatty()

RED    = "\033[91m" if _USE_COLOR else ""
GREEN  = "\033[92m" if _USE_COLOR else ""
YELLOW = "\033[93m" if _USE_COLOR else ""
CYAN   = "\033[96m" if _USE_COLOR else ""
BOLD   = "\033[1m"  if _USE_COLOR else ""
RESET  = "\033[0m"  if _USE_COLOR else ""

# Move cursor up N lines and clear them (for in-place telemetry refresh)
_UP_CLEAR = "\033[F\033[K" if _USE_COLOR else ""

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("deploy_monitor")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
OBS_DIM         = 26   # per-agent observation dimension
GLOBAL_OBS_DIM  = 52   # 2 agents × 26
TLS_IDS         = ["tls_0", "tls_1"]
TLS_OBS_SLICES  = {"tls_0": slice(0, 26), "tls_1": slice(26, 52)}

# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Deployment monitoring harness - MAPPO traffic signal control",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--policy",
        default=str(_ROOT / "models" / "mappo" / "20260418_215140" / "best_model.pt"),
        help="Path to MAPPO actor checkpoint (.pt)",
    )
    p.add_argument(
        "--config",
        default=str(_ROOT / "configs" / "state_config.json"),
        help="Path to state_config.json",
    )
    p.add_argument(
        "--n-steps",
        type=int,
        default=0,
        help="Number of inference steps to run (0 = run until Ctrl-C)",
    )
    p.add_argument(
        "--obs-mode",
        choices=["uniform", "zeros", "noise"],
        default="uniform",
        help=(
            "Synthetic observation generation mode: "
            "'uniform' = random [0,1], 'zeros' = all zeros, "
            "'noise' = zeros + small Gaussian noise"
        ),
    )
    p.add_argument(
        "--latency-threshold-ms",
        type=float,
        default=100.0,
        help="Latency spike threshold in milliseconds",
    )
    p.add_argument(
        "--min-green-time",
        type=float,
        default=5.0,
        help="Minimum green phase duration in seconds (flicker detection threshold)",
    )
    p.add_argument(
        "--telemetry-every",
        type=int,
        default=10,
        help="Print telemetry every N steps",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducible synthetic observations",
    )
    p.add_argument(
        "--ema-alpha",
        type=float,
        default=0.6,
        help="EMA smoothing alpha for VisionBuffer (matches production default)",
    )
    p.add_argument(
        "--target-fps",
        type=float,
        default=0.0,
        help="Target inference rate in steps/sec (0 = unlimited)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Show DEBUG logging from policy loader",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Telemetry printer
# ─────────────────────────────────────────────────────────────────────────────

class TelemetryPrinter:
    """Prints and refreshes 3-line in-place telemetry block."""

    _TELEMETRY_LINES = 3  # number of lines in the telemetry block

    def __init__(self, n_steps: int, obs_mode: str, latency_threshold_ms: float) -> None:
        self._n_steps = n_steps
        self._obs_mode = obs_mode
        self._threshold = latency_threshold_ms
        self._first_print = True

    def print_telemetry(
        self,
        step: int,
        fps: float,
        infer_ms: float,
        preproc_ms: float,
        total_ms: float,
        avg_ms: float,
        action_state: Dict[str, Dict],
        flicker_count: int,
        nan_total: int,
    ) -> None:
        step_label = f"{step:05d}" + (f"/{self._n_steps}" if self._n_steps > 0 else "")
        fps_str = f"{fps:.1f}" if fps > 0 else "n/a"

        # Line 1: FPS / step / mode
        line1 = (
            f"{BOLD}FPS:{RESET} {fps_str:>5}"
            f"  |  Step: {step_label}"
            f"  |  Mode: synthetic({self._obs_mode})"
        )

        # Line 2: latency breakdown
        latency_tag = (
            f"{GREEN}[OK]{RESET}"
            if total_ms <= self._threshold
            else f"{RED}[SPIKE]{RESET}"
        )
        line2 = (
            f"Latency:  infer={infer_ms:>6.1f}ms"
            f"  pp={preproc_ms:>5.1f}ms"
            f"  total={total_ms:>6.1f}ms"
            f"  avg={avg_ms:>6.1f}ms"
            f"  {latency_tag}"
        )

        # Line 3: actions + flicker + NaN
        action_parts = []
        now = time.perf_counter()
        for tls_id in TLS_IDS:
            st = action_state[tls_id]
            stable_s = now - st["phase_start"]
            action_parts.append(f"{tls_id}={st['action']}(stable {stable_s:.1f}s)")
        nan_color   = RED if nan_total > 0 else GREEN
        flick_color = RED if flicker_count > 0 else GREEN
        line3 = (
            "Actions:  "
            + "  ".join(action_parts)
            + f"  |  Flicker: {flick_color}{flicker_count}{RESET}"
            + f"  |  NaN: {nan_color}{nan_total}{RESET}"
        )

        # Erase previous telemetry block (except on first print)
        if not self._first_print and _USE_COLOR:
            sys.stdout.write(_UP_CLEAR * self._TELEMETRY_LINES)
        else:
            self._first_print = False

        print(line1)
        print(line2)
        print(line3)
        sys.stdout.flush()

    @staticmethod
    def alert_spike(step: int, total_ms: float, threshold: float) -> None:
        print(
            f"{RED}{BOLD}[SPIKE]{RESET}    "
            f"step={step:05d}  latency={total_ms:.1f}ms  "
            f"(threshold={threshold:.0f}ms)"
        )
        sys.stdout.flush()

    @staticmethod
    def alert_nan(step: int, nan_tls0: List[int], nan_tls1: List[int]) -> None:
        parts = []
        if nan_tls0:
            parts.append(f"tls_0 dims={nan_tls0}")
        if nan_tls1:
            parts.append(f"tls_1 dims={nan_tls1}")
        print(
            f"{RED}{BOLD}[CRITICAL]{RESET} "
            f"step={step:05d}  NaN in obs  "
            + "  ".join(parts)
        )
        sys.stdout.flush()

    @staticmethod
    def alert_flicker(
        step: int, tls_id: str, old_action: int, new_action: int, held_s: float, min_s: float
    ) -> None:
        print(
            f"{YELLOW}{BOLD}[FLICKER]{RESET}  "
            f"step={step:05d}  {tls_id}  {old_action}->{new_action}  "
            f"(phase held {held_s:.2f}s < min_green={min_s:.1f}s)"
        )
        sys.stdout.flush()


# ─────────────────────────────────────────────────────────────────────────────
# Observation generator
# ─────────────────────────────────────────────────────────────────────────────

def make_obs_generator(mode: str, seed: int):
    """Returns a callable () -> np.ndarray of shape (52,) in [0,1]."""
    rng = np.random.default_rng(seed)

    if mode == "zeros":
        def _gen() -> np.ndarray:
            return np.zeros(GLOBAL_OBS_DIM, dtype=np.float32)

    elif mode == "noise":
        def _gen() -> np.ndarray:
            return np.clip(
                rng.normal(0.0, 0.02, GLOBAL_OBS_DIM).astype(np.float32), 0.0, 1.0
            )

    else:  # uniform
        def _gen() -> np.ndarray:
            return rng.random(GLOBAL_OBS_DIM).astype(np.float32)

    return _gen


# ─────────────────────────────────────────────────────────────────────────────
# EMA smoother (inline, avoids VisionBuffer's VisionStatePacket dependency)
# ─────────────────────────────────────────────────────────────────────────────

class EMABuffer:
    """Lightweight EMA smoother matching VisionBuffer semantics."""

    def __init__(self, alpha: float = 0.6) -> None:
        self._alpha = float(alpha)
        self._prev: Optional[np.ndarray] = None

    def push(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        if self._prev is None:
            self._prev = obs.copy()
            return self._prev.copy()
        smoothed = self._alpha * obs + (1.0 - self._alpha) * self._prev
        self._prev = smoothed
        return np.clip(smoothed, 0.0, 1.0).astype(np.float32)

    def reset(self) -> None:
        self._prev = None


# ─────────────────────────────────────────────────────────────────────────────
# Action state tracker
# ─────────────────────────────────────────────────────────────────────────────

def _make_action_state() -> Dict[str, Dict]:
    now = time.perf_counter()
    return {
        tls: {"action": 0, "phase_start": now, "change_count": 0}
        for tls in TLS_IDS
    }


def _update_action_state(
    action_state: Dict[str, Dict],
    actions: Dict[str, int],
    step: int,
    min_green_time: float,
    flicker_count_ref: List[int],
    printer: TelemetryPrinter,
) -> None:
    now = time.perf_counter()
    for tls_id, new_action in actions.items():
        st = action_state[tls_id]
        if new_action != st["action"]:
            held_s = now - st["phase_start"]
            if held_s < min_green_time:
                flicker_count_ref[0] += 1
                printer.alert_flicker(
                    step, tls_id, st["action"], new_action, held_s, min_green_time
                )
            action_state[tls_id] = {
                "action": new_action,
                "phase_start": now,
                "change_count": st["change_count"] + 1,
            }


# ─────────────────────────────────────────────────────────────────────────────
# Summary printer
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(
    total_steps: int,
    elapsed_s: float,
    spike_count: int,
    flicker_count: int,
    nan_total: int,
    latencies: deque,
    action_state: Dict[str, Dict],
) -> None:
    print("\n" + "-" * 60)
    print(f"{BOLD}Deployment Monitor - Session Summary{RESET}")
    print("-" * 60)
    print(f"  Steps run      : {total_steps}")
    print(f"  Wall time      : {elapsed_s:.1f}s")
    avg_fps = total_steps / elapsed_s if elapsed_s > 0 else 0
    print(f"  Avg FPS        : {avg_fps:.2f}")
    if latencies:
        arr = np.array(list(latencies))
        print(
            f"  Latency (ms)   : "
            f"mean={arr.mean():.1f}  p50={np.percentile(arr,50):.1f}"
            f"  p95={np.percentile(arr,95):.1f}  max={arr.max():.1f}"
        )
    spike_color = RED if spike_count > 0 else GREEN
    flick_color = RED if flicker_count > 0 else GREEN
    nan_color   = RED if nan_total > 0 else GREEN
    print(f"  Latency spikes : {spike_color}{spike_count}{RESET}")
    print(f"  Flicker events : {flick_color}{flicker_count}{RESET}")
    print(f"  NaN detections : {nan_color}{nan_total}{RESET}")
    print("  Phase changes  :", {t: s["change_count"] for t, s in action_state.items()})
    verdict = (
        f"{GREEN}PASS - no anomalies detected{RESET}"
        if spike_count == 0 and flicker_count == 0 and nan_total == 0
        else f"{RED}REVIEW REQUIRED - see alerts above{RESET}"
    )
    print(f"  Verdict        : {verdict}")
    print("-" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    args = _parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Banner ────────────────────────────────────────────────────────────────
    print(f"\n{BOLD}Traffic-Guard-AI - Deployment Monitor{RESET}")
    print(f"  Policy   : {args.policy}")
    print(f"  Obs mode : synthetic({args.obs_mode})")
    print(f"  Steps    : {'unlimited' if args.n_steps == 0 else args.n_steps}")
    print(f"  Spike thr: {args.latency_threshold_ms:.0f}ms")
    print(f"  MinGreen : {args.min_green_time:.1f}s")
    print()

    # ── Load policy ───────────────────────────────────────────────────────────
    policy_path = Path(args.policy)
    if not policy_path.exists():
        print(f"{RED}[ERROR]{RESET} Policy checkpoint not found: {policy_path}")
        print("  Run training first, or specify --policy <path>")
        return 1

    print("Loading policy checkpoint...", end=" ", flush=True)
    try:
        from src.core.policy_loader import PolicyLoader
        policy = PolicyLoader(
            checkpoint_path=policy_path,
            obs_dim=OBS_DIM,
            action_dim=2,
        ).load()
        print(f"{GREEN}OK{RESET}")
        print(f"  {policy}")
    except Exception as exc:
        print(f"{RED}FAILED{RESET}")
        print(f"  {exc}")
        return 1

    # ── Optionally read EMA alpha from config ──────────────────────────────────
    ema_alpha = args.ema_alpha
    config_path = Path(args.config)
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                cfg = json.load(f)
            ema_alpha = float(
                cfg.get("system_params", {})
                   .get("temporal_smoothing", {})
                   .get("buffer_ema_alpha", ema_alpha)
            )
            print(f"  EMA alpha from config: {ema_alpha}")
        except Exception:
            pass

    # ── Init components ───────────────────────────────────────────────────────
    obs_gen     = make_obs_generator(args.obs_mode, args.seed)
    ema_buffer  = EMABuffer(alpha=ema_alpha)
    action_state = _make_action_state()
    printer     = TelemetryPrinter(args.n_steps, args.obs_mode, args.latency_threshold_ms)

    # Counters
    spike_count  = 0
    flicker_ref  = [0]  # mutable int in list for closure
    nan_total    = 0
    step         = 0
    latencies: deque = deque(maxlen=200)
    recent_latencies: deque = deque(maxlen=args.telemetry_every)

    step_delay = 1.0 / args.target_fps if args.target_fps > 0 else 0.0

    # FPS tracking
    fps_window: deque = deque(maxlen=50)
    last_fps_time = time.perf_counter()
    fps = 0.0

    print(f"\nStarting inference loop... {CYAN}(Ctrl-C to stop){RESET}\n")

    # Reserve 3 blank lines for the telemetry block
    for _ in range(TelemetryPrinter._TELEMETRY_LINES):
        print()

    # ── Graceful shutdown ─────────────────────────────────────────────────────
    _stop = [False]

    def _sigint(_sig, _frame):
        _stop[0] = True

    signal.signal(signal.SIGINT, _sigint)

    t_session_start = time.perf_counter()

    # ── Inference loop ────────────────────────────────────────────────────────
    try:
        while not _stop[0]:
            if args.n_steps > 0 and step >= args.n_steps:
                break

            t_step_start = time.perf_counter()

            # 1. Generate synthetic raw obs (52-dim)
            raw_obs = obs_gen()

            # 2. EMA smoothing (pre-inference post-processing)
            t0_pp = time.perf_counter()
            smoothed_obs = ema_buffer.push(raw_obs)
            t1_pp = time.perf_counter()
            preproc_ms = (t1_pp - t0_pp) * 1000.0

            # 3. NaN check on smoothed obs
            nan_mask = np.isnan(smoothed_obs)
            if nan_mask.any():
                nan_dims = np.where(nan_mask)[0].tolist()
                nan_tls0 = [d for d in nan_dims if d < 26]
                nan_tls1 = [d - 26 for d in nan_dims if d >= 26]
                nan_total += len(nan_dims)
                printer.alert_nan(step, nan_tls0, nan_tls1)
                # Replace NaN with zeros to allow inference to continue
                smoothed_obs = np.where(nan_mask, 0.0, smoothed_obs).astype(np.float32)

            # 4. Slice per-agent obs
            per_agent_obs = {
                tls: smoothed_obs[sl].copy()
                for tls, sl in TLS_OBS_SLICES.items()
            }

            # 5. Policy inference
            t0_infer = time.perf_counter()
            actions = policy.predict(per_agent_obs, deterministic=True)
            t1_infer = time.perf_counter()
            infer_ms = (t1_infer - t0_infer) * 1000.0

            # 6. Total latency
            total_ms = (t1_infer - t_step_start) * 1000.0
            latencies.append(total_ms)
            recent_latencies.append(total_ms)

            # 7. Latency spike alert
            if total_ms > args.latency_threshold_ms:
                spike_count += 1
                printer.alert_spike(step, total_ms, args.latency_threshold_ms)

            # 8. Action flicker check (skip step 0: initial action state is
            #    arbitrary; first inference result is always a "change")
            if step > 0:
                _update_action_state(
                    action_state, actions, step,
                    args.min_green_time, flicker_ref, printer
                )
            else:
                # Seed action state from first real inference output
                now = time.perf_counter()
                for tls_id, act in actions.items():
                    action_state[tls_id] = {"action": act, "phase_start": now, "change_count": 0}

            # 9. FPS calculation
            fps_window.append(t1_infer)
            if len(fps_window) >= 2:
                fps = (len(fps_window) - 1) / (fps_window[-1] - fps_window[0])

            # 10. Telemetry refresh
            if step % args.telemetry_every == 0:
                avg_ms = (
                    float(np.mean(list(recent_latencies)))
                    if recent_latencies else total_ms
                )
                printer.print_telemetry(
                    step=step,
                    fps=fps,
                    infer_ms=infer_ms,
                    preproc_ms=preproc_ms,
                    total_ms=total_ms,
                    avg_ms=avg_ms,
                    action_state=action_state,
                    flicker_count=flicker_ref[0],
                    nan_total=nan_total,
                )

            step += 1

            # 11. Rate limit
            if step_delay > 0:
                elapsed = time.perf_counter() - t_step_start
                sleep_s = step_delay - elapsed
                if sleep_s > 0:
                    time.sleep(sleep_s)

    except Exception as exc:
        print(f"\n{RED}{BOLD}[CRASH]{RESET} Unhandled exception at step {step}:")
        traceback.print_exc()
        elapsed_s = time.perf_counter() - t_session_start
        print_summary(step, elapsed_s, spike_count, flicker_ref[0], nan_total,
                      latencies, action_state)
        return 1

    # ── Session summary ───────────────────────────────────────────────────────
    elapsed_s = time.perf_counter() - t_session_start
    print()
    print_summary(step, elapsed_s, spike_count, flicker_ref[0], nan_total,
                  latencies, action_state)

    return 0 if (spike_count == 0 and flicker_ref[0] == 0 and nan_total == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
