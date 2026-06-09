from __future__ import annotations

"""
core sumo wrapper for traffic rl environments.

this module contains only sumo process management and simulation stepping.
it does not contain any rl-specific logic such as rewards, observations,
actions, or agent bookkeeping.
"""

import os
import sys
import uuid
from typing import Any, List, Optional, Sequence


def _ensure_sumo_api(gui: bool) -> Any:
    """import a sumo python api with a safe fallback order."""
    if not gui:
        try:
            import libsumo as api  # type: ignore

            return api
        except Exception:
            pass

    try:
        import traci as api  # type: ignore

        return api
    except Exception:
        pass

    if "SUMO_HOME" not in os.environ:
        raise ImportError(
            "neither libsumo nor traci could be imported, and SUMO_HOME is not set"
        )

    tools_path = os.path.join(os.environ["SUMO_HOME"], "tools")
    if tools_path not in sys.path:
        sys.path.append(tools_path)

    try:
        import traci as api  # type: ignore

        return api
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "failed to import traci after adding SUMO_HOME/tools to sys.path"
        ) from exc


class BaseSumoEnv:
    """minimal sumo lifecycle wrapper for upper-layer rl environments."""

    def __init__(
        self,
        sumo_cfg_path: str,
        gui: bool = False,
        use_libsumo: bool = True,
        no_step_log: bool = True,
        waiting_time_memory: int = 1000,
        print_warnings: bool = False,
        add_default_flags: bool = True,
    ) -> None:
        self.sumo_cfg_path = str(sumo_cfg_path)
        self.gui = bool(gui)
        self.use_libsumo = bool(use_libsumo)
        self.no_step_log = bool(no_step_log)
        self.waiting_time_memory = int(waiting_time_memory)
        self.print_warnings = bool(print_warnings)
        self.add_default_flags = bool(add_default_flags)

        if not self.sumo_cfg_path:
            raise ValueError("sumo_cfg_path must not be empty")
        if self.waiting_time_memory < 0:
            raise ValueError("waiting_time_memory must be >= 0")

        self._api = _ensure_sumo_api(self.gui and not self.use_libsumo)
        self._is_libsumo = self._api.__name__.lower().endswith("libsumo")

        self.label = str(uuid.uuid4())
        self.sumo_conn: Any = None
        self.sim_started = False
        self.last_command: List[str] = []

    # ------------------------------------------------------------------
    # command builders
    # ------------------------------------------------------------------
    def _build_sumo_cmd(
        self,
        seed: Optional[int] = None,
        extra_args: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """build a sumo command line suitable for start or reload."""
        cmd: List[str] = ["sumo-gui" if self.gui else "sumo", "-c", self.sumo_cfg_path]

        if self.add_default_flags:
            cmd.append("--start")
            cmd.append("--time-to-teleport")
            cmd.append("160")

        if self.no_step_log:
            cmd.append("--no-step-log")

        if self.waiting_time_memory >= 0:
            cmd.extend(["--waiting-time-memory", str(self.waiting_time_memory)])

        if not self.print_warnings:
            cmd.append("--no-warnings")

        if seed is not None:
            cmd.extend(["--seed", str(int(seed))])

        if extra_args:
            cmd.extend([str(arg) for arg in extra_args])

        return cmd

    def _build_load_cmd(
        self,
        seed: Optional[int] = None,
        extra_args: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """build a reload command for an already started backend."""
        cmd: List[str] = ["-c", self.sumo_cfg_path]

        if self.add_default_flags:
            cmd.append("--start")
            cmd.append("--time-to-teleport")
            cmd.append("160")
            
        if self.no_step_log:
            cmd.append("--no-step-log")

        if self.waiting_time_memory >= 0:
            cmd.extend(["--waiting-time-memory", str(self.waiting_time_memory)])

        if not self.print_warnings:
            cmd.append("--no-warnings")

        if seed is not None:
            cmd.extend(["--seed", str(int(seed))])

        if extra_args:
            cmd.extend([str(arg) for arg in extra_args])

        return cmd

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def start_simulation(
        self,
        seed: Optional[int] = None,
        extra_args: Optional[Sequence[str]] = None,
    ) -> Any:
        """start or reload the sumo simulation."""
        cmd = self._build_sumo_cmd(seed=seed, extra_args=extra_args)
        self.last_command = list(cmd)

        try:
            if not self.sim_started:
                if self._is_libsumo:
                    self._api.start(cmd)
                    self.sumo_conn = self._api
                else:
                    self._api.start(cmd, label=self.label)
                    self.sumo_conn = self._api.getConnection(self.label)
                self.sim_started = True
                return self.sumo_conn

            load_cmd = self._build_load_cmd(seed=seed, extra_args=extra_args)
            self.last_command = list(load_cmd)
            if self._is_libsumo:
                self._api.load(load_cmd)
                self.sumo_conn = self._api
            else:
                if self.sumo_conn is None:
                    self._api.start(cmd, label=self.label)
                    self.sumo_conn = self._api.getConnection(self.label)
                else:
                    self.sumo_conn.load(load_cmd)
            self.sim_started = True
            return self.sumo_conn
        except Exception as exc:
            self.sim_started = False
            self.sumo_conn = None
            raise RuntimeError("failed to start or load sumo simulation") from exc

    def sim_step(self, steps: int = 1) -> int:
        """advance the simulation by a number of steps."""
        if steps < 0:
            raise ValueError("steps must be >= 0")
        if not self.sim_started or self.sumo_conn is None:
            raise RuntimeError("sumo simulation is not started")

        advanced = 0
        try:
            for _ in range(int(steps)):
                self.sumo_conn.simulationStep()
                advanced += 1
            return advanced
        except Exception as exc:
            raise RuntimeError("sumo simulation crashed during sim_step") from exc

    def close(self) -> None:
        """close the sumo connection safely and idempotently."""
        if not self.sim_started:
            return

        try:
            if self._is_libsumo:
                try:
                    self._api.close()
                except Exception:
                    pass
            elif self.sumo_conn is not None:
                try:
                    self.sumo_conn.close()
                except Exception:
                    pass
        finally:
            self.sim_started = False
            self.sumo_conn = None

    # ------------------------------------------------------------------
    # convenience helpers
    # ------------------------------------------------------------------
    @property
    def api(self) -> Any:
        """return the underlying sumo api module."""
        return self._api

    @property
    def is_libsumo(self) -> bool:
        """return whether the wrapper is using libsumo."""
        return self._is_libsumo

    def is_running(self) -> bool:
        """return true when the simulation connection is active."""
        return bool(self.sim_started and self.sumo_conn is not None)


__all__ = ["BaseSumoEnv"]