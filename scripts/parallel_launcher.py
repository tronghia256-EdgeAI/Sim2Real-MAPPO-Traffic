"""
parallel_launcher.py
====================
W5-9 main-campaign launcher: runs the full (network x seed x algo) training
matrix as throttled, staggered, isolated subprocesses.

Why subprocesses (not multiprocessing.Pool): each training run owns a SUMO
engine (libsumo in-process) and a torch session; full process isolation means
one crashed run can never poison the others, and logs/artifacts stay per-run.

Features
--------
- Worker-pool throttle      --max-workers (default 3: laptop-safe against RAM
                            pressure and thermal throttling; raise to ~10 on
                            a 16-core server — each run is ~1-1.5 cores).
- Staggered starts          --stagger seconds (default 45) between spawns so
                            concurrent SUMO inits don't spike CPU/IO together.
- Dynamic per-job args      unique --seed, per-job --ckpt-dir/--log-dir
                            (train_ppo.py creates a timestamped run dir inside;
                            those flags are this codebase's "run dir" control),
                            network-specific --sumo-cfg/--lane-groups, plus
                            arbitrary passthrough overrides after `--`.
- Isolation                 each job's stdout+stderr -> <campaign>/<job>.log.
- Resume                    jobs whose ckpt dir already holds a bench.json are
                            skipped, so a killed campaign restarts cleanly.
- Manifest                  <campaign>/campaign_manifest.json rewritten on
                            every state change (queued/running/done/failed,
                            pid, runtime, exit code) for external monitoring.
- Graceful Ctrl+C           terminates all children, marks them 'killed'.

Usage (from project root):
    # full default campaign: {n2_corridor, n3_grid} x 5 seeds, MAPPO, 2M steps
    python scripts/parallel_launcher.py

    # laptop smoke: tiny budget, 2 workers
    python scripts/parallel_launcher.py --total-timesteps 2048 --max-workers 2 \
        --seeds 42 123 --networks n2_corridor

    # server: 10 workers, include the IPPO baseline matrix
    python scripts/parallel_launcher.py --max-workers 10 --algos mappo ippo

    # passthrough overrides for train_ppo.py
    python scripts/parallel_launcher.py -- --rollout-horizon 256 --policy-lr 1e-4
"""
from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
NETWORKS_DIR = ROOT / "sumo_configs" / "networks"
DEFAULT_SEEDS = (42, 123, 456, 789, 1337)


# ---------------------------------------------------------------------------
# Job model
# ---------------------------------------------------------------------------

@dataclass
class Job:
    name: str                    # e.g. n3_grid_mappo_seed42
    network: str
    algo: str
    seed: int
    cmd: List[str]
    log_path: Path
    ckpt_dir: Path
    status: str = "queued"       # queued | running | done | failed | killed | skipped
    pid: Optional[int] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    runtime_s: Optional[float] = None
    exit_code: Optional[int] = None
    proc: Optional[subprocess.Popen] = field(default=None, repr=False, compare=False)
    log_handle: object = field(default=None, repr=False, compare=False)

    def manifest_row(self) -> Dict:
        # built explicitly — dataclasses.asdict() would deepcopy the live
        # Popen/file-handle fields and fail on their thread locks
        return {
            "name": self.name,
            "network": self.network,
            "algo": self.algo,
            "seed": self.seed,
            "status": self.status,
            "pid": self.pid,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "runtime_s": self.runtime_s,
            "exit_code": self.exit_code,
            "cmd": " ".join(self.cmd),
            "log_path": str(self.log_path),
            "ckpt_dir": str(self.ckpt_dir),
        }


def build_jobs(args: argparse.Namespace, campaign_dir: Path, passthrough: List[str]) -> List[Job]:
    jobs: List[Job] = []
    for network in args.networks:
        net_dir = NETWORKS_DIR / network
        sumo_cfg = net_dir / "sumo_config.sumocfg"
        lane_groups = net_dir / "lane_groups.json"
        for missing in (p for p in (sumo_cfg, lane_groups) if not p.exists()):
            raise SystemExit(f"missing {missing} — run generate_networks.py / generate_demand.py")

        for algo in args.algos:
            for seed in args.seeds:
                name = f"{network}_{algo}_seed{seed}"
                ckpt_dir = campaign_dir / "models" / name
                log_dir = campaign_dir / "tblogs" / name
                cmd = [
                    sys.executable, str(ROOT / "experiment" / "runners" / "train_ppo.py"),
                    "--mode", "train",
                    "--algo", algo,
                    "--sumo-cfg", str(sumo_cfg),
                    "--lane-groups", str(lane_groups),
                    "--seed", str(seed),
                    "--total-timesteps", str(args.total_timesteps),
                    "--rollout-horizon", str(args.rollout_horizon),
                    "--ckpt-dir", str(ckpt_dir),
                    "--log-dir", str(log_dir),
                ] + passthrough
                jobs.append(Job(
                    name=name, network=network, algo=algo, seed=seed, cmd=cmd,
                    log_path=campaign_dir / f"{name}.log", ckpt_dir=ckpt_dir,
                ))
    return jobs


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class CampaignScheduler:
    def __init__(self, jobs: List[Job], campaign_dir: Path,
                 max_workers: int, stagger_s: float, poll_s: float = 5.0) -> None:
        self.jobs = jobs
        self.campaign_dir = campaign_dir
        self.max_workers = max(1, int(max_workers))
        self.stagger_s = max(0.0, float(stagger_s))
        self.poll_s = poll_s
        self._last_spawn_t = 0.0
        self._interrupted = False

    # -- manifest ----------------------------------------------------------
    def write_manifest(self) -> None:
        manifest = {
            "campaign_dir": str(self.campaign_dir),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "max_workers": self.max_workers,
            "stagger_s": self.stagger_s,
            "counts": {
                status: sum(1 for j in self.jobs if j.status == status)
                for status in ("queued", "running", "done", "failed", "killed", "skipped")
            },
            "jobs": [j.manifest_row() for j in self.jobs],
        }
        path = self.campaign_dir / "campaign_manifest.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        tmp.replace(path)

    # -- lifecycle ---------------------------------------------------------
    def _already_complete(self, job: Job) -> bool:
        """A finished run leaves bench.json inside its timestamped run dir."""
        return any(job.ckpt_dir.glob("*/bench.json")) if job.ckpt_dir.exists() else False

    def _spawn(self, job: Job) -> None:
        job.log_handle = open(job.log_path, "a", encoding="utf-8", errors="replace")
        job.log_handle.write(
            f"\n===== launch {datetime.now().isoformat(timespec='seconds')} =====\n"
            f"$ {' '.join(job.cmd)}\n\n"
        )
        job.log_handle.flush()
        job.proc = subprocess.Popen(
            job.cmd, cwd=str(ROOT),
            stdout=job.log_handle, stderr=subprocess.STDOUT,
        )
        job.status = "running"
        job.pid = job.proc.pid
        job.started_at = datetime.now().isoformat(timespec="seconds")
        self._last_spawn_t = time.time()
        print(f"[launch] {job.name} (pid {job.pid}) -> {job.log_path.name}")

    def _reap(self, job: Job) -> None:
        assert job.proc is not None
        job.exit_code = job.proc.returncode
        job.finished_at = datetime.now().isoformat(timespec="seconds")
        if job.started_at:
            t0 = datetime.fromisoformat(job.started_at)
            job.runtime_s = round((datetime.fromisoformat(job.finished_at) - t0).total_seconds(), 1)
        job.status = "done" if job.exit_code == 0 else "failed"
        if job.log_handle:
            try:
                job.log_handle.close()
            except Exception:
                pass
        marker = "OK" if job.status == "done" else f"FAILED exit={job.exit_code}"
        print(f"[finish] {job.name}: {marker} ({job.runtime_s}s)")

    def _terminate_all(self) -> None:
        for job in self.jobs:
            if job.status == "running" and job.proc is not None:
                try:
                    job.proc.terminate()
                except Exception:
                    pass
        deadline = time.time() + 20
        for job in self.jobs:
            if job.status == "running" and job.proc is not None:
                try:
                    job.proc.wait(timeout=max(0.1, deadline - time.time()))
                except Exception:
                    try:
                        job.proc.kill()
                    except Exception:
                        pass
                job.status = "killed"
                job.finished_at = datetime.now().isoformat(timespec="seconds")
                if job.log_handle:
                    try:
                        job.log_handle.close()
                    except Exception:
                        pass

    # -- main loop ---------------------------------------------------------
    def run(self) -> int:
        def on_sigint(_sig, _frame):
            self._interrupted = True
            print("\n[ctrl-c] terminating running jobs ...")

        signal.signal(signal.SIGINT, on_sigint)

        for job in self.jobs:
            if self._already_complete(job):
                job.status = "skipped"
                print(f"[skip]   {job.name}: bench.json already present (resume)")
        self.write_manifest()

        while not self._interrupted:
            # reap finished
            changed = False
            for job in self.jobs:
                if job.status == "running" and job.proc is not None and job.proc.poll() is not None:
                    self._reap(job)
                    changed = True

            running = [j for j in self.jobs if j.status == "running"]
            queued = [j for j in self.jobs if j.status == "queued"]

            # spawn next (throttle + stagger)
            if queued and len(running) < self.max_workers:
                since_last = time.time() - self._last_spawn_t
                if since_last >= self.stagger_s or self._last_spawn_t == 0.0:
                    self._spawn(queued[0])
                    changed = True

            if changed:
                self.write_manifest()

            if not queued and not running:
                break
            time.sleep(self.poll_s)

        if self._interrupted:
            self._terminate_all()
        self.write_manifest()

        # summary
        print("\n" + "=" * 66)
        print("CAMPAIGN SUMMARY")
        print("=" * 66)
        print(f"  {'job':<28} {'status':<8} {'runtime':>9}  exit")
        for job in self.jobs:
            rt = f"{job.runtime_s:.0f}s" if job.runtime_s else "-"
            print(f"  {job.name:<28} {job.status:<8} {rt:>9}  {job.exit_code}")
        n_bad = sum(1 for j in self.jobs if j.status in ("failed", "killed"))
        print(f"\n  manifest: {self.campaign_dir / 'campaign_manifest.json'}")
        return 1 if n_bad else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    argv = sys.argv[1:]
    passthrough: List[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, passthrough = argv[:split], argv[split + 1:]

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--networks", nargs="+", default=["n2_corridor", "n3_grid"],
                        choices=["n1", "n2_corridor", "n3_grid"])
    parser.add_argument("--algos", nargs="+", default=["mappo"], choices=["mappo", "ippo"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--rollout-horizon", type=int, default=128)
    parser.add_argument("--max-workers", type=int, default=3,
                        help="concurrent training processes (3 = laptop-safe; "
                             "~10 on a 16-core server)")
    parser.add_argument("--stagger", type=float, default=45.0,
                        help="seconds between spawns (avoids simultaneous "
                             "SUMO-init CPU spikes)")
    parser.add_argument("--campaign-id", type=str, default=None,
                        help="reuse an existing campaign dir to resume it")
    args = parser.parse_args(argv)

    campaign_id = args.campaign_id or datetime.now().strftime("campaign_%Y%m%d_%H%M%S")
    campaign_dir = ROOT / "results" / "paper1_mappo" / campaign_id
    campaign_dir.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(args, campaign_dir, passthrough)
    total = len(jobs)
    print(f"campaign {campaign_id}: {total} jobs "
          f"({len(args.networks)} nets x {len(args.algos)} algos x {len(args.seeds)} seeds), "
          f"max_workers={args.max_workers}, stagger={args.stagger}s")
    print(f"artifacts -> {campaign_dir}")

    scheduler = CampaignScheduler(
        jobs, campaign_dir, max_workers=args.max_workers, stagger_s=args.stagger
    )
    return scheduler.run()


if __name__ == "__main__":
    sys.exit(main())
