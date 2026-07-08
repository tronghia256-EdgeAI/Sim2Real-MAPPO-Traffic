"""Record a demo GIF of a trained MAPPO policy driving a network in sumo-gui.

Drives MappoTrafficEnv with gui=True (traci). Frames are captured by
scheduling <snapshot> entries in a generated gui-settings file, which
sumo-gui writes on its own; the PNGs are then assembled into an animated GIF
with Pillow. No screen-recording software needed.

Examples (run from the project root):

  # whole-grid view, ~20 s GIF starting mid ramp-up
  python scripts/make_demo_gif.py \
    --checkpoint "models/paper1_mappo/ckpts_clean/old_main/<run>/best_model.pt"

  # close-up on one intersection (see motorcycle filtering)
  python scripts/make_demo_gif.py --checkpoint <ckpt> --track J5 --zoom 900

Notes
-----
* sumo-gui opens a real window while recording; do not minimize it
  (minimized windows may produce black frames on Windows).
* Do NOT set --delay 0: without a render delay, sumo-gui 1.24 (Windows)
  deadlocks after writing the first snapshot. The default 30 ms is safe and
  adds ~30 s of wall time per 1000 sim seconds.
* Times are sim seconds (this project's networks run at 1 s per SUMO step),
  so --capture-start/--every are effectively in SUMO steps as well.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from src.core.policy_loader import PolicyLoader  # noqa: E402
from src.traffic_env.config import (  # noqa: E402
    build_default_config,
    load_lane_groups_json,
)
from src.traffic_env.envs.multi_agent import MappoTrafficEnv  # noqa: E402

NETWORKS = {
    "n3_grid": "sumo_configs/networks/n3_grid",
    "n2_corridor": "sumo_configs/networks/n2_corridor",
}
VIEW = "View #0"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True, help="best_model.pt of a trained run")
    p.add_argument("--network", choices=sorted(NETWORKS), default="n3_grid")
    p.add_argument("--sumocfg", default="sumo_config.sumocfg",
                   help="which .sumocfg inside the network dir to load "
                        "(e.g. sumo_config_ood_high.sumocfg for a congested network)")
    p.add_argument("--seed", type=int, default=42, help="route seed")
    p.add_argument("--gif", default="figures/demo_mappo.gif", help="output GIF path")
    p.add_argument("--capture-start", type=int, default=600,
                   help="sim seconds to skip before recording (default lands in the demand ramp)")
    p.add_argument("--frames", type=int, default=240, help="number of frames to capture")
    p.add_argument("--every", type=int, default=2, help="capture every Nth sim second")
    p.add_argument("--fps", type=int, default=12, help="GIF playback frame rate")
    p.add_argument("--width", type=int, default=720, help="GIF width in px (height keeps aspect)")
    p.add_argument("--window-size", default="1280,720", help="sumo-gui window size WxH")
    p.add_argument("--track", default=None, metavar="TLS_ID",
                   help="center the view on this junction (default: whole network)")
    p.add_argument("--zoom", type=float, default=None,
                   help="view zoom in %% (e.g. 900 with --track; default: fit network)")
    p.add_argument("--delay", type=int, default=100,
                   help="sumo-gui render delay in ms (MUST be > 0: without it the GUI "
                        "deadlocks after the first snapshot on SUMO 1.24/Windows; "
                        "values below ~50 ms are also unreliable)")
    p.add_argument("--stall-timeout", type=int, default=60,
                   help="abort if no new frame appears for this many wall seconds "
                        "(the partial GIF is still assembled)")
    p.add_argument("--keep-frames", action="store_true", help="keep the raw PNG frames")
    args = p.parse_args()
    if args.delay <= 0:
        p.error("--delay must be > 0 (delay 0 deadlocks sumo-gui snapshots)")
    return args


def write_gui_settings(path: Path, frames_dir: Path, args: argparse.Namespace) -> int:
    """Write a gui-settings file that schedules one <snapshot> per frame.

    sumo-gui writes each file itself during the step AFTER its scheduled time,
    which avoids the traci.gui.screenshot() re-entry deadlock entirely.
    Returns the sim time (s) of the last scheduled snapshot.
    """
    lines = [
        "<viewsettings>",
        '    <scheme name="real world"/>',
        f'    <delay value="{args.delay}"/>',
    ]
    last_t = args.capture_start
    for i in range(args.frames):
        last_t = args.capture_start + i * args.every
        out = frames_dir / f"frame_{i:05d}.png"
        lines.append(f'    <snapshot file="{out}" time="{last_t}"/>')
    lines.append("</viewsettings>")
    path.write_text("\n".join(lines), encoding="utf-8")
    return last_t


def main() -> int:
    args = parse_args()

    net_dir = PROJECT_ROOT / NETWORKS[args.network]
    tls_ids, lane_groups = load_lane_groups_json(net_dir / "lane_groups.json")
    sumocfg_path = net_dir / args.sumocfg
    if not sumocfg_path.exists():
        raise SystemExit(f"[gif] no such sumocfg: {sumocfg_path}")
    cfg = build_default_config(
        sumo_cfg_path=str(sumocfg_path),
        gui=True,
        tls_ids=tls_ids,
        manual_lane_groups=lane_groups,
        max_steps=10_000,  # generous; the frame budget below is what stops the run
    )

    policy = PolicyLoader(
        checkpoint_path=args.checkpoint, obs_dim=cfg.local_obs_dim, action_dim=2,
    ).load()

    frames_dir = Path(tempfile.mkdtemp(prefix="demo_gif_frames_"))
    print(f"[gif] frames dir: {frames_dir}")

    # snapshots are scheduled declaratively in a gui-settings file: sumo-gui
    # writes each frame itself, so no traci.gui.screenshot() calls are needed
    # (repeated screenshot calls deadlock sumo-gui 1.24 on Windows).
    settings_path = frames_dir / "gui_settings.xml"
    last_snapshot_t = write_gui_settings(settings_path, frames_dir, args)

    env = MappoTrafficEnv(
        config=cfg,
        gui=True,
        use_libsumo=False,
        extra_sumo_args=[
            # NOTE: no "--start" here - BaseSumoEnv.add_default_flags already
            # passes it, and SUMO errors out on duplicate options.
            "--quit-on-end",
            "--window-size", args.window_size.replace("x", ","),
            "--gui-settings-file", str(settings_path),
        ],
    )
    # disable the subscription-based obs fast path: per-vehicle traci
    # subscriptions make sumo-gui 1.24/Windows stall after writing the first
    # scheduled snapshot (plain getters are stable; speed is irrelevant here).
    env.obs_builder._fast_metrics_enabled = False

    obs_dict, _ = env.reset(seed=args.seed)
    conn = env.sumo_conn

    # ---- view setup (viewport only; schema/delay come from the settings file)
    if args.track:
        x, y = conn.junction.getPosition(args.track)
        conn.gui.setOffset(VIEW, x, y)
        conn.gui.setZoom(VIEW, args.zoom if args.zoom else 900.0)
    elif args.zoom:
        conn.gui.setZoom(VIEW, args.zoom)

    # ---- stall watchdog: sumo-gui 1.24/Windows can deadlock while writing a
    # snapshot; if no new frame lands for --stall-timeout seconds, kill the
    # GUI so the blocked traci call raises and the partial GIF is assembled.
    import subprocess
    import threading
    import time as _time

    stop_watchdog = threading.Event()

    def watchdog():
        last_count, last_change = -1, _time.monotonic()
        while not stop_watchdog.wait(2.0):
            n = len(list(frames_dir.glob("frame_*.png")))
            now = _time.monotonic()
            if n != last_count:
                last_count, last_change = n, now
            elif n > 0 and now - last_change > args.stall_timeout:
                print(f"[gif] WATCHDOG: no new frame for {args.stall_timeout}s "
                      f"({n} captured) - killing sumo-gui", flush=True)
                subprocess.run(["taskkill", "/IM", "sumo-gui.exe", "/F"],
                               capture_output=True)
                return

    threading.Thread(target=watchdog, daemon=True).start()

    # ---- drive the policy until every snapshot has been written -----------
    print(f"[gif] running policy to t={last_snapshot_t + 2} s "
          f"({args.frames} frames from t={args.capture_start} s, every {args.every} s) ...")
    try:
        while conn.simulation.getTime() <= last_snapshot_t + 1:
            actions = policy.predict(
                {t: np.asarray(obs_dict[t], dtype=np.float32) for t in env.tls_ids},
                deterministic=True,
            )
            obs_dict, _, terminated, truncated, _ = env.step(actions)
            if any(terminated.values()) or any(truncated.values()):
                print("[gif] episode ended before the last snapshot time")
                break
    except Exception as e:
        print(f"[gif] run aborted early: {type(e).__name__}: {e}")
        print("[gif] assembling whatever frames were captured ...")
    finally:
        stop_watchdog.set()
        try:
            env.close()
        except Exception:
            pass

    # ---- assemble the GIF --------------------------------------------------
    from PIL import Image

    frame_paths = sorted(frames_dir.glob("frame_*.png"))
    if not frame_paths:
        print("[gif] ERROR: no frames were written - was the sumo-gui window visible?")
        return 1
    print(f"[gif] assembling {len(frame_paths)} frames ...")

    frames = []
    for fp in frame_paths:
        img = Image.open(fp)
        h = round(img.height * args.width / img.width)
        img = img.resize((args.width, h), Image.LANCZOS)
        frames.append(img.convert("P", palette=Image.ADAPTIVE, colors=128))

    gif_path = PROJECT_ROOT / args.gif
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=round(1000 / args.fps),
        loop=0,
        optimize=True,
    )
    print(f"[gif] wrote {gif_path} ({gif_path.stat().st_size / 1e6:.1f} MB)")

    if args.keep_frames:
        print(f"[gif] raw frames kept in {frames_dir}")
    else:
        shutil.rmtree(frames_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
