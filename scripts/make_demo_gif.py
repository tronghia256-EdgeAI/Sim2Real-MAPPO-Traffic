"""Record a demo GIF of a trained MAPPO policy driving a network in sumo-gui.

Drives MappoTrafficEnv with gui=True (traci), schedules a sumo-gui screenshot
after every Nth SUMO step via the TraCI gui domain, then assembles the frames
into an animated GIF with Pillow. No screen-recording software needed.

Examples (run from the project root):

  # whole-grid view, ~20 s GIF starting mid ramp-up
  python scripts/make_demo_gif.py \
    --checkpoint "models/paper1_mappo/ckpts_clean/old_main/<run>/best_model.pt"

  # close-up on one intersection (see motorcycle filtering)
  python scripts/make_demo_gif.py --checkpoint <ckpt> --track J5 --zoom 900

Notes
-----
* sumo-gui opens a real window while recording; do not minimize it
  (minimized windows may produce black screenshots on Windows).
* Screenshots are scheduled by traci.gui.screenshot() and written during the
  *next* simulationStep, so the last scheduled frame may be skipped.
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
    p.add_argument("--seed", type=int, default=42, help="route seed")
    p.add_argument("--gif", default="figures/demo_mappo.gif", help="output GIF path")
    p.add_argument("--capture-start", type=int, default=600,
                   help="SUMO steps to skip before recording (default lands in the demand ramp)")
    p.add_argument("--frames", type=int, default=240, help="number of frames to capture")
    p.add_argument("--every", type=int, default=2, help="capture every Nth SUMO step")
    p.add_argument("--fps", type=int, default=12, help="GIF playback frame rate")
    p.add_argument("--width", type=int, default=720, help="GIF width in px (height keeps aspect)")
    p.add_argument("--window-size", default="1280,720", help="sumo-gui window size WxH")
    p.add_argument("--track", default=None, metavar="TLS_ID",
                   help="center the view on this junction (default: whole network)")
    p.add_argument("--zoom", type=float, default=None,
                   help="view zoom in %% (e.g. 900 with --track; default: fit network)")
    p.add_argument("--keep-frames", action="store_true", help="keep the raw PNG frames")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    net_dir = PROJECT_ROOT / NETWORKS[args.network]
    tls_ids, lane_groups = load_lane_groups_json(net_dir / "lane_groups.json")
    cfg = build_default_config(
        sumo_cfg_path=str(net_dir / "sumo_config.sumocfg"),
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

    env = MappoTrafficEnv(
        config=cfg,
        gui=True,
        use_libsumo=False,
        extra_sumo_args=[
            "--start", "--quit-on-end",
            "--window-size", args.window_size.replace("x", ","),
        ],
    )
    obs_dict, _ = env.reset(seed=args.seed)
    conn = env.sumo_conn

    # ---- view setup -------------------------------------------------------
    try:
        conn.gui.setSchema(VIEW, "real world")
    except Exception as e:  # schema name varies across SUMO builds; not fatal
        print(f"[gif] setSchema skipped: {e}")
    if args.track:
        x, y = conn.junction.getPosition(args.track)
        conn.gui.setOffset(VIEW, x, y)
        conn.gui.setZoom(VIEW, args.zoom if args.zoom else 900.0)
    elif args.zoom:
        conn.gui.setZoom(VIEW, args.zoom)

    # ---- capture hook: schedule a screenshot every Nth SUMO step ----------
    counter = {"sumo_steps": 0, "scheduled": 0}
    orig_step = conn.simulationStep

    def stepped(*a, **kw):
        result = orig_step(*a, **kw)
        counter["sumo_steps"] += 1
        i = counter["sumo_steps"]
        if (i >= args.capture_start
                and counter["scheduled"] < args.frames
                and (i - args.capture_start) % args.every == 0):
            out = frames_dir / f"frame_{counter['scheduled']:05d}.png"
            conn.gui.screenshot(VIEW, str(out))
            counter["scheduled"] += 1
        return result

    conn.simulationStep = stepped

    # ---- drive the policy until the frame budget is spent -----------------
    print(f"[gif] running policy (skip {args.capture_start} SUMO steps, "
          f"capture {args.frames} frames, every {args.every} steps) ...")
    try:
        while counter["scheduled"] < args.frames:
            actions = policy.predict(
                {t: np.asarray(obs_dict[t], dtype=np.float32) for t in env.tls_ids},
                deterministic=True,
            )
            obs_dict, _, terminated, truncated, _ = env.step(actions)
            if any(terminated.values()) or any(truncated.values()):
                print("[gif] episode ended before the frame budget was spent")
                break
        env.sim_step(1)  # flush the last scheduled screenshot
    finally:
        conn.simulationStep = orig_step
        env.close()

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
