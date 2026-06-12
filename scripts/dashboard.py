"""
Traffic-Guard-AI — Streamlit Web Dashboard

Run:
    streamlit run scripts/dashboard.py

Pages
-----
  Camera Dashboard : 8 camera feeds (from recorded .mp4 or live RTSP) with
                     real-time YOLO bounding boxes, class counters, MAPPO-driven
                     3-colour TLS SVG with yellow-transition logic, and accident
                     alert banner.

  Live Simulation  : run MAPPO / Max Pressure / SOTL / Fixed-Time inside SUMO,
                     stream real-time queue, speed, waiting, throughput charts.

  Training Curves  : load logs/rl/<run>/*.csv, display reward / loss / entropy
                     learning curves with a smoothing slider.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st

try:
    import serial.tools.list_ports as _list_ports
    _HAS_SERIAL = True
except ImportError:
    _list_ports = None
    _HAS_SERIAL = False

# ── project root ──────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Traffic-Guard-AI",
    page_icon="traffic_light",
    layout="wide",
    initial_sidebar_state="expanded",
)


def theme_css() -> str:
    """
    Enterprise light theme CSS.
    .streamlit/config.toml forces base="light"; this layer adds card styling.
    Selectors match Streamlit's actual rendered DOM.
    """
    return """
<style>
/* ── metric cards ─────────────────────────────────────── */
[data-testid="metric-container"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 14px 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
}
[data-testid="stMetricLabel"] p {
    color: #64748b !important;
    font-size: 0.76rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="stMetricValue"] > div {
    color: #1e293b !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}

/* ── section headers & markdown text ─────────────────── */
.main .block-container h1,
.main .block-container h2,
.main .block-container h3,
.main .block-container p {
    color: #1e293b !important;
}

/* ── sidebar ──────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #f1f5f9 !important;
    border-right: 1px solid #e2e8f0;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: #1e293b !important;
}

/* ── accident banner ──────────────────────────────────── */
.accident-banner {
    background: linear-gradient(135deg, #dc2626, #b91c1c);
    color: #ffffff !important;
    font-size: 1.05rem;
    font-weight: 700;
    padding: 14px 24px;
    border-radius: 10px;
    text-align: center;
    border-left: 4px solid #fca5a5;
    animation: blink 1s step-start infinite;
}
@keyframes blink { 50% { opacity: 0.55; } }

/* ── camera-label caption ─────────────────────────────── */
.cam-label {
    font-size: 0.72rem;
    color: #64748b !important;
    text-align: center;
    margin-top: 3px;
}
</style>"""


st.markdown(theme_css(), unsafe_allow_html=True)

# ── session state defaults ─────────────────────────────────────────────────────
if "serial_bridge" not in st.session_state:
    st.session_state.serial_bridge = None

# ── constants ─────────────────────────────────────────────────────────────────
CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (0,   0,   255),   # accident  — red
    1: (0,   165, 255),   # bus       — orange
    2: (0,   255, 0),     # car       — green
    3: (255, 255, 0),     # motorcycle— cyan
    4: (255, 0,   255),   # truck     — magenta
}
CLASS_NAMES: Dict[int, str] = {
    0: "accident", 1: "bus", 2: "car", 3: "motorcycle", 4: "truck",
}
DIRECTION_ICON: Dict[str, str] = {
    "north": "^", "south": "v", "east": ">", "west": "<",
}

_DEFAULT_POLICY = str(_ROOT / "models" / "mappo" / "20260418_215140" / "best_model.pt")
_DEFAULT_STATE_CFG = str(_ROOT / "configs" / "state_config.json")

# MAPPO: action index -> SUMO phase index
_PHASE_ACTION_MAP = [0, 2]

# Arduino RYG constants (must match SerialBridge)
_S_RED    = 0
_S_YELLOW = 1
_S_GREEN  = 2

# Safe-mode thresholds
SAFE_MODE_THRESHOLD = 0.25   # fraction of stale cameras that triggers safe mode
SAFE_MODE_GREEN_SEC = 30.0   # seconds per phase in fixed-cycle fallback


def _list_com_ports() -> List[str]:
    """Return sorted list of available COM ports; fallback to common names if pyserial absent."""
    if _HAS_SERIAL and _list_ports is not None:
        ports = sorted(p.device for p in _list_ports.comports())
        return ports if ports else ["COM3", "COM4", "/dev/ttyUSB0"]
    return ["COM3", "COM4", "/dev/ttyUSB0", "/dev/ttyACM0"]


def _tls_states_to_serial(tls_state: Dict[str, Dict]) -> List[int]:
    """
    Convert current tls_state to the 8-value RYG list expected by SerialBridge.

    Layout: [tls_0_dir0, tls_0_dir1, tls_0_dir2, tls_0_dir3,
             tls_1_dir0, tls_1_dir1, tls_1_dir2, tls_1_dir3]

    Mapping:
      yellow transition  → all 4 directions YELLOW
      phase 0 (group A)  → dirs 0,1 = GREEN  |  dirs 2,3 = RED
      phase 1 (group B)  → dirs 0,1 = RED    |  dirs 2,3 = GREEN
    """
    result: List[int] = []
    for tid in ["tls_0", "tls_1"]:
        ts = tls_state[tid]
        if ts["yellow_start"] is not None:
            result.extend([_S_YELLOW, _S_YELLOW, _S_YELLOW, _S_YELLOW])
        elif ts["phase"] == 0:
            result.extend([_S_GREEN, _S_GREEN, _S_RED, _S_RED])
        else:
            result.extend([_S_RED, _S_RED, _S_GREEN, _S_GREEN])
    return result


# =============================================================================
# Cached resources
# =============================================================================

@st.cache_resource(show_spinner="Loading YOLO model (OpenVINO)...")
def _load_yolo(model_path: str):
    from src.vision.detector import _load_yolo as _ov_load
    return _ov_load(model_path, use_openvino=True)


@st.cache_resource(show_spinner="Loading MAPPO policy...")
def _load_mappo_policy(policy_path: str) -> Optional[Any]:
    try:
        from src.core.policy_loader import PolicyLoader
        return PolicyLoader(
            checkpoint_path=policy_path,
            obs_dim=26,
            action_dim=2,
        ).load()
    except Exception as exc:
        st.warning(f"MAPPO policy load failed: {exc}")
        return None


@st.cache_resource(show_spinner="Initializing StateExtractor...")
def _build_state_extractor(cfg_path: str) -> Optional[Any]:
    try:
        from src.vision.state_extractor import StateExtractor, LaneROI

        with open(cfg_path) as f:
            cfg = json.load(f)

        sp = cfg["system_params"]

        lane_rois: Dict[str, Any] = {}
        for lane_id, ld in cfg["lane_definitions"].items():
            polygon = [tuple(p) for p in ld["polygon"]]
            lane_rois[lane_id] = LaneROI(
                lane_id=lane_id,
                tls_id=ld["tls_id"],
                polygon=polygon,
                cam_id=ld["cam_id"],
                lane_length_px=float(ld["lane_length_px"]) if ld.get("lane_length_px") else None,
                lane_width_px=float(ld["lane_width_px"])   if ld.get("lane_width_px")  else None,
            )

        lane_groups = {
            tid: (list(grps[0]), list(grps[1]))
            for tid, grps in cfg["lane_groups"].items()
        }

        vd = sp["vehicle_detection"]
        return StateExtractor(
            tls_ids=cfg["tls_ids"],
            controlled_lanes_dict=cfg["controlled_lanes_dict"],
            lane_rois=lane_rois,
            lane_groups=lane_groups,
            motorbike_class_ids=vd["motorbike_class_ids"],
            heavy_class_ids=vd["heavy_class_ids"],
            px_per_meter=float(sp.get("px_per_meter", 20.0)),
            stop_speed_m_s=float(sp.get("stop_speed_m_s", 0.5)),
            speed_cap=float(sp.get("speed_cap", 15.0)),
            max_green_time=float(sp.get("max_green_time", 60.0)),
            fps=float(sp.get("fps", 10.0)),
        )
    except Exception as exc:
        st.warning(f"StateExtractor init failed: {exc}")
        return None


# =============================================================================
# Detection helpers
# =============================================================================

def _draw_boxes(
    frame: np.ndarray,
    boxes,
    conf_thresh: float,
    cam_counts: Dict[int, int],
) -> Tuple[np.ndarray, bool]:
    """Draw custom-coloured bounding boxes; return annotated frame + accident flag."""
    annotated = frame.copy()
    accident_detected = False

    for box in boxes:
        conf = float(box.conf[0].item())
        if conf < conf_thresh:
            continue
        cls_id = int(box.cls[0].item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        color = CLASS_COLORS.get(cls_id, (200, 200, 200))
        label = f"{CLASS_NAMES.get(cls_id, str(cls_id))} {conf:.2f}"

        # Font and line thickness scale with bbox height:
        #   h=30px → font≈0.25  |  h=56px → font≈0.45  |  h=80px → font≈0.64
        box_h = max(y2 - y1, 1)
        font_scale = float(np.clip(box_h * 0.008, 0.25, 0.65))
        box_thickness = 1 if box_h < 50 else 2
        txt_thickness = 1

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, box_thickness)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, txt_thickness)
        cv2.rectangle(annotated, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), txt_thickness)

        cam_counts[cls_id] = cam_counts.get(cls_id, 0) + 1
        if cls_id == 0:
            accident_detected = True

    return annotated, accident_detected


def _boxes_to_tracks(
    boxes,
    cam_idx: int,
    conf_thresh: float,
    timestamp: float,
    src_w: int = 320,
    src_h: int = 240,
    dst_w: int = 1280,
    dst_h: int = 720,
) -> List[Dict[str, Any]]:
    """Convert ultralytics Boxes to track dicts expected by StateExtractor.

    Scales bbox coordinates from the YOLO inference resolution (src) back to
    the original camera resolution (dst) so that ROI polygons in state_config.json
    (defined at dst resolution) correctly assign vehicles to lanes.
    """
    sx = dst_w / src_w
    sy = dst_h / src_h
    tracks: List[Dict[str, Any]] = []
    for i, box in enumerate(boxes):
        if float(box.conf[0].item()) < conf_thresh:
            continue
        cls_id = int(box.cls[0].item())
        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
        x1, x2 = x1 * sx, x2 * sx
        y1, y2 = y1 * sy, y2 * sy
        tracks.append({
            "track_id":   i,
            "cls_id":     cls_id,
            "bbox":       [x1, y1, x2, y2],
            "speed_px_s": 0.0,
            "center":     [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
            "cam_id":     cam_idx,
            "timestamp":  timestamp,
        })
    return tracks


# =============================================================================
# SVG rendering — 3-colour TLS intersections
# =============================================================================

def render_intersection_svg(
    tls_id: str,
    phase: int,
    is_yellow: bool = False,
) -> str:
    """
    Render one top-down intersection as SVG with full 3-colour TL housings.

    phase=0          -> N/S green,  E/W red
    phase=1          -> E/W green,  N/S red
    is_yellow=True   -> outgoing-green direction shows yellow (transition state)
    """
    W = RW = 220
    H      = 280
    ARM    = 74
    C0     = (W - ARM) // 2   # 73
    C1     = C0 + ARM          # 147
    MID    = W // 2            # 110

    # Per-direction state
    if not is_yellow:
        ns_state = "green" if phase == 0 else "red"
        ew_state = "red"   if phase == 0 else "green"
    else:
        ns_state = "yellow" if phase == 0 else "red"
        ew_state = "red"    if phase == 0 else "yellow"

    _hex   = {"green": "#16a34a", "yellow": "#f59e0b", "red": "#dc2626"}
    _label = {"green": "GO",      "yellow": "WAIT",    "red": "STOP"}
    NS     = _hex[ns_state]
    EW     = _hex[ew_state]
    ns_lbl = _label[ns_state]
    ew_lbl = _label[ew_state]
    fid    = tls_id.replace("_", "")

    # Pre-compute 3-colour circle attributes (avoids quote conflicts in f-string)
    R = "#ef4444"
    Y = "#f59e0b"
    G = "#16a34a"

    def _op(state: str, circle: str) -> str:
        return "1" if state == circle else "0.12"

    def _gf(state: str, circle: str) -> str:
        return f'filter="url(#glow{fid})"' if state == circle else ""

    # N/S housing attributes
    ns_r_op = _op(ns_state, "red");    ns_r_gf = _gf(ns_state, "red")
    ns_y_op = _op(ns_state, "yellow"); ns_y_gf = _gf(ns_state, "yellow")
    ns_g_op = _op(ns_state, "green");  ns_g_gf = _gf(ns_state, "green")
    # E/W housing attributes
    ew_r_op = _op(ew_state, "red");    ew_r_gf = _gf(ew_state, "red")
    ew_y_op = _op(ew_state, "yellow"); ew_y_gf = _gf(ew_state, "yellow")
    ew_g_op = _op(ew_state, "green");  ew_g_gf = _gf(ew_state, "green")

    # Computed pixel coordinates
    # North housing: vertical, above N arm, left side
    Nh_x = C0 + 4;   Nh_y = C0 - 37   # housing top-left (y: 36)
    Nr_cy = C0 - 30;  Ny_cy = C0 - 20;  Ng_cy = C0 - 10
    Ncx   = C0 + 11

    # South housing: vertical, below S arm, right side
    Sh_x = C1 - 18;  Sh_y = C1 + 3    # housing top-left (y: 150)
    Sr_cy = C1 + 9;  Sy_cy = C1 + 19;  Sg_cy = C1 + 29
    Scx   = C1 - 11

    # West housing: horizontal, left of W arm, top side
    Wh_x = C0 - 37;  Wh_y = C0 + 4   # housing top-left (x: 36)
    Wr_cx = C0 - 31;  Wy_cx = C0 - 21; Wg_cx = C0 - 11
    Wcy   = C0 + 11

    # East housing: horizontal, right of E arm, bottom side
    Eh_x = C1 + 3;   Eh_y = C1 - 18  # housing top-left (x: 150)
    Er_cx = C1 + 9;  Ey_cx = C1 + 19; Eg_cx = C1 + 29
    Ecy   = C1 - 11

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">
  <defs>
    <filter id="glow{fid}" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="2.5" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>

  <!-- background (grass / kerb) -->
  <rect width="{W}" height="{RW}" fill="#e5e7eb" rx="6"/>

  <!-- road arms -->
  <rect x="{C0}" y="0"    width="{ARM}"    height="{C0}"    fill="#2d3748"/>
  <rect x="{C0}" y="{C1}" width="{ARM}"    height="{RW-C1}" fill="#2d3748"/>
  <rect x="0"    y="{C0}" width="{C0}"     height="{ARM}"   fill="#2d3748"/>
  <rect x="{C1}" y="{C0}" width="{RW-C1}" height="{ARM}"   fill="#2d3748"/>
  <!-- intersection box -->
  <rect x="{C0}" y="{C0}" width="{ARM}" height="{ARM}" fill="#374151"/>
  <line x1="{C0}" y1="{C0}" x2="{C1}" y2="{C1}" stroke="#4b5563" stroke-width="1" opacity="0.5"/>
  <line x1="{C1}" y1="{C0}" x2="{C0}" y2="{C1}" stroke="#4b5563" stroke-width="1" opacity="0.5"/>

  <!-- lane-centre dashes -->
  <line x1="{MID}" y1="4"      x2="{MID}" y2="{C0-8}"   stroke="white" stroke-width="2" stroke-dasharray="6,5" opacity="0.35"/>
  <line x1="{MID}" y1="{C1+8}" x2="{MID}" y2="{RW-4}"   stroke="white" stroke-width="2" stroke-dasharray="6,5" opacity="0.35"/>
  <line x1="4"     y1="{MID}"  x2="{C0-8}" y2="{MID}"   stroke="white" stroke-width="2" stroke-dasharray="6,5" opacity="0.35"/>
  <line x1="{C1+8}" y1="{MID}" x2="{RW-4}" y2="{MID}"   stroke="white" stroke-width="2" stroke-dasharray="6,5" opacity="0.35"/>

  <!-- coloured stop lines (glow) -->
  <line x1="{C0+3}" y1="{C0-3}"  x2="{C1-3}" y2="{C0-3}"  stroke="{NS}" stroke-width="4" stroke-linecap="round" filter="url(#glow{fid})"/>
  <line x1="{C0+3}" y1="{C1+3}"  x2="{C1-3}" y2="{C1+3}"  stroke="{NS}" stroke-width="4" stroke-linecap="round" filter="url(#glow{fid})"/>
  <line x1="{C0-3}" y1="{C0+3}"  x2="{C0-3}" y2="{C1-3}"  stroke="{EW}" stroke-width="4" stroke-linecap="round" filter="url(#glow{fid})"/>
  <line x1="{C1+3}" y1="{C0+3}"  x2="{C1+3}" y2="{C1-3}"  stroke="{EW}" stroke-width="4" stroke-linecap="round" filter="url(#glow{fid})"/>

  <!-- North TL housing (vertical) -->
  <rect x="{C0+10}" y="{C0-42}" width="2" height="6" fill="#9ca3af"/>
  <rect x="{Nh_x}" y="{Nh_y}" width="14" height="34" rx="3" fill="#111827"/>
  <circle cx="{Ncx}" cy="{Nr_cy}" r="4" fill="{R}" opacity="{ns_r_op}" {ns_r_gf}/>
  <circle cx="{Ncx}" cy="{Ny_cy}" r="4" fill="{Y}" opacity="{ns_y_op}" {ns_y_gf}/>
  <circle cx="{Ncx}" cy="{Ng_cy}" r="4" fill="{G}" opacity="{ns_g_op}" {ns_g_gf}/>

  <!-- South TL housing (vertical) -->
  <rect x="{Sh_x}" y="{Sh_y}" width="14" height="34" rx="3" fill="#111827"/>
  <circle cx="{Scx}" cy="{Sr_cy}" r="4" fill="{R}" opacity="{ns_r_op}" {ns_r_gf}/>
  <circle cx="{Scx}" cy="{Sy_cy}" r="4" fill="{Y}" opacity="{ns_y_op}" {ns_y_gf}/>
  <circle cx="{Scx}" cy="{Sg_cy}" r="4" fill="{G}" opacity="{ns_g_op}" {ns_g_gf}/>

  <!-- West TL housing (horizontal) -->
  <rect x="{Wh_x}" y="{Wh_y}" width="34" height="14" rx="3" fill="#111827"/>
  <circle cx="{Wr_cx}" cy="{Wcy}" r="4" fill="{R}" opacity="{ew_r_op}" {ew_r_gf}/>
  <circle cx="{Wy_cx}" cy="{Wcy}" r="4" fill="{Y}" opacity="{ew_y_op}" {ew_y_gf}/>
  <circle cx="{Wg_cx}" cy="{Wcy}" r="4" fill="{G}" opacity="{ew_g_op}" {ew_g_gf}/>

  <!-- East TL housing (horizontal) -->
  <rect x="{Eh_x}" y="{Eh_y}" width="34" height="14" rx="3" fill="#111827"/>
  <circle cx="{Er_cx}" cy="{Ecy}" r="4" fill="{R}" opacity="{ew_r_op}" {ew_r_gf}/>
  <circle cx="{Ey_cx}" cy="{Ecy}" r="4" fill="{Y}" opacity="{ew_y_op}" {ew_y_gf}/>
  <circle cx="{Eg_cx}" cy="{Ecy}" r="4" fill="{G}" opacity="{ew_g_op}" {ew_g_gf}/>

  <!-- direction labels -->
  <text x="{MID}"   y="16"       text-anchor="middle" fill="{NS}" font-size="13" font-weight="800" font-family="system-ui,sans-serif">N</text>
  <text x="{MID}"   y="{RW-4}"   text-anchor="middle" fill="{NS}" font-size="13" font-weight="800" font-family="system-ui,sans-serif">S</text>
  <text x="{RW-5}"  y="{MID+5}"  text-anchor="end"    fill="{EW}" font-size="13" font-weight="800" font-family="system-ui,sans-serif">E</text>
  <text x="5"       y="{MID+5}"  text-anchor="start"  fill="{EW}" font-size="13" font-weight="800" font-family="system-ui,sans-serif">W</text>

  <!-- label panel -->
  <rect x="0" y="{RW}" width="{W}" height="{H-RW}" fill="#ffffff"/>
  <rect x="0" y="{RW}" width="{W}" height="2" fill="#e2e8f0"/>
  <text x="{W//2}" y="{RW+20}" text-anchor="middle" fill="#1e293b" font-size="15" font-weight="700" font-family="system-ui,sans-serif">{tls_id}</text>
  <rect x="{W//2-36}" y="{RW+26}" width="72" height="1" fill="#e2e8f0"/>
  <text x="{W//2-10}" y="{RW+42}" text-anchor="end"   fill="#6b7280" font-size="11" font-family="system-ui,sans-serif">N/S</text>
  <text x="{W//2-4}"  y="{RW+42}" text-anchor="start" fill="{NS}"    font-size="11" font-weight="700" font-family="system-ui,sans-serif">{ns_lbl}</text>
  <text x="{W//2-10}" y="{RW+55}" text-anchor="end"   fill="#6b7280" font-size="11" font-family="system-ui,sans-serif">E/W</text>
  <text x="{W//2-4}"  y="{RW+55}" text-anchor="start" fill="{EW}"    font-size="11" font-weight="700" font-family="system-ui,sans-serif">{ew_lbl}</text>
</svg>"""


def render_twin_intersection_html(
    tls_states: Dict[str, Dict[str, Any]],
) -> str:
    """
    Return inline HTML for st.markdown(unsafe_allow_html=True).
    tls_states: {tls_id: {"phase": int, "yellow": bool}}
    """
    svgs = "".join(
        render_intersection_svg(
            tid,
            tls_states[tid].get("phase", 0),
            tls_states[tid].get("yellow", False),
        )
        for tid in sorted(tls_states.keys())
    )
    return (
        '<div style="'
        "display:flex;gap:56px;justify-content:center;align-items:flex-start;"
        "padding:16px 24px 8px;background:#ffffff;"
        "border:1px solid #e2e8f0;border-radius:12px;"
        'box-shadow:0 2px 8px rgba(0,0,0,0.06);">'
        + svgs
        + "</div>"
    )


# =============================================================================
# Per-direction traffic panel
# =============================================================================

def _render_intersection_panel(
    int_counts: Dict[str, Dict[str, Dict[int, int]]],
    tls_states: Dict[str, Dict[str, Any]],
) -> str:
    """
    Two side-by-side intersection cards (tls_0 and tls_1), each showing
    per-direction vehicle counts and type breakdown.

    int_counts : {tls_id: {direction: {cls_id: count}}}
    tls_states : current TLS display state {tls_id: {"phase": int, "yellow": bool}}
    """
    _type_bg  = {1: "#f97316", 2: "#22c55e", 3: "#a855f7", 4: "#6366f1"}
    _type_lbl = {1: "Bus", 2: "Car", 3: "Moto", 4: "Truck"}
    _dir_label = {"north": "N", "south": "S", "east": "E", "west": "W", "unknown": "?"}
    _tls_title = {"tls_0": "Intersection 1 (tls_0)", "tls_1": "Intersection 2 (tls_1)"}

    def _sig_color(tls_id: str) -> str:
        ts = tls_states.get(tls_id, {})
        if ts.get("yellow", False):
            return "#f59e0b"
        return "#16a34a" if ts.get("phase", 0) == 0 else "#dc2626"

    outer_cards = []
    for tls_id in ["tls_0", "tls_1"]:
        dir_data  = int_counts.get(tls_id, {})
        sig_color = _sig_color(tls_id)
        ts        = tls_states.get(tls_id, {})
        phase_lbl = "Phase A" if ts.get("phase", 0) == 0 else "Phase B"
        sig_dot   = f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{sig_color};margin-right:5px;"></span>'
        total_all = sum(c for dc in dir_data.values() for c in dc.values())

        dir_cards = ""
        for direction in sorted(dir_data.keys()):
            counts = dir_data[direction]
            total  = sum(counts.values())
            label  = _dir_label.get(direction, direction[:1].upper())
            tags   = ""
            for cls_id in [2, 3, 1, 4]:
                cnt = counts.get(cls_id, 0)
                bg  = _type_bg[cls_id] if cnt > 0 else "#e2e8f0"
                fg  = "white" if cnt > 0 else "#94a3b8"
                tags += (
                    f'<span style="background:{bg};color:{fg};border-radius:3px;'
                    f'padding:1px 5px;font-size:0.62rem;font-weight:600;white-space:nowrap;">'
                    f'{_type_lbl[cls_id]}:{cnt}</span>'
                )
            dir_cards += (
                f'<div style="background:#f1f5f9;border-radius:7px;'
                f'padding:6px 8px;text-align:center;min-width:80px;">'
                f'<div style="font-size:0.8rem;font-weight:700;color:#475569">{label}</div>'
                f'<div style="font-size:1.5rem;font-weight:700;color:#1e293b;line-height:1.1">{total}</div>'
                f'<div style="display:flex;flex-wrap:wrap;gap:2px;justify-content:center;margin-top:3px">{tags}</div>'
                f'</div>'
            )

        outer_cards.append(
            f'<div style="flex:1;background:#fff;border:1.5px solid #e2e8f0;'
            f'border-radius:12px;padding:12px 14px;">'
            f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">'
            f'<span style="font-size:0.82rem;font-weight:700;color:#334155">'
            f'{sig_dot}{_tls_title[tls_id]}</span>'
            f'<span style="font-size:0.72rem;color:{sig_color};font-weight:600">'
            f'{phase_lbl} &bull; {total_all} veh</span>'
            f'</div>'
            f'<div style="display:flex;gap:6px;flex-wrap:wrap;">{dir_cards}</div>'
            f'</div>'
        )

    return (
        '<div style="display:flex;gap:10px;padding:2px 0;">'
        + "".join(outer_cards)
        + "</div>"
    )


# =============================================================================
# Camera Dashboard
# =============================================================================

def show_camera_dashboard(
    cam_cfg_path: str,
    yolo_model_path: str,
    use_yolo: bool,
    conf_thresh: float,
    tls_phases: Dict[str, int],
    target_fps: int,
    use_mappo: bool,
    mappo_path: str,
    safe_mode_threshold: float = 0.25,
    safe_mode_green_sec: float = 30.0,
    force_safe_mode: bool = False,
) -> None:
    # ── load camera config ────────────────────────────────────────────────────
    cfg_path = Path(cam_cfg_path)
    if not cfg_path.exists():
        st.error(f"Camera config not found: `{cam_cfg_path}`")
        return

    with open(cfg_path) as f:
        cam_cfg = json.load(f)

    cameras = [c for c in cam_cfg["cameras"] if c.get("enabled", True)]
    n_cams  = len(cameras)

    # Build phase_group mapping: {tls_id: {"A": [cam_indices], "B": [cam_indices]}}
    # phase_group A → phase 0 (N+S get green); B → phase 1 (E+W get green)
    tls_phase_groups: Dict[str, Dict[str, List[int]]] = {}
    for ci, cam in enumerate(cameras):
        tid = cam.get("tls_id", "tls_0")
        pg  = cam.get("phase_group", "A")
        if tid not in tls_phase_groups:
            tls_phase_groups[tid] = {"A": [], "B": []}
        tls_phase_groups[tid][pg].append(ci)

    # ── load YOLO ─────────────────────────────────────────────────────────────
    yolo = None
    if use_yolo:
        yolo_p = Path(yolo_model_path)
        if not yolo_p.exists():
            st.warning(f"YOLO model not found at `{yolo_model_path}` — running without detection.")
            use_yolo = False
        else:
            yolo = _load_yolo(str(yolo_p))

    # ── load MAPPO policy + StateExtractor ────────────────────────────────────
    policy         = None
    state_extractor = None
    if use_mappo:
        mappo_p = Path(mappo_path)
        if not mappo_p.exists():
            st.warning(f"MAPPO policy not found at `{mappo_path}` — using simulated phase cycle.")
            use_mappo = False
        else:
            policy          = _load_mappo_policy(str(mappo_p))
            state_extractor = _build_state_extractor(_DEFAULT_STATE_CFG)
            if policy is None or state_extractor is None:
                st.warning("MAPPO init failed — using simulated phase cycle.")
                use_mappo = False

    # ── open video captures ───────────────────────────────────────────────────
    caps: List[Optional[cv2.VideoCapture]] = []
    for cam in cameras:
        src = str(_ROOT / cam["source"])
        cap = cv2.VideoCapture(src)
        caps.append(cap if cap.isOpened() else None)
        if not cap.isOpened():
            st.warning(f"Cannot open `{src}` — slot will show placeholder.")

    # ── accident event detector (Telegram alerts + image save) ───────────────
    try:
        from src.vision.event_detector import AccidentEventDetector
        _accident_detector = AccidentEventDetector(
            tele_config_path=str(_ROOT / "configs" / "tele.json"),
            accident_class_id=0,
            accident_conf_thresh=0.7,
            confirm_frames=3,
            cooldown_sec=60.0,
            position="INTERSECTION_DASHBOARD",
            enabled=True,
            event_dir=str(_ROOT / "logs" / "events"),
        )
    except Exception as _e:
        st.warning(f"AccidentEventDetector init failed: {_e}")
        _accident_detector = None

    # ── UI layout ─────────────────────────────────────────────────────────────
    st.markdown("### Camera Feed")

    COLS_PER_ROW = 4
    rows = [st.columns(COLS_PER_ROW) for _ in range((n_cams + COLS_PER_ROW - 1) // COLS_PER_ROW)]
    img_slots: List[Any]   = []
    label_slots: List[Any] = []
    for idx in range(n_cams):
        r, c = divmod(idx, COLS_PER_ROW)
        img_slots.append(rows[r][c].empty())
        label_slots.append(rows[r][c].empty())

    st.markdown("---")
    inference_mode = "MAPPO" if use_mappo else "Simulated cycle"
    st.markdown(f"### Traffic Light Status  <small style='color:#64748b;font-size:0.8rem'>({inference_mode})</small>",
                unsafe_allow_html=True)
    tls_placeholder  = st.empty()
    safe_mode_banner = st.empty()

    st.markdown("---")
    st.markdown("### Per-Intersection Traffic")
    dir_placeholder = st.empty()

    st.markdown("---")
    st.markdown("### Detection Summary")
    det_cols = st.columns(5)
    det_slots = {
        0: det_cols[0].empty(),
        2: det_cols[1].empty(),
        3: det_cols[2].empty(),
        1: det_cols[3].empty(),
        4: det_cols[4].empty(),
    }

    accident_banner = st.empty()

    # ── Arduino status strip ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Arduino")
    _ard_cols = st.columns([1, 1, 2])
    _ard_status  = _ard_cols[0].empty()   # connected / disconnected
    _ard_stats   = _ard_cols[1].empty()   # send / fail counts
    _ard_lastcmd = _ard_cols[2].empty()   # last RYG packet sent

    st.markdown("---")
    progress_bar = st.progress(0.0)
    status_text  = st.empty()

    # ── TLS state ─────────────────────────────────────────────────────────────
    YELLOW_DURATION_S = 3.0    # exactly 3 real seconds of yellow, regardless of fps
    MIN_GREEN_S       = 10.0   # minimum green hold before any switch is allowed (matches training)
    _phase_hold_s     = 15.0   # simulated: force switch after 15 real seconds
    _mappo_max_s      = 30.0   # MAPPO safety: force switch if stuck > 30 real seconds

    # tls_0 and tls_1 start at OPPOSITE phases so they are always in opposition
    init_0 = int(tls_phases.get("tls_0", 0))
    _now_init = time.perf_counter()
    tls_state: Dict[str, Dict[str, Any]] = {
        "tls_0": {"phase": init_0,     "next_phase": None, "yellow_start": None, "phase_start": _now_init},
        "tls_1": {"phase": 1 - init_0, "next_phase": None, "yellow_start": None, "phase_start": _now_init},
    }

    # ── playback loop (infinite) ──────────────────────────────────────────────
    frame_delay  = 1.0 / max(target_fps, 1)
    any_accident = False
    frame_idx          = 0
    _last_serial_states: List[int] = []   # track last sent packet to avoid redundant sends
    _last_serial_send_t: float     = 0.0  # time of last send (for 1-second heartbeat)
    _safe_mode_active:       bool  = False
    _safe_mode_phase_start:  float = 0.0
    _last_frame_times: Dict[int, float] = {i: time.perf_counter() for i in range(n_cams)}

    while True:
        t_start = time.perf_counter()

        global_counts: Dict[int, int]       = {k: 0 for k in CLASS_NAMES}
        cam_accident_flags: List[bool]       = []
        parsed_frames: Dict[int, Dict]       = {}
        annotated_frames: List[Optional[np.ndarray]] = [None] * n_cams

        # ── read + detect all cameras ─────────────────────────────────────────
        for cam_idx, (cam, cap) in enumerate(zip(cameras, caps)):
            if cap is None:
                ph = np.zeros((240, 320, 3), dtype=np.uint8)
                cv2.putText(ph, "NO SIGNAL", (60, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)
                annotated_frames[cam_idx] = ph
                cam_accident_flags.append(False)
                parsed_frames[cam_idx] = {"tracks": [], "accident_found": False, "cam_id": cam_idx}
                continue

            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            if ret:
                _last_frame_times[cam_idx] = t_start
            else:
                frame = np.zeros((240, 320, 3), dtype=np.uint8)

            frame_sm = cv2.resize(frame, (320, 240))
            tracks:         List[Dict] = []
            accident_in_cam            = False
            cam_counts:     Dict[int, int] = {}

            if use_yolo and yolo is not None:
                results = yolo.predict(frame_sm, conf=conf_thresh, verbose=False, device="cpu")
                annotated, accident_in_cam = _draw_boxes(
                    frame_sm, results[0].boxes, conf_thresh, cam_counts
                )
                orig_h, orig_w = frame.shape[:2]
                tracks = _boxes_to_tracks(
                    results[0].boxes, cam_idx, conf_thresh, t_start,
                    src_w=320, src_h=240, dst_w=orig_w, dst_h=orig_h,
                )
                for cls_id, cnt in cam_counts.items():
                    global_counts[cls_id] = global_counts.get(cls_id, 0) + cnt
            else:
                annotated = frame_sm

            # Camera label overlay
            direction = cam.get("direction", "")
            icon      = DIRECTION_ICON.get(direction, "CAM")
            cam_label = f"{icon} {cam['id'].upper()}  {cam.get('name', '')}"
            cv2.rectangle(annotated, (0, 0), (len(cam_label) * 9 + 8, 22), (20, 20, 20), -1)
            cv2.putText(annotated, cam_label, (4, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 180), 1)

            annotated_frames[cam_idx] = annotated
            cam_accident_flags.append(accident_in_cam)

            # Find best accident confidence from boxes
            _acc_conf = 0.0
            if use_yolo and yolo is not None and accident_in_cam:
                for _box in results[0].boxes:
                    if int(_box.cls[0].item()) == 0:
                        _acc_conf = max(_acc_conf, float(_box.conf[0].item()))

            parsed_frames[cam_idx] = {
                "tracks":         tracks,
                "accident_found": accident_in_cam,
                "accident_conf":  _acc_conf,
                "cam_id":         cam_idx,
                "name":           cam.get("id", str(cam_idx)),
                "timestamp":      t_start,
                "frame_count":    frame_idx,
                "vehicle_count":  sum(c for k, c in cam_counts.items() if k != 0),
                "annotated_frame": annotated if accident_in_cam else None,
            }

        # ── accident Telegram alert (3-frame confirm + image) ────────────────
        if _accident_detector is not None:
            try:
                _accident_detector.update(parsed_frames)
            except Exception:
                pass

        # ── commit yellow transitions (time-based, not frame-based) ─────────
        _now = time.perf_counter()
        for ts in tls_state.values():
            if (ts["yellow_start"] is not None
                    and (_now - ts["yellow_start"]) >= YELLOW_DURATION_S
                    and ts["next_phase"] is not None):
                ts["phase"]        = ts["next_phase"]
                ts["next_phase"]   = None
                ts["yellow_start"] = None
                ts["phase_start"]  = _now  # reset hold timer on green start

        # ── Stale camera check → SAFE MODE ───────────────────────────────────
        _stale_count = sum(
            1 for i, cap in enumerate(caps)
            if cap is None or (t_start - _last_frame_times.get(i, 0.0)) > 5.0
        )
        _in_safe_mode = force_safe_mode or (
            n_cams > 0 and _stale_count / n_cams > safe_mode_threshold
        )

        if _in_safe_mode:
            # ── SAFE MODE: fixed-time cycling (safe_mode_green_sec / phase) ───
            if not _safe_mode_active:
                _safe_mode_active = True
                _safe_mode_phase_start = time.perf_counter()
            elif time.perf_counter() - _safe_mode_phase_start >= safe_mode_green_sec:
                for _tid, _ts in tls_state.items():
                    if _ts["yellow_start"] is None and _ts["next_phase"] is None:
                        _ts["next_phase"]   = 1 - _ts["phase"]
                        _ts["yellow_start"] = time.perf_counter()
                _safe_mode_phase_start = time.perf_counter()
        else:
            if _safe_mode_active:
                _safe_mode_active = False

            # ── Layer 1: pressure from raw vehicle counts ─────────────────────
            pressure_preferred: Dict[str, int] = {}
            pressure_counts: Dict[str, Tuple[int, int]] = {}
            for tls_id, groups in tls_phase_groups.items():
                cnt_a = sum(
                    sum(1 for t in parsed_frames.get(ci, {}).get("tracks", [])
                        if t.get("cls_id", -1) != 0)
                    for ci in groups.get("A", [])
                )
                cnt_b = sum(
                    sum(1 for t in parsed_frames.get(ci, {}).get("tracks", [])
                        if t.get("cls_id", -1) != 0)
                    for ci in groups.get("B", [])
                )
                pressure_preferred[tls_id] = 0 if cnt_a >= cnt_b else 1
                pressure_counts[tls_id] = (cnt_a, cnt_b)

            # ── Layer 2: MAPPO — primary controller ───────────────────────────
            if use_mappo and policy is not None and state_extractor is not None:
                _phase_map_m = {
                    tid: _PHASE_ACTION_MAP[ts["phase"]]
                    for tid, ts in tls_state.items()
                }
                _green_timers_m = {
                    tid: _now - ts["phase_start"]
                    for tid, ts in tls_state.items()
                }
                try:
                    obs52 = state_extractor.build_state(
                        parsed_frames, _phase_map_m, _green_timers_m
                    )
                    actions = policy.predict(
                        {"tls_0": obs52[0:26], "tls_1": obs52[26:52]},
                        deterministic=True,
                    )
                    for tid, new_act in actions.items():
                        ts = tls_state[tid]
                        green_held = _now - ts["phase_start"]
                        if new_act == ts["phase"] or ts["yellow_start"] is not None:
                            continue
                        if green_held < MIN_GREEN_S:
                            continue
                        ts["next_phase"]   = new_act
                        ts["yellow_start"] = time.perf_counter()
                except Exception:
                    pass

            # ── Layer 3: pressure override (only when MAPPO is disabled) ──────
            if not use_mappo:
                _now3 = time.perf_counter()
                for tls_id, pref_phase in pressure_preferred.items():
                    ts = tls_state[tls_id]
                    green_held = _now3 - ts["phase_start"]
                    if (ts["yellow_start"] is None
                            and ts["next_phase"] is None
                            and pref_phase != ts["phase"]
                            and green_held >= MIN_GREEN_S):
                        ts["next_phase"]   = pref_phase
                        ts["yellow_start"] = time.perf_counter()

            # ── Layer 4: safety timer fallback ────────────────────────────────
            _hold_s = _phase_hold_s if not use_mappo else _mappo_max_s
            _now4   = time.perf_counter()
            for tid, ts in tls_state.items():
                green_held = _now4 - ts["phase_start"]
                if (ts["yellow_start"] is None
                        and ts["next_phase"] is None
                        and green_held >= _hold_s):
                    ts["next_phase"]   = 1 - ts["phase"]
                    ts["yellow_start"] = time.perf_counter()

        # ── render TLS SVG ────────────────────────────────────────────────────
        display = {
            tid: {"phase": ts["phase"], "yellow": ts["yellow_start"] is not None}
            for tid, ts in tls_state.items()
        }
        tls_placeholder.markdown(
            render_twin_intersection_html(display),
            unsafe_allow_html=True,
        )

        # ── SAFE MODE banner ──────────────────────────────────────────────────
        if _in_safe_mode:
            _reason = "forced" if force_safe_mode else f"{_stale_count}/{n_cams} cameras stale"
            safe_mode_banner.markdown(
                f"<div style='background:linear-gradient(135deg,#d97706,#b45309);"
                f"color:#ffffff;font-size:1rem;font-weight:700;padding:12px 20px;"
                f"border-radius:10px;text-align:center;border-left:4px solid #fde68a;'>"
                f"SAFE MODE &mdash; {_reason}"
                f"&nbsp;|&nbsp; Fixed-cycle {safe_mode_green_sec:.0f}s/phase"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            safe_mode_banner.empty()

        # ── Arduino: send RYG command + update status strip ─────────────────
        _bridge = st.session_state.get("serial_bridge")
        _serial_states = _tls_states_to_serial(tls_state)
        _ryg_labels    = {_S_RED: "R", _S_YELLOW: "Y", _S_GREEN: "G"}
        _cmd_str = "  ".join(
            f"{'tls_0' if i < 4 else 'tls_1'}[{i % 4}]={_ryg_labels[v]}"
            for i, v in enumerate(_serial_states)
        )
        _now_serial = time.perf_counter()
        _state_changed  = (_serial_states != _last_serial_states)
        _heartbeat_due  = (_now_serial - _last_serial_send_t) >= 1.0
        if _bridge is not None and _bridge.is_healthy():
            if _state_changed or _heartbeat_due:
                _bridge.pack_and_send_data(_serial_states, retry_on_fail=False)
                _last_serial_states = _serial_states
                _last_serial_send_t = _now_serial
            _st = _bridge.get_stats()
            _ard_status.success("Connected", icon="🟢")
            _ard_stats.metric("Sent / Fail", f"{_st['send_count']} / {_st['fail_count']}")
        elif _bridge is not None:
            _ard_status.warning("Unhealthy", icon="🟡")
            _ard_stats.empty()
        else:
            _ard_status.error("No Arduino", icon="🔴")
            _ard_stats.empty()
        _ard_lastcmd.caption(f"Last packet: `{_cmd_str}`")

        # ── per-intersection vehicle counts ──────────────────────────────────
        # cameras 0..3 → tls_0, cameras 4..7 → tls_1
        _mid = max(n_cams // 2, 1)
        _cam_to_tls = {i: ("tls_0" if i < _mid else "tls_1") for i in range(n_cams)}
        int_counts: Dict[str, Dict[str, Dict[int, int]]] = {
            "tls_0": {},
            "tls_1": {},
        }
        for _ci, _cam in enumerate(cameras):
            _tls = _cam_to_tls.get(_ci, "tls_0")
            _dir = _cam.get("direction", "unknown")
            if _dir not in int_counts[_tls]:
                int_counts[_tls][_dir] = {}
            for _t in parsed_frames.get(_ci, {}).get("tracks", []):
                _cid = _t.get("cls_id", -1)
                if _cid in CLASS_NAMES and _cid != 0:
                    int_counts[_tls][_dir][_cid] = int_counts[_tls][_dir].get(_cid, 0) + 1
        dir_placeholder.markdown(
            _render_intersection_panel(int_counts, display),
            unsafe_allow_html=True,
        )

        # ── push camera frames ────────────────────────────────────────────────
        for cam_idx, af in enumerate(annotated_frames):
            if af is not None:
                img_slots[cam_idx].image(
                    cv2.cvtColor(af, cv2.COLOR_BGR2RGB),
                    use_container_width=True,
                )

        # ── detection counters ────────────────────────────────────────────────
        det_slots[0].metric("Accidents",    global_counts.get(0, 0))
        det_slots[2].metric("Cars",         global_counts.get(2, 0))
        det_slots[3].metric("Motorcycles",  global_counts.get(3, 0))
        det_slots[1].metric("Buses",        global_counts.get(1, 0))
        det_slots[4].metric("Trucks",       global_counts.get(4, 0))

        # ── accident alert ────────────────────────────────────────────────────
        if any(cam_accident_flags):
            any_accident = True
            involved     = [cameras[i]["id"] for i, f in enumerate(cam_accident_flags) if f]
            accident_banner.markdown(
                f"<div class='accident-banner'>"
                f"ACCIDENT DETECTED - {', '.join(involved).upper()}"
                f"</div>",
                unsafe_allow_html=True,
            )
        elif not any_accident:
            accident_banner.empty()

        # ── status + phase progress bar ───────────────────────────────────────
        _now_s  = time.perf_counter()
        _hold_s = (
            safe_mode_green_sec if _in_safe_mode
            else (_phase_hold_s if not use_mappo else _mappo_max_s)
        )
        # Show progress of the intersection that has been green the longest
        _ph_elapsed = max(_now_s - ts["phase_start"] for ts in tls_state.values()
                          if ts["yellow_start"] is None)  if any(
                              ts["yellow_start"] is None for ts in tls_state.values()
                          ) else 0.0
        progress_bar.progress(
            min(_ph_elapsed / _hold_s, 1.0),
            text=f"Green hold: {_ph_elapsed:.0f}s / {_hold_s:.0f}s (min green: {MIN_GREEN_S:.0f}s)",
        )
        _p0 = tls_state["tls_0"]["phase"]
        _p1 = tls_state["tls_1"]["phase"]
        _y0 = "Y" if tls_state["tls_0"]["yellow_start"] is not None else ("A" if _p0 == 0 else "B")
        _y1 = "Y" if tls_state["tls_1"]["yellow_start"] is not None else ("A" if _p1 == 0 else "B")
        _ctrl_mode = (
            f"SAFE MODE ({_stale_count}/{n_cams} stale)" if _in_safe_mode
            else ("MAPPO" if use_mappo else "Simulated")
        )
        status_text.caption(
            f"Frame {frame_idx + 1} | "
            f"tls_0: {_y0} | tls_1: {_y1} | "
            f"{'YOLO ON' if use_yolo else 'YOLO OFF'} | "
            f"{_ctrl_mode}"
        )

        # Frame-rate throttle
        elapsed = time.perf_counter() - t_start
        sleep_t = max(0.0, frame_delay - elapsed)
        if sleep_t > 0:
            time.sleep(sleep_t)

        frame_idx += 1

    # Cleanup (reached only if loop is broken externally)
    for cap in caps:
        if cap is not None:
            cap.release()


# =============================================================================
# Live Simulation (SUMO)
# =============================================================================

def _update_sim_charts(
    records, current_actions, tls_ids,
    progress_bar, status_bar,
    kpi_queue, kpi_speed, kpi_wait, kpi_passed,
    queue_chart, speed_chart, wait_chart, pass_chart,
    tls_displays, n_steps, step,
    halted, speed, waiting, passed,
) -> None:
    frac = min((step + 1) / n_steps, 1.0)
    progress_bar.progress(frac)
    status_bar.info(
        f"Step **{step + 1}/{n_steps}** | "
        f"Queue: **{halted:.1f} veh** | "
        f"Speed: **{speed:.2f} m/s** | "
        f"Waiting: **{waiting:.2f} s**"
    )
    kpi_queue.metric("Queue (halted veh)",  f"{halted:.1f}")
    kpi_speed.metric("Avg Speed (m/s)",     f"{speed:.2f}")
    kpi_wait.metric("Step Waiting (s)",     f"{waiting:.2f}")
    kpi_passed.metric("Throughput (veh)",   f"{passed:.0f}")

    df = pd.DataFrame(records)
    queue_chart.line_chart(df.set_index("step")[["queue"]], use_container_width=True)
    speed_chart.line_chart(df.set_index("step")[["speed"]], use_container_width=True)
    wait_chart.line_chart(df.set_index("step")[["cumulative_wait"]], use_container_width=True)
    pass_chart.line_chart(df.set_index("step")[["throughput"]], use_container_width=True)

    for i, tid in enumerate(sorted(tls_ids)):
        phase = current_actions.get(tid, 0)
        label = "Group A GREEN" if phase == 0 else "Group B GREEN"
        tls_displays[i].markdown(f"**{tid}**\n\n{label}")


def run_simulation(
    controller, n_steps, seed, sumo_cfg_path, model_path,
    phi_min, phi_max, kappa,
) -> None:
    UPDATE_EVERY = 5

    try:
        from src.traffic_env.config import build_default_config
        from src.traffic_env.envs.multi_agent import MappoTrafficEnv
    except ImportError as exc:
        st.error(f"Import error — is SUMO_HOME set?\n{exc}")
        return

    if not Path(sumo_cfg_path).exists():
        st.error(f"SUMO config not found: `{sumo_cfg_path}`")
        return

    with st.spinner("Initialising SUMO..."):
        cfg = build_default_config(sumo_cfg_path=sumo_cfg_path, gui=False, max_steps=n_steps)
    tls_ids = sorted(cfg.tls_ids)

    policy = mp_policy = sotl_policy = None
    if controller == "MAPPO":
        if not Path(model_path).exists():
            st.error(f"Checkpoint not found: `{model_path}`")
            return
        with st.spinner("Loading MAPPO checkpoint..."):
            from src.core.policy_loader import PolicyLoader
            policy = PolicyLoader(checkpoint_path=model_path,
                                  obs_dim=cfg.local_obs_dim, action_dim=2).load()
    elif controller == "Max Pressure":
        from experiment.baselines.max_pressure import MaxPressurePolicy
        lg = cfg.get_lane_groups()
        mp_policy = MaxPressurePolicy({tid: lg[tid] for tid in tls_ids if tid in lg})
    elif controller == "SOTL":
        from experiment.baselines.sotl import SOTLPolicy
        lg = cfg.get_lane_groups()
        sotl_policy = SOTLPolicy(
            lane_groups={tid: lg[tid] for tid in tls_ids if tid in lg},
            phi_min=phi_min, phi_max=phi_max, kappa=kappa,
        )

    status_bar   = st.empty()
    progress_bar = st.progress(0.0)
    st.markdown("#### Key Metrics")
    kpi_c = st.columns(4)
    kpi_queue  = kpi_c[0].empty()
    kpi_speed  = kpi_c[1].empty()
    kpi_wait   = kpi_c[2].empty()
    kpi_passed = kpi_c[3].empty()

    st.markdown("#### Time-Series Charts")
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Queue Length (halted vehicles)")
        queue_chart = st.empty()
    with c2:
        st.caption("Average Lane Speed (m/s)")
        speed_chart = st.empty()
    c3, c4 = st.columns(2)
    with c3:
        st.caption("Cumulative Waiting (s)")
        wait_chart = c3.empty()
    with c4:
        st.caption("Throughput (vehicles passed)")
        pass_chart = c4.empty()

    st.markdown("#### Traffic Light Phases")
    tls_disp = [c.empty() for c in st.columns(len(tls_ids))]

    env = MappoTrafficEnv(config=cfg, gui=False, use_libsumo=True)
    obs, _ = env.reset(seed=int(seed))
    if sotl_policy:
        sotl_policy.reset()

    records = {"step": [], "queue": [], "speed": [], "cumulative_wait": [], "throughput": []}
    cumulative_wait = 0.0
    current_actions = {tid: 0 for tid in tls_ids}
    ft_counter = 0

    try:
        for step in range(n_steps):
            if controller == "MAPPO":
                current_actions = policy.predict(
                    {tid: np.asarray(obs[tid], dtype=np.float32) for tid in tls_ids},
                    deterministic=True)
            elif controller == "Max Pressure":
                lc = env.obs_builder.build_lane_cache(env.tls_ids)
                current_actions = mp_policy.select_actions(lc)
            elif controller == "SOTL":
                lc = env.obs_builder.build_lane_cache(env.tls_ids)
                current_actions = sotl_policy.select_actions(lc)
            else:
                phase = (ft_counter // 60) % 2
                current_actions = {tid: phase for tid in tls_ids}
                ft_counter += 1

            obs, _, terminated, truncated, info = env.step(current_actions)
            done = any(terminated.values()) or any(truncated.values())

            lc_post = env.obs_builder.build_lane_cache(env.tls_ids)
            halted  = sum(float(info.get(tid, {}).get("queue_total_proxy",  0.0)) for tid in tls_ids)
            waiting = float(np.mean([float(info.get(tid, {}).get("avg_waiting_proxy", 0.0)) for tid in tls_ids]))
            speeds  = [float(m.get("raw_speed", 0.0)) for m in lc_post.values() if m]
            speed   = float(np.mean(speeds)) if speeds else 0.0
            passed  = float(np.mean([float(info.get(tid, {}).get("current_passed", 0.0)) for tid in tls_ids]))
            cumulative_wait += waiting

            records["step"].append(step)
            records["queue"].append(halted)
            records["speed"].append(speed)
            records["cumulative_wait"].append(cumulative_wait)
            records["throughput"].append(passed)

            if step % UPDATE_EVERY == 0 or done:
                _update_sim_charts(
                    records, current_actions, tls_ids,
                    progress_bar, status_bar,
                    kpi_queue, kpi_speed, kpi_wait, kpi_passed,
                    queue_chart, speed_chart, wait_chart, pass_chart,
                    tls_disp, n_steps, step, halted, speed, waiting, passed,
                )
            if done:
                break
    finally:
        env.close()

    progress_bar.progress(1.0)
    status_bar.success(f"Simulation complete — {len(records['step'])} steps.")
    st.markdown("---")
    st.subheader("Final Summary")
    s = st.columns(4)
    s[0].metric("Total Waiting (s)", f"{records['cumulative_wait'][-1]:,.1f}")
    s[1].metric("Mean Queue (veh)",  f"{float(np.mean(records['queue'])):.2f}")
    s[2].metric("Mean Speed (m/s)",  f"{float(np.mean(records['speed'])):.2f}")
    s[3].metric("Final Throughput",  f"{records['throughput'][-1]:.0f} veh")
    st.download_button(
        "Download CSV",
        data=pd.DataFrame(records).to_csv(index=False),
        file_name=f"sim_{controller.lower().replace(' ','_')}_seed{seed}.csv",
        mime="text/csv",
    )


# =============================================================================
# Training Curves
# =============================================================================

def show_training_curves(run_name: str, smooth_window: int) -> None:
    run_dir = _ROOT / "logs" / "rl" / run_name
    ep_csv  = run_dir / "train_episodes.csv"
    up_csv  = run_dir / "train_updates.csv"

    if not ep_csv.exists():
        st.error(f"File not found: `{ep_csv}`")
        return

    ep_df = pd.read_csv(ep_csv)

    def smooth(s: pd.Series) -> pd.Series:
        return s.rolling(window=smooth_window, min_periods=1).mean()

    st.subheader("Episode Metrics")
    st.caption(f"Run `{run_name}` — {len(ep_df)} episodes — smoothing window: {smooth_window}")

    ec1, ec2 = st.columns(2)
    with ec1:
        st.caption("Mean Reward per Episode")
        rdf = ep_df[["episode", "mean_reward"]].copy()
        rdf["smoothed"] = smooth(rdf["mean_reward"])
        st.line_chart(rdf.set_index("episode"), use_container_width=True)
    with ec2:
        st.caption("Throughput (vehicles) per Episode")
        tdf = ep_df[["episode", "throughput"]].copy()
        tdf["smoothed"] = smooth(tdf["throughput"])
        st.line_chart(tdf.set_index("episode"), use_container_width=True)

    ec3, ec4 = st.columns(2)
    with ec3:
        st.caption("Queue Total (halted vehicle-steps)")
        qdf = ep_df[["episode", "queue_total_proxy"]].copy()
        qdf["smoothed"] = smooth(qdf["queue_total_proxy"])
        st.line_chart(qdf.set_index("episode"), use_container_width=True)
    with ec4:
        st.caption("Average Waiting Proxy (s)")
        wdf = ep_df[["episode", "avg_waiting_proxy"]].copy()
        wdf["smoothed"] = smooth(wdf["avg_waiting_proxy"])
        st.line_chart(wdf.set_index("episode"), use_container_width=True)

    if up_csv.exists():
        st.markdown("---")
        st.subheader("Update Metrics")
        up_df = pd.read_csv(up_csv)
        uc1, uc2, uc3 = st.columns(3)
        with uc1:
            st.caption("Policy Loss")
            pl = up_df[["update_idx", "policy_loss"]].copy()
            pl["smoothed"] = smooth(pl["policy_loss"])
            st.line_chart(pl.set_index("update_idx"), use_container_width=True)
        with uc2:
            st.caption("Value Loss")
            vl = up_df[["update_idx", "value_loss"]].copy()
            vl["smoothed"] = smooth(vl["value_loss"])
            st.line_chart(vl.set_index("update_idx"), use_container_width=True)
        with uc3:
            st.caption("Entropy")
            en = up_df[["update_idx", "entropy"]].copy()
            en["smoothed"] = smooth(en["entropy"])
            st.line_chart(en.set_index("update_idx"), use_container_width=True)

    st.markdown("---")
    st.subheader("Summary Statistics")
    st.dataframe(
        ep_df[["mean_reward", "throughput", "queue_total_proxy", "avg_waiting_proxy"]]
        .describe().round(3),
        use_container_width=True,
    )
    with st.expander("Raw episode data"):
        st.dataframe(ep_df, use_container_width=True)


# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.title("Traffic-Guard-AI")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["Camera Dashboard", "Live Simulation", "Training Curves"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    # ── Camera Dashboard controls ──────────────────────────────────────────────
    if page == "Camera Dashboard":
        st.subheader("Camera Settings")

        cam_cfg_path = st.text_input(
            "Camera Config",
            str(_ROOT / "configs" / "camera_config.json"),
        )
        yolo_model_path = st.text_input(
            "YOLO Model",
            str(_ROOT / "models" / "yolo" / "yolov11.pt"),
        )
        use_yolo    = st.toggle("YOLO Detection", value=True)
        conf_thresh = st.slider("Confidence Threshold", 0.10, 0.90, 0.45, 0.05,
                                disabled=not use_yolo)
        target_fps  = st.slider("Playback FPS", 1, 15, 5)


        st.markdown("---")
        st.subheader("MAPPO Inference")
        use_mappo  = st.toggle("Real-time MAPPO", value=True)
        mappo_path = st.text_input(
            "Policy Checkpoint",
            _DEFAULT_POLICY,
            disabled=not use_mappo,
            key="mappo_path_input",
        )

        st.markdown("---")
        st.subheader("Arduino Controller")

        _available_ports = _list_com_ports()
        _bridge_now      = st.session_state.serial_bridge

        # Status badge
        if _bridge_now is not None and _bridge_now.is_healthy():
            st.success(f"Connected — {_bridge_now.port}", icon="🟢")
            _stats = _bridge_now.get_stats()
            st.caption(
                f"Sent: **{_stats['send_count']}**  "
                f"Fail: **{_stats['fail_count']}**  "
                f"Reconnects: **{_stats['reconnect_attempts']}**"
            )
        elif _bridge_now is not None:
            st.warning("Port open but unhealthy — try reconnecting.", icon="🟡")
        else:
            st.error("Not connected", icon="🔴")

        # Port selector — preselect currently connected port if any
        _default_port = _bridge_now.port if _bridge_now is not None else (
            _available_ports[0] if _available_ports else "COM3"
        )
        _port_options = _available_ports if _default_port in _available_ports else (
            [_default_port] + _available_ports
        )
        arduino_port = st.selectbox(
            "COM Port",
            _port_options,
            index=_port_options.index(_default_port),
            key="arduino_port_select",
        )
        arduino_baud = st.selectbox(
            "Baud Rate", [115200, 9600, 57600], index=0, key="arduino_baud_select"
        )

        _col_conn, _col_disc = st.columns(2)
        if _col_conn.button("Connect", use_container_width=True, key="btn_arduino_connect"):
            if not _HAS_SERIAL:
                st.error("pyserial not installed. Run: pip install pyserial")
            else:
                try:
                    if st.session_state.serial_bridge is not None:
                        st.session_state.serial_bridge.close()
                        st.session_state.serial_bridge = None
                    from src.utils.serial_bridge import SerialBridge
                    with st.spinner(f"Connecting to {arduino_port}..."):
                        st.session_state.serial_bridge = SerialBridge(
                            port=arduino_port,
                            baudrate=arduino_baud,
                        )
                    st.success(f"Connected to {arduino_port} @ {arduino_baud}")
                    st.rerun()
                except Exception as _e:
                    st.session_state.serial_bridge = None
                    st.error(f"Connection failed: {_e}")

        if _col_disc.button("Disconnect", use_container_width=True, key="btn_arduino_disconnect"):
            if st.session_state.serial_bridge is not None:
                st.session_state.serial_bridge.close()
                st.session_state.serial_bridge = None
                st.info("Disconnected.")
                st.rerun()

        st.markdown("---")
        st.subheader("Fallback / Initial Phase")
        tls_phases: Dict[str, int] = {}
        for tid in ["tls_0", "tls_1"]:
            tls_phases[tid] = st.selectbox(
                f"{tid} initial phase",
                [0, 1],
                format_func=lambda x: "0 - Group A GREEN" if x == 0 else "1 - Group B GREEN",
                key=f"tls_phase_{tid}",
            )

        st.markdown("---")
        st.subheader("Safe Mode")
        st.caption("Activates when too many cameras lose signal for >5 s.")

        _sm_pct = st.slider(
            "Trigger threshold",
            min_value=10, max_value=75, value=25, step=5,
            format="%d%%",
            help="Safe mode activates when this % of cameras become stale.",
            key="sm_threshold_pct",
        )
        safe_mode_threshold = _sm_pct / 100.0

        safe_mode_green_sec = float(st.slider(
            "Phase duration (s)",
            min_value=10, max_value=90, value=30, step=5,
            help="How long each phase lasts during fixed-cycle safe mode.",
            key="sm_green_sec",
        ))

        force_safe_mode = st.checkbox(
            "Force Safe Mode",
            value=False,
            help="Manually activate fixed-cycle control to test safe mode behaviour.",
            key="sm_force",
        )
        if force_safe_mode:
            st.warning("Safe Mode forced ON — MAPPO disabled.", icon="⚠️")
        else:
            st.caption("Status shown in main area during playback.")

        st.markdown("---")
        run_btn = st.button("Play Cameras", type="primary", use_container_width=True)

    # ── Live Simulation controls ───────────────────────────────────────────────
    elif page == "Live Simulation":
        st.subheader("Simulation Settings")

        controller = st.selectbox("Controller",
                                  ["MAPPO", "Max Pressure", "SOTL", "Fixed-Time"])
        n_steps = st.slider("Max Steps", 100, 1080, 400, step=50)
        seed    = st.number_input("Seed", 0, 9999, 42)

        sumo_cfg_path = st.text_input(
            "SUMO Config",
            str(_ROOT / "sumo_configs" / "evaluation" / "medium" / "sumo_config.sumocfg"),
        )
        model_path = str(_ROOT / "models" / "mappo" / "20260418_215140" / "best_model.pt")
        if controller == "MAPPO":
            model_path = st.text_input("Model Checkpoint", model_path)

        phi_min, phi_max, kappa = 2, 12, 5
        if controller == "SOTL":
            phi_min = st.number_input("phi_min", 1, 10, 2)
            phi_max = st.number_input("phi_max", 5, 24, 12)
            kappa   = st.number_input("kappa",   1, 30,  5)

        st.markdown("---")
        run_btn = st.button("Run Simulation", type="primary", use_container_width=True)

    # ── Training Curves controls ───────────────────────────────────────────────
    else:
        st.subheader("Run Selection")
        rl_dir        = _ROOT / "logs" / "rl"
        runs          = sorted([d.name for d in rl_dir.iterdir() if d.is_dir()]) if rl_dir.exists() else []
        selected_run  = st.selectbox("Training Run", runs if runs else ["(no runs found)"])
        smooth_window = st.slider("Smoothing window", 1, 30, 5)
        run_btn       = False


# =============================================================================
# Router
# =============================================================================

if page == "Camera Dashboard":
    st.title("Real-Time Camera Dashboard")
    st.caption(
        f"Sources: `configs/camera_config.json` &nbsp;|&nbsp; "
        f"YOLO: **{'ON' if use_yolo else 'OFF'}** &nbsp;|&nbsp; "
        f"Conf: **{conf_thresh:.2f}** &nbsp;|&nbsp; "
        f"FPS: **{target_fps}** &nbsp;|&nbsp; "
        f"MAPPO: **{'ON' if use_mappo else 'OFF (simulated)'}**"
    )

    if run_btn:
        show_camera_dashboard(
            cam_cfg_path=cam_cfg_path,
            yolo_model_path=yolo_model_path,
            use_yolo=use_yolo,
            conf_thresh=conf_thresh,
            tls_phases=tls_phases,
            target_fps=target_fps,
            use_mappo=use_mappo,
            mappo_path=mappo_path,
            safe_mode_threshold=safe_mode_threshold,
            safe_mode_green_sec=safe_mode_green_sec,
            force_safe_mode=force_safe_mode,
        )
    else:
        st.info(
            "Configure settings in the sidebar, then click **Play Cameras**.\n\n"
            "Video sources come from `data/raw_video/*.mp4`. "
            "YOLO runs on CPU — lower FPS reduces lag."
        )

        st.markdown("### Camera Grid Layout")
        prev_r1 = st.columns(4)
        prev_r2 = st.columns(4)
        CAM_NAMES = [
            ("cam_0", "north_in",  "N"), ("cam_1", "north_out", "N"),
            ("cam_2", "east_in",   "E"), ("cam_3", "east_out",  "E"),
            ("cam_4", "south_in",  "S"), ("cam_5", "south_out", "S"),
            ("cam_6", "west_in",   "W"), ("cam_7", "west_out",  "W"),
        ]
        all_prev_cols = prev_r1 + prev_r2
        for i, (cam_id, name, icon) in enumerate(CAM_NAMES):
            all_prev_cols[i].markdown(
                f"<div style='background:#f1f5f9;border:1px solid #e2e8f0;border-radius:8px;"
                f"padding:30px 8px;text-align:center;color:#64748b;'>"
                f"<div style='font-size:1.5rem;font-weight:800;color:#94a3b8'>{icon}</div>"
                f"<div style='font-size:0.8rem;margin-top:6px;color:#475569'>{cam_id.upper()}</div>"
                f"<div style='font-size:0.7rem;color:#94a3b8'>{name}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("### Preview — Traffic Light Status")
        st.markdown(
            render_twin_intersection_html({
                "tls_0": {"phase": tls_phases.get("tls_0", 0), "yellow": False},
                "tls_1": {"phase": tls_phases.get("tls_1", 1), "yellow": False},
            }),
            unsafe_allow_html=True,
        )

        st.markdown("### Color Legend")
        leg = st.columns(5)
        legends = [
            ("Accident", "#e74c3c"), ("Car", "#27ae60"),
            ("Motorcycle", "#f0e68c"), ("Bus", "#e67e22"),
            ("Truck", "#9b59b6"),
        ]
        for col, (label, color) in zip(leg, legends):
            col.markdown(
                f"<div style='background:{color};color:white;padding:8px;border-radius:6px;"
                f"text-align:center;font-size:0.85rem;font-weight:bold'>{label}</div>",
                unsafe_allow_html=True,
            )

elif page == "Live Simulation":
    st.title("Live SUMO Simulation")
    st.caption(
        f"Controller: **{controller}** &nbsp;|&nbsp; "
        f"Steps: **{n_steps}** &nbsp;|&nbsp; Seed: **{seed}**"
    )
    if run_btn:
        run_simulation(
            controller=controller, n_steps=n_steps, seed=seed,
            sumo_cfg_path=sumo_cfg_path, model_path=model_path,
            phi_min=phi_min, phi_max=phi_max, kappa=kappa,
        )
    else:
        st.info("Configure settings in the sidebar, then click **Run Simulation**.")

else:
    st.title("Training Curves")
    if runs:
        show_training_curves(run_name=selected_run, smooth_window=smooth_window)
    else:
        st.info(f"No training runs found in `logs/rl/`. Run `train_ppo.py` first.")
