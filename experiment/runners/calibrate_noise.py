from __future__ import annotations

"""
calibrate_noise.py  (paper III-E -> IV-C)
=========================================
Turns measured feature-recovery error into a calibrated sensing-noise config
(configs/noise_config.json) consumed by eval_noise_robustness.py. This is the
bridge that flips NoiseConfig.is_calibrated to True so the VI-E robustness
numbers become paper-ready.

It is SOURCE-AGNOSTIC by design (a real YOLO cannot read SUMO-GUI renders — the
domain gap is total — so we never depend on a specific detection format). You
supply the *measurement* as two row-aligned CSVs and the script computes the per
-feature (bias, sigma) and maps them onto the NoiseConfig parameters with
documented provenance:

    --gt-csv   ground-truth per-(ROI,frame) feature values
    --est-csv  the SAME rows, estimated by your YOLOv11 + ByteTrack stack
        columns (any subset of): effective_queue_norm, occupancy_norm,
        avg_speed_norm, motorbike_share, heavy_vehicle_share

Parameter mapping (model in src/traffic_env/components/obs_noise.py):
    effective_queue_norm  -> queue_calib_sigma   = std(relative error)   [multiplicative]
    occupancy_norm        -> occupancy_bias_sigma= std(relative error)   [multiplicative]
    avg_speed_norm        -> speed_sigma_mps      = std(additive err) * speed_cap
    motorbike/heavy_share -> class_flip_rate      = mean std(additive err) of the two
    (pressure_calib_sigma inherits queue_calib_sigma)

Convenience / fallbacks:
    --confusion-matrix CSV   square count matrix (rows=true, cols=pred) over the
                             custom YOLO classes -> class_flip_rate directly
                             (off-diagonal mass over the VEHICLE classes 1..4).
                             Your ultralytics val run already produces this.
    --set name=value         hard override any NoiseConfig field (repeatable)

Outputs:
    configs/noise_config.json                       calibrated config (+ provenance)
    results/paper1_mappo/iii_e/feature_recovery.csv per-feature bias/sigma/MAE (FIG-1a data)

Usage (from project root):
    python experiment/runners/calibrate_noise.py \
        --gt-csv data/iii_e/gt.csv --est-csv data/iii_e/est.csv \
        --confusion-matrix runs/detect/val/confusion_matrix.csv
    # then:
    python experiment/runners/eval_noise_robustness.py --checkpoint ... --network n3_grid
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.traffic_env.components.obs_noise import NoiseConfig
from src.traffic_env.config import DEFAULT_LANE_FEATURE_NAMES

CONFIG_OUT = _ROOT / "configs" / "noise_config.json"
IIIE_OUT = _ROOT / "results" / "paper1_mappo" / "iii_e" / "feature_recovery.csv"

# custom YOLO vehicle classes (NOTES.md 4.1): 1=bus 2=car 3=moto 4=truck; 0=accident
VEHICLE_CLASS_IDS = (1, 2, 3, 4)


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _read_feature_csv(path: Path) -> Dict[str, np.ndarray]:
    """Read a CSV of feature columns -> {feature: float array}."""
    cols: Dict[str, List[float]] = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                if k is None:
                    continue
                try:
                    cols.setdefault(k.strip(), []).append(float(v))
                except (TypeError, ValueError):
                    cols.setdefault(k.strip(), []).append(np.nan)
    return {k: np.asarray(v, dtype=float) for k, v in cols.items()}


def _read_matrix_csv(path: Path) -> Tuple[np.ndarray, Optional[List[str]]]:
    """Read a square numeric matrix; tolerate a header/index column of labels."""
    rows: List[List[str]] = []
    with open(path, encoding="utf-8") as f:
        for r in csv.reader(f):
            if r:
                rows.append(r)
    if not rows:
        raise ValueError(f"empty matrix file: {path}")
    # detect a non-numeric header row
    def _is_num(x: str) -> bool:
        try:
            float(x)
            return True
        except ValueError:
            return False
    header = None
    if not all(_is_num(c) for c in rows[0]):
        header = [c.strip() for c in rows[0]]
        rows = rows[1:]
    mat: List[List[float]] = []
    for r in rows:
        # drop a leading label column if present
        cells = r if _is_num(r[0]) else r[1:]
        mat.append([float(c) for c in cells])
    arr = np.asarray(mat, dtype=float)
    return arr, header


# ---------------------------------------------------------------------------
# Error statistics
# ---------------------------------------------------------------------------

def per_feature_errors(
    gt: Dict[str, np.ndarray], est: Dict[str, np.ndarray], eps: float = 0.05,
) -> Dict[str, Dict[str, float]]:
    """bias/sigma/MAE per feature; relative stats for the multiplicative ones."""
    out: Dict[str, Dict[str, float]] = {}
    for feat in DEFAULT_LANE_FEATURE_NAMES:
        if feat not in gt or feat not in est:
            continue
        g, e = gt[feat], est[feat]
        n = min(g.size, e.size)
        g, e = g[:n], e[:n]
        mask = np.isfinite(g) & np.isfinite(e)
        g, e = g[mask], e[mask]
        if g.size == 0:
            continue
        err = e - g
        rec: Dict[str, float] = {
            "bias": float(np.mean(err)),
            "sigma": float(np.std(err, ddof=1)) if g.size > 1 else 0.0,
            "mae": float(np.mean(np.abs(err))),
            "n": int(g.size),
        }
        # relative error on rows with meaningful ground truth (for multiplicative feats)
        rel_mask = g > eps
        if rel_mask.any():
            rel = (e[rel_mask] - g[rel_mask]) / g[rel_mask]
            rec["rel_bias"] = float(np.mean(rel))
            rec["rel_sigma"] = float(np.std(rel, ddof=1)) if rel.size > 1 else 0.0
        out[feat] = rec
    return out


def class_flip_from_confusion(mat: np.ndarray) -> float:
    """Mean over vehicle classes of P(predicted class != true class).

    Assumes mat[i][j] = count of true-class-i predicted-as-j over the model's
    class index space (custom YOLO: 0=accident,1=bus,2=car,3=moto,4=truck).
    Only vehicle classes 1..4 (intersected with the matrix size) are used.
    """
    n = mat.shape[0]
    ids = [c for c in VEHICLE_CLASS_IDS if c < n]
    flips: List[float] = []
    for i in ids:
        row = mat[i]
        total = float(row.sum())
        if total <= 0:
            continue
        correct = float(row[i])
        flips.append(1.0 - correct / total)
    return float(np.mean(flips)) if flips else float("nan")


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate(
    feat_err: Dict[str, Dict[str, float]],
    confusion: Optional[np.ndarray],
    overrides: Dict[str, float],
    speed_units: str,
    speed_cap: float,
) -> Tuple[NoiseConfig, Dict[str, str], int]:
    base = NoiseConfig()  # documented-basis defaults for what we cannot measure
    params: Dict[str, float] = {
        "queue_calib_sigma": base.queue_calib_sigma,
        "speed_sigma_mps": base.speed_sigma_mps,
        "pressure_calib_sigma": base.pressure_calib_sigma,
        "class_flip_rate": base.class_flip_rate,
        "occupancy_bias_sigma": base.occupancy_bias_sigma,
    }
    prov: Dict[str, str] = {k: "default (docs/paper1_tsc/state.md basis)" for k in params}
    n_samples = 0

    def _n(feat: str) -> int:
        return int(feat_err.get(feat, {}).get("n", 0))

    if "effective_queue_norm" in feat_err:
        params["queue_calib_sigma"] = feat_err["effective_queue_norm"].get(
            "rel_sigma", params["queue_calib_sigma"])
        prov["queue_calib_sigma"] = f"measured rel-sigma (n={_n('effective_queue_norm')})"
        params["pressure_calib_sigma"] = params["queue_calib_sigma"]
        prov["pressure_calib_sigma"] = "derived: inherits queue_calib_sigma"
        n_samples = max(n_samples, _n("effective_queue_norm"))

    if "occupancy_norm" in feat_err:
        params["occupancy_bias_sigma"] = feat_err["occupancy_norm"].get(
            "rel_sigma", feat_err["occupancy_norm"]["sigma"])
        prov["occupancy_bias_sigma"] = f"measured rel-sigma (n={_n('occupancy_norm')})"
        n_samples = max(n_samples, _n("occupancy_norm"))

    if "avg_speed_norm" in feat_err:
        sig = feat_err["avg_speed_norm"]["sigma"]
        # CSV speed is normalized [0,1] by default -> convert sigma back to m/s
        params["speed_sigma_mps"] = sig * speed_cap if speed_units == "norm" else sig
        prov["speed_sigma_mps"] = f"measured additive sigma (n={_n('avg_speed_norm')}, {speed_units})"
        n_samples = max(n_samples, _n("avg_speed_norm"))

    share_sigmas = [feat_err[f]["sigma"] for f in
                    ("motorbike_share", "heavy_vehicle_share") if f in feat_err]
    if share_sigmas:
        params["class_flip_rate"] = float(np.mean(share_sigmas))
        prov["class_flip_rate"] = (
            f"measured share-sigma (n={max(_n('motorbike_share'), _n('heavy_vehicle_share'))})")

    if confusion is not None:
        cfr = class_flip_from_confusion(confusion)
        if cfr == cfr:  # not nan
            params["class_flip_rate"] = cfr
            prov["class_flip_rate"] = "measured: YOLO confusion matrix (vehicle classes)"

    for k, v in overrides.items():
        if k in params:
            params[k] = float(v)
            prov[k] = "manual override"

    cfg = NoiseConfig(
        queue_calib_sigma=params["queue_calib_sigma"],
        speed_sigma_mps=params["speed_sigma_mps"],
        pressure_calib_sigma=params["pressure_calib_sigma"],
        class_flip_rate=params["class_flip_rate"],
        occupancy_bias_sigma=params["occupancy_bias_sigma"],
        speed_cap_mps=speed_cap,
        calibrated=True,
    )
    cfg.validate()
    return cfg, prov, n_samples


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_config(cfg: NoiseConfig, prov: Dict[str, str], n_samples: int,
                 feat_err: Dict[str, Dict[str, float]]) -> None:
    CONFIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "calibrated": True,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "n_samples": n_samples,
        "params": {
            "queue_calib_sigma": cfg.queue_calib_sigma,
            "speed_sigma_mps": cfg.speed_sigma_mps,
            "speed_low_regime_mps": cfg.speed_low_regime_mps,
            "speed_high_regime_factor": cfg.speed_high_regime_factor,
            "pressure_calib_sigma": cfg.pressure_calib_sigma,
            "speed_cap_mps": cfg.speed_cap_mps,
            "class_flip_rate": cfg.class_flip_rate,
            "occupancy_bias_sigma": cfg.occupancy_bias_sigma,
            "calibrated": True,
        },
        "provenance": prov,
        "per_feature_error": feat_err,
    }
    CONFIG_OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[calib] wrote {CONFIG_OUT}")

    IIIE_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(IIIE_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["feature", "bias", "sigma", "mae", "rel_bias", "rel_sigma", "n"])
        for feat in DEFAULT_LANE_FEATURE_NAMES:
            r = feat_err.get(feat)
            if not r:
                continue
            w.writerow([feat, f"{r.get('bias', float('nan')):.5f}",
                        f"{r.get('sigma', float('nan')):.5f}",
                        f"{r.get('mae', float('nan')):.5f}",
                        f"{r.get('rel_bias', float('nan')):.5f}",
                        f"{r.get('rel_sigma', float('nan')):.5f}", r.get("n", 0)])
    print(f"[calib] wrote {IIIE_OUT}  (FIG-1a per-feature recovery)")


def parse_overrides(items: Optional[List[str]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--set expects name=value, got {it!r}")
        k, v = it.split("=", 1)
        out[k.strip()] = float(v)
    return out


def main() -> None:
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt-csv", type=str, default=None, help="ground-truth feature CSV")
    p.add_argument("--est-csv", type=str, default=None, help="detector-estimated feature CSV")
    p.add_argument("--confusion-matrix", type=str, default=None,
                   help="square count matrix CSV (true x pred) for class_flip_rate")
    p.add_argument("--speed-units", choices=["norm", "mps"], default="norm",
                   help="units of avg_speed_norm column in the CSVs")
    p.add_argument("--speed-cap", type=float, default=15.0)
    p.add_argument("--set", dest="overrides", action="append", default=None,
                   help="override a NoiseConfig field, e.g. --set class_flip_rate=0.07")
    args = p.parse_args()

    feat_err: Dict[str, Dict[str, float]] = {}
    if args.gt_csv and args.est_csv:
        gt = _read_feature_csv(Path(args.gt_csv))
        est = _read_feature_csv(Path(args.est_csv))
        feat_err = per_feature_errors(gt, est)
        if not feat_err:
            print("[calib] WARNING: no overlapping feature columns measured")

    confusion = None
    if args.confusion_matrix:
        confusion, _ = _read_matrix_csv(Path(args.confusion_matrix))

    overrides = parse_overrides(args.overrides)

    if not (feat_err or confusion is not None or overrides):
        sys.exit("[calib] nothing to calibrate from — provide --gt-csv/--est-csv, "
                 "--confusion-matrix, or --set overrides")

    cfg, prov, n = calibrate(feat_err, confusion, overrides, args.speed_units, args.speed_cap)

    print("\n=== Calibrated sensing-noise parameters ===")
    for k in ("queue_calib_sigma", "occupancy_bias_sigma", "speed_sigma_mps",
              "class_flip_rate", "pressure_calib_sigma"):
        print(f"  {k:<22} = {getattr(cfg, k):.4f}   [{prov.get(k, '')}]")
    print(f"  is_calibrated          = {cfg.is_calibrated}")

    write_config(cfg, prov, n, feat_err)


if __name__ == "__main__":
    main()
