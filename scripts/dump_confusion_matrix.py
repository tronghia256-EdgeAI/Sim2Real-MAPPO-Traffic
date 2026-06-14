from __future__ import annotations

"""
dump_confusion_matrix.py  (III-E, path 1)
=========================================
Run `yolo val` on YOUR trained detector and dump its confusion matrix to a CSV
that experiment/runners/calibrate_noise.py reads directly for `class_flip_rate`.

No new annotation needed — this uses the validation split you already trained on.

Orientation note (important): ultralytics stores the matrix as [predicted, true].
calibrate_noise expects [true, predicted] (so a row sum is "all instances of a
true class" → 1 − diagonal/row = misclassification / class-flip rate). This
script TRANSPOSES by default to produce the [true, predicted] convention.

Class index order = your model's class order. NOTES.md §4.1 assumes
0=accident, 1=bus, 2=car, 3=motorcycle, 4=truck; calibrate_noise averages the
flip rate over the VEHICLE classes 1..4 only (accident + background ignored).
The printed class names let you verify the order matches.

Usage (from project root):
    python scripts/dump_confusion_matrix.py \
        --model runs/detect/train/weights/best.pt \
        --data  path/to/data.yaml \
        --out   confusion_matrix.csv
    # then:
    python experiment/runners/calibrate_noise.py \
        --confusion-matrix confusion_matrix.csv \
        --set occupancy_bias_sigma=0.08        # path 3: assumed/cited (see docs)
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="trained detector weights (.pt)")
    p.add_argument("--data", required=True, help="dataset YAML used for validation")
    p.add_argument("--split", default="val", choices=["val", "test", "train"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--out", default="confusion_matrix.csv")
    p.add_argument("--no-transpose", action="store_true",
                   help="keep ultralytics [pred,true] orientation (default transposes "
                        "to [true,pred] for calibrate_noise)")
    args = p.parse_args()

    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover
        sys.exit(f"[cm] ultralytics not importable ({exc}); pip install ultralytics")

    print(f"[cm] validating {args.model} on {args.data} (split={args.split})")
    model = YOLO(args.model)
    metrics = model.val(data=args.data, split=args.split,
                        imgsz=args.imgsz, conf=args.conf, iou=args.iou)

    cm = getattr(metrics, "confusion_matrix", None)
    mat = getattr(cm, "matrix", None) if cm is not None else None
    if mat is None:
        sys.exit("[cm] no confusion_matrix on the val metrics — check ultralytics version")
    mat = np.asarray(mat, dtype=float)            # ultralytics: rows=pred, cols=true
    out_mat = mat if args.no_transpose else mat.T  # -> rows=true, cols=pred

    # class names (model order); background is the trailing row/col
    names = getattr(model, "names", None) or {}
    ordered = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
    print(f"[cm] class order (index 0..): {ordered}")
    print(f"[cm] matrix shape {out_mat.shape} "
          f"({'true x pred' if not args.no_transpose else 'pred x true'}); "
          f"last index = background")

    out_path = Path(args.out)
    np.savetxt(out_path, out_mat, delimiter=",", fmt="%g")
    print(f"[cm] wrote {out_path}")
    print("[cm] next: python experiment/runners/calibrate_noise.py "
          f"--confusion-matrix {out_path}")


if __name__ == "__main__":
    main()
