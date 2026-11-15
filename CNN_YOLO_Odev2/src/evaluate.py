#!/usr/bin/env python3
"""Evaluate the trained detector on the VOC2007 test split exactly once.

Outputs:
  results/test_metrics.json
  results/test_metrics_per_class.csv
  results/test_visual_examples.json
  report/figures/test_pred_*.jpg + test_predictions_grid.png
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend (etkileşimsiz arka uç)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "report" / "figures"
CONF_THRESHOLD = 0.25  # Fixed a priori, not tuned on test (test üzerinde ayarlanmadı)
IOU_THRESHOLD = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=str(REPO_ROOT / "weights" / "best.pt"))
    parser.add_argument("--data", default=str(REPO_ROOT / "configs" / "voc_split.yaml"))
    parser.add_argument("--split", default="test", choices=["test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU between one box and many boxes (Tek kutu ile çoklu kutu IoU)."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (box[2] - box[0]) * (box[3] - box[1])
    area_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0)


def load_ground_truth(label_path: Path, width: int, height: int) -> list[dict]:
    """Read normalized YOLO labels and convert to absolute xyxy (Etiketleri oku)."""
    objects = []
    if not label_path.exists():
        return objects
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls_id, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
        objects.append(
            {
                "class_id": cls_id,
                "box": np.array(
                    [
                        (cx - w / 2) * width,
                        (cy - h / 2) * height,
                        (cx + w / 2) * width,
                        (cy + h / 2) * height,
                    ],
                    dtype=np.float32,
                ),
            }
        )
    return objects


def match_image(
    gt_objects: list[dict],
    pred_objects: list[dict],
    iou_threshold: float,
) -> dict:
    """Greedy class-wise matching of predictions to ground truth (Eşleştirme)."""
    tp = fp = fn = 0
    matched_ious = []
    per_class = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "ious": []})

    for cls_id in sorted({o["class_id"] for o in gt_objects} | {o["class_id"] for o in pred_objects}):
        gts = [o for o in gt_objects if o["class_id"] == cls_id]
        preds = sorted(
            [o for o in pred_objects if o["class_id"] == cls_id],
            key=lambda o: o["confidence"],
            reverse=True,
        )
        used = np.zeros(len(gts), dtype=bool)
        for pred in preds:
            if not gts:
                per_class[cls_id]["fp"] += 1
                fp += 1
                continue
            ious = box_iou(pred["box"], np.stack([g["box"] for g in gts]))
            ious_masked = np.where(used, -1.0, ious)
            best_idx = int(np.argmax(ious_masked))
            if ious_masked[best_idx] >= iou_threshold:
                used[best_idx] = True
                per_class[cls_id]["tp"] += 1
                per_class[cls_id]["ious"].append(float(ious_masked[best_idx]))
                matched_ious.append(float(ious_masked[best_idx]))
                tp += 1
            else:
                per_class[cls_id]["fp"] += 1
                fp += 1
        missed = int((~used).sum())
        per_class[cls_id]["fn"] += missed
        fn += missed

    return {"tp": tp, "fp": fp, "fn": fn, "matched_ious": matched_ious, "per_class": dict(per_class)}


def safe_div(num: float, den: float) -> float | None:
    return round(num / den, 6) if den > 0 else None


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    class_names = [model.names[i] for i in range(len(model.names))]

    print(f"Running Ultralytics validation on split='{args.split}' (one-time test use)...", flush=True)
    val = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        conf=0.001,
        iou=0.7,
        device=args.device,
        plots=True,
        project=str(REPO_ROOT / "runs" / "eval"),
        name="test_once",
        exist_ok=True,
        verbose=True,
    )

    # Copy test-set validation figures for the report (Test figürlerini rapora kopyala)
    eval_figure_dir = Path(val.save_dir)
    for src_name, dst_name in {
        "BoxPR_curve.png": "test_pr_curve.png",
        "BoxF1_curve.png": "test_f1_curve.png",
        "confusion_matrix.png": "test_confusion_matrix.png",
        "confusion_matrix_normalized.png": "test_confusion_matrix_normalized.png",
    }.items():
        src = eval_figure_dir / src_name
        if src.exists():
            shutil.copy2(src, FIGURES_DIR / dst_name)

    box_metrics = val.box
    ap_class_index = list(box_metrics.ap_class_index)
    per_class_ap = {
        class_names[idx]: {
            "ap50": round(float(box_metrics.ap50[i]), 6),
            "ap50_95": round(float(box_metrics.ap[i]), 6),
        }
        for i, idx in enumerate(ap_class_index)
    }

    # Manual matching at a fixed confidence threshold for precision / recall / IoU
    # (Sabit güven eşiğinde precision / recall / IoU hesabı)
    test_list = (REPO_ROOT / "data" / "splits" / f"{args.split}.txt").read_text(encoding="utf-8")
    image_paths = [Path(line) for line in test_list.splitlines() if line.strip()]
    print(f"Matching predictions on {len(image_paths)} test images at conf={args.conf} ...", flush=True)

    totals = {"tp": 0, "fp": 0, "fn": 0}
    class_totals = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "ious": []})
    image_records = []
    all_matched_ious = []

    for image_path in image_paths:
        with Image.open(image_path) as img:
            width, height = img.size
        label_path = (
            REPO_ROOT / "data" / "voc_yolo" / "labels" / args.split
            / (image_path.stem + ".txt")
        )
        gt_objects = load_ground_truth(label_path, width, height)

        result = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=0.7,
            device=args.device,
            verbose=False,
        )[0]
        pred_objects = []
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            pred_objects = [
                {"box": xyxy[i], "confidence": float(confs[i]), "class_id": int(classes[i])}
                for i in range(len(xyxy))
            ]

        match = match_image(gt_objects, pred_objects, args.iou)
        totals["tp"] += match["tp"]
        totals["fp"] += match["fp"]
        totals["fn"] += match["fn"]
        all_matched_ious.extend(match["matched_ious"])
        for cls_id, counts in match["per_class"].items():
            for key in ("tp", "fp", "fn"):
                class_totals[cls_id][key] += counts[key]
            class_totals[cls_id]["ious"].extend(counts["ious"])

        n_gt = len(gt_objects)
        n_pred = len(pred_objects)
        if n_gt + n_pred > 0:
            img_tp, img_fp, img_fn = match["tp"], match["fp"], match["fn"]
            img_p = safe_div(img_tp, img_tp + img_fp)
            img_r = safe_div(img_tp, img_tp + img_fn)
            img_f1 = None
            if img_p is not None and img_r is not None and img_p + img_r > 0:
                img_f1 = 2 * img_p * img_r / (img_p + img_r)
            image_records.append(
                {
                    "path": str(image_path.relative_to(REPO_ROOT)),
                    "gt": n_gt,
                    "pred": n_pred,
                    "tp": img_tp,
                    "fp": img_fp,
                    "fn": img_fn,
                    "precision": img_p,
                    "recall": img_r,
                    "f1": img_f1,
                }
            )

    precision = safe_div(totals["tp"], totals["tp"] + totals["fp"])
    recall = safe_div(totals["tp"], totals["tp"] + totals["fn"])
    f1 = None
    if precision is not None and recall is not None and precision + recall > 0:
        f1 = round(2 * precision * recall / (precision + recall), 6)

    per_class = {}
    macro_p, macro_r, macro_f1 = [], [], []
    csv_rows = []
    for idx, name in enumerate(class_names):
        counts = class_totals.get(idx, {"tp": 0, "fp": 0, "fn": 0, "ious": []})
        p = safe_div(counts["tp"], counts["tp"] + counts["fp"])
        r = safe_div(counts["tp"], counts["tp"] + counts["fn"])
        c_f1 = None
        if p is not None and r is not None and p + r > 0:
            c_f1 = round(2 * p * r / (p + r), 6)
        if p is not None:
            macro_p.append(p)
        if r is not None:
            macro_r.append(r)
        if c_f1 is not None:
            macro_f1.append(c_f1)
        mean_iou = round(float(np.mean(counts["ious"])), 6) if counts["ious"] else None
        ap = per_class_ap.get(name, {"ap50": 0.0, "ap50_95": 0.0})
        per_class[name] = {
            "tp": counts["tp"],
            "fp": counts["fp"],
            "fn": counts["fn"],
            "precision": p,
            "recall": r,
            "f1": c_f1,
            "mean_matched_iou": mean_iou,
            "ap50": ap["ap50"],
            "ap50_95": ap["ap50_95"],
        }
        csv_rows.append(
            {
                "class": name,
                "tp": counts["tp"],
                "fp": counts["fp"],
                "fn": counts["fn"],
                "precision": p,
                "recall": r,
                "f1": c_f1,
                "mean_matched_iou": mean_iou,
                "ap50": ap["ap50"],
                "ap50_95": ap["ap50_95"],
            }
        )

    pd.DataFrame(csv_rows).to_csv(RESULTS_DIR / "test_metrics_per_class.csv", index=False)

    ious_array = np.array(all_matched_ious) if all_matched_ious else np.array([])
    payload = {
        "split": args.split,
        "weights": str(Path(args.weights).relative_to(REPO_ROOT)),
        "images": len(image_paths),
        "imgsz": args.imgsz,
        "ultralytics_metrics": {
            "mAP50": round(float(box_metrics.map50), 6),
            "mAP50_95": round(float(box_metrics.map), 6),
            "precision_best_f1": round(float(box_metrics.mp), 6),
            "recall_best_f1": round(float(box_metrics.mr), 6),
            "note": "Ultralytics val() metrics at conf=0.001; precision/recall at best-F1 threshold.",
        },
        "fixed_threshold_metrics": {
            "confidence_threshold": args.conf,
            "iou_threshold": args.iou,
            "overall": {
                "tp": totals["tp"],
                "fp": totals["fp"],
                "fn": totals["fn"],
                "precision": precision,
                "recall": recall,
                "f1": f1,
            },
            "macro_over_classes_with_predictions": {
                "precision": round(float(np.mean(macro_p)), 6) if macro_p else None,
                "recall": round(float(np.mean(macro_r)), 6) if macro_r else None,
                "f1": round(float(np.mean(macro_f1)), 6) if macro_f1 else None,
            },
            "matched_boxes": len(all_matched_ious),
            "mean_matched_iou": round(float(ious_array.mean()), 6) if len(ious_array) else None,
            "median_matched_iou": round(float(np.median(ious_array)), 6) if len(ious_array) else None,
            "std_matched_iou": round(float(ious_array.std()), 6) if len(ious_array) else None,
            "min_matched_iou": round(float(ious_array.min()), 6) if len(ious_array) else None,
            "max_matched_iou": round(float(ious_array.max()), 6) if len(ious_array) else None,
        },
        "per_class": per_class,
        "evaluation_wall_clock_seconds": round(time.perf_counter() - start, 2),
    }
    (RESULTS_DIR / "test_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Visual examples: 4 best and 4 worst images by per-image F1 (İyi/kötü örnekler)
    with_gt = [rec for rec in image_records if rec["gt"] > 0 and rec["f1"] is not None]
    good = sorted(with_gt, key=lambda r: (-r["f1"], -r["tp"]))[:4]
    bad = sorted(with_gt, key=lambda r: (r["f1"], -r["fn"]))[:4]
    selected = good + bad
    # Ensure distinct images (Tekrarları kaldır)
    seen = set()
    selected = [rec for rec in selected if not (rec["path"] in seen or seen.add(rec["path"]))]

    saved = []
    for rank, rec in enumerate(selected, start=1):
        image_path = REPO_ROOT / rec["path"]
        result = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=0.7,
            device=args.device,
            verbose=False,
        )[0]
        plotted = result.plot(line_width=2, font_size=10, labels=True, conf=True)
        out = FIGURES_DIR / f"test_pred_{rank:02d}_f1_{rec['f1']:.3f}.jpg"
        cv2.imwrite(str(out), plotted)
        saved.append({**rec, "figure": str(out.relative_to(REPO_ROOT))})

    if saved:
        cols = 2
        rows = (len(saved) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(14, 5.2 * rows))
        axes = np.atleast_1d(axes).ravel()
        for ax, rec in zip(axes, saved):
            img = cv2.cvtColor(cv2.imread(str(REPO_ROOT / rec["figure"])), cv2.COLOR_BGR2RGB)
            ax.imshow(img)
            ax.set_title(
                f"F1={rec['f1']:.2f} | TP={rec['tp']} FP={rec['fp']} FN={rec['fn']}",
                fontsize=10,
            )
            ax.axis("off")
        for ax in axes[len(saved):]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(
            FIGURES_DIR / "test_predictions_grid.jpg",
            dpi=160,
            pil_kwargs={"quality": 90, "optimize": True},
        )
        plt.close(fig)

    (RESULTS_DIR / "test_visual_examples.json").write_text(
        json.dumps(saved, indent=2), encoding="utf-8"
    )

    print(json.dumps(payload["ultralytics_metrics"], indent=2), flush=True)
    print(json.dumps(payload["fixed_threshold_metrics"], indent=2), flush=True)
    print(f"Saved {RESULTS_DIR / 'test_metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
