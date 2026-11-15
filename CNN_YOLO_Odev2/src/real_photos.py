#!/usr/bin/env python3
"""Run the detector on the 5 real phone photos and compare counts with ground truth.

Input : real_photos/ground_truth.csv  (filename,ground_truth,notes)
        ground_truth format: "car:3;person:2"
Output: report/figures/real_photo_*.jpg
        results/real_photos_comparison.csv
        results/real_photos_summary.json
        results/real_photos_draft.md
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import pandas as pd
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

REPO_ROOT = Path(__file__).resolve().parents[1]
PHOTO_DIR = REPO_ROOT / "real_photos"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "report" / "figures"
CSV_PATH = PHOTO_DIR / "ground_truth.csv"

register_heif_opener()  # HEIC support (HEIC desteği)

from ultralytics import YOLO  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=str(REPO_ROOT / "weights" / "best.pt"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def parse_ground_truth(text: str) -> dict[str, int]:
    """Parse 'car:3;person:2' into a dict (Ground truth metnini ayrıştır)."""
    counts: dict[str, int] = {}
    if not text or pd.isna(text):
        return counts
    for part in str(text).split(";"):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid ground_truth entry: {part!r}")
        name, value = part.split(":", 1)
        name = name.strip()
        counts[name] = counts.get(name, 0) + int(value.strip())
    return counts


def main() -> None:
    args = parse_args()
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing {CSV_PATH}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH).fillna("")
    if len(df) == 0:
        raise RuntimeError("ground_truth.csv is empty.")

    model = YOLO(args.weights)
    valid_names = set(model.names.values())

    rows = []
    summary = []
    missing_photos = []

    for _, row in df.iterrows():
        filename = str(row["filename"]).strip()
        if not filename:
            continue
        photo_path = PHOTO_DIR / filename
        if not photo_path.exists():
            missing_photos.append(filename)
            continue

        gt_counts = parse_ground_truth(str(row["ground_truth"]))
        unknown = [name for name in gt_counts if name not in valid_names]
        if unknown:
            raise ValueError(f"Unknown class names in ground truth: {unknown}")

        with Image.open(photo_path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")

        t0 = time.perf_counter()
        result = model.predict(source=img, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        wall_ms = (time.perf_counter() - t0) * 1000.0

        pred_counts: dict[str, int] = {}
        confidences: dict[str, list[float]] = {}
        if result.boxes is not None and len(result.boxes) > 0:
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
            for cls_id, conf in zip(classes, confs):
                name = model.names[int(cls_id)]
                pred_counts[name] = pred_counts.get(name, 0) + 1
                confidences.setdefault(name, []).append(float(conf))

        annotated = cv2.cvtColor(result.plot(line_width=2, font_size=11), cv2.COLOR_BGR2RGB)
        out_figure = FIGURES_DIR / f"real_photo_{photo_path.stem}.jpg"
        Image.fromarray(annotated).save(out_figure, quality=92)

        all_classes = sorted(set(gt_counts) | set(pred_counts))
        total_tp = total_fp = total_fn = 0
        for name in all_classes:
            gt_n = int(gt_counts.get(name, 0))
            pred_n = int(pred_counts.get(name, 0))
            correct = min(gt_n, pred_n)
            missed = max(gt_n - pred_n, 0)
            false_pos = max(pred_n - gt_n, 0)
            total_tp += correct
            total_fp += false_pos
            total_fn += missed
            rows.append(
                {
                    "filename": filename,
                    "class": name,
                    "predicted": pred_n,
                    "actual": gt_n,
                    "correct": correct,
                    "missed": missed,
                    "false_positive": false_pos,
                    "mean_confidence": round(float(sum(confidences.get(name, [])) / pred_n), 4) if pred_n else None,
                }
            )

        precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else None
        recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall > 0
            else None
        )
        summary.append(
            {
                "filename": filename,
                "figure": str(out_figure.relative_to(REPO_ROOT)),
                "notes": str(row["notes"]),
                "ground_truth": gt_counts,
                "predicted": pred_counts,
                "correct": total_tp,
                "missed": total_fn,
                "false_positive": total_fp,
                "count_precision": round(precision, 4) if precision is not None else None,
                "count_recall": round(recall, 4) if recall is not None else None,
                "count_f1": round(f1, 4) if f1 is not None else None,
                "inference_ms": round(wall_ms, 2),
                "mean_confidence": {
                    name: round(sum(vals) / len(vals), 4) for name, vals in confidences.items()
                },
            }
        )

    if missing_photos:
        print(f"WARNING: missing photos: {missing_photos}")

    if not summary:
        raise RuntimeError("No matching photos found. Put photos into real_photos/ and fill the CSV.")

    comparison = pd.DataFrame(rows)
    comparison.to_csv(RESULTS_DIR / "real_photos_comparison.csv", index=False)
    (RESULTS_DIR / "real_photos_summary.json").write_text(
        json.dumps(
            {
                "confidence_threshold": args.conf,
                "imgsz": args.imgsz,
                "photos": summary,
                "missing_photos": missing_photos,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    md_lines = [
        "# Gerçek Fotoğraf Testi - Otomatik Taslak",
        "",
        "> Bu dosya `src/real_photos.py` tarafından üretilmiştir. Yorumlar taslaktır; "
        "teslimden önce kontrol edilmelidir.",
        "",
        f"Confidence threshold: {args.conf}",
        "",
        "| Fotoğraf | Doğru | Kaçırılan | Yanlış tespit | Precision | Recall | F1 |",
        "|---|---|---|---|---|---|---|",
    ]
    for item in summary:
        md_lines.append(
            f"| {item['filename']} | {item['correct']} | {item['missed']} | "
            f"{item['false_positive']} | {item['count_precision']} | "
            f"{item['count_recall']} | {item['count_f1']} |"
        )
    md_lines += ["", "## Fotoğraf bazında sınıf karşılaştırması", ""]
    for item in summary:
        md_lines.append(f"### {item['filename']}")
        md_lines.append("")
        md_lines.append(f"Not: {item['notes']}")
        md_lines.append("")
        md_lines.append("| Sınıf | Tahmin | Gerçek | Doğru | Kaçırılan | Yanlış |")
        md_lines.append("|---|---|---|---|---|---|")
        for row in rows:
            if row["filename"] == item["filename"]:
                md_lines.append(
                    f"| {row['class']} | {row['predicted']} | {row['actual']} | "
                    f"{row['correct']} | {row['missed']} | {row['false_positive']} |"
                )
        md_lines += ["", "Yorum (taslak): _TODO_", ""]
    (RESULTS_DIR / "real_photos_draft.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(json.dumps({item["filename"]: {
        "correct": item["correct"], "missed": item["missed"],
        "false_positive": item["false_positive"], "f1": item["count_f1"],
    } for item in summary}, indent=2))
    print(f"Saved {RESULTS_DIR / 'real_photos_comparison.csv'}")
    print(f"Saved {RESULTS_DIR / 'real_photos_draft.md'}")


if __name__ == "__main__":
    main()
