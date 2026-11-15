#!/usr/bin/env python3
"""Fill measured numbers from results/*.json into REPORT.md and NOTES_TR.md.

Placeholders use the {{NAME}} syntax. Real photo comments are read from
results/real_photos_comments_tr.md when available.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
REPORT = REPO_ROOT / "report" / "REPORT.md"
NOTES = REPO_ROOT / "NOTES_TR.md"


def load(name: str) -> dict | None:
    path = RESULTS / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value, digits: int = 3, suffix: str = "") -> str:
    if value is None:
        return "yok"
    if isinstance(value, str):
        return value + suffix
    return f"{value:.{digits}f}{suffix}"


def build_class_table(metrics: dict) -> str:
    rows = []
    per_class = metrics["per_class"]
    for name, item in per_class.items():
        rows.append(
            "| {name} | {ap50} | {ap} | {p} | {r} | {f1} | {iou} | {tp} | {fp} | {fn} |".format(
                name=name,
                ap50=fmt(item["ap50"]),
                ap=fmt(item["ap50_95"]),
                p=fmt(item["precision"]),
                r=fmt(item["recall"]),
                f1=fmt(item["f1"]),
                iou=fmt(item["mean_matched_iou"]),
                tp=item["tp"],
                fp=item["fp"],
                fn=item["fn"],
            )
        )
    rows.sort(key=lambda row: -float(row.split("|")[2].strip()))
    header = (
        "| Sınıf | AP50 | AP50-95 | P | R | F1 | IoU | TP | FP | FN |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    return header + "\n" + "\n".join(rows)


def class_examples(metrics: dict) -> str:
    items = [
        (name, item["ap50"])
        for name, item in metrics["per_class"].items()
        if item["ap50_95"] is not None
    ]
    items.sort(key=lambda pair: -pair[1])
    best = ", ".join(f"{name} (AP@0.5 {ap:.3f})" for name, ap in items[:3])
    worst = ", ".join(f"{name} (AP@0.5 {ap:.3f})" for name, ap in items[-3:])
    return f"En iyi sınıflar: {best}. En zayıf sınıflar: {worst}."


def build_photo_class_table(item: dict) -> str:
    ground_truth = item.get("ground_truth", {})
    predicted = item.get("predicted", {})
    confidence = item.get("mean_confidence", {})
    classes = sorted(set(ground_truth) | set(predicted))
    if not classes:
        return ""
    lines = [
        "| Sınıf | Tahmin | Gerçek | Doğru | Kaçırılan | Yanlış | Ort. güven |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cls in classes:
        pred_count = predicted.get(cls, 0)
        gt_count = ground_truth.get(cls, 0)
        lines.append(
            f"| {cls} | {pred_count} | {gt_count} | {min(pred_count, gt_count)} | "
            f"{max(gt_count - pred_count, 0)} | {max(pred_count - gt_count, 0)} | "
            f"{fmt(confidence.get(cls))} |"
        )
    return "\n".join(lines)


def build_real_photos_section(summary: dict | None) -> str:
    if not summary or not summary.get("photos"):
        return (
            "> Gerçek fotoğraflar henüz eklenmedi. `real_photos/ground_truth.csv` doldurulup "
            "`src/real_photos.py` çalıştırıldıktan sonra bu bölüm gerçek sonuçlarla güncellenecektir."
        )
    comments_path = RESULTS / "real_photos_comments_tr.md"
    comments = comments_path.read_text(encoding="utf-8") if comments_path.exists() else ""
    lines = [
        "Aşağıdaki tablo, telefonla çekilen 5 fotoğraf üzerinde modelin fotoğraf bazında "
        "tahmin sayılarını gerçek sayılarla karşılaştırır. `doğru` = min(tahmin, gerçek); "
        "`kaçırılan` = max(gerçek - tahmin, 0); `yanlış` = max(tahmin - gerçek, 0). "
        f"Confidence eşiği {summary.get('confidence_threshold')}; sınıf bazında ayrıntılar "
        "her fotoğrafın altındadır. Karşılaştırma kutu eşleştirmesi değil, sınıf başına "
        "sayı karşılaştırmasıdır.",
        "",
        "| Fotoğraf | Doğru | Kaçırılan | Yanlış | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary["photos"]:
        lines.append(
            f"| {item['filename']} | {item['correct']} | {item['missed']} | "
            f"{item['false_positive']} | {fmt(item['count_precision'])} | "
            f"{fmt(item['count_recall'])} | {fmt(item['count_f1'])} |"
        )
    lines += ["", "### Fotoğraflar ve tahminler", ""]
    for item in summary["photos"]:
        figure = Path(item["figure"]).name
        lines.append(f"**{item['filename']}** — Not: {item['notes']}")
        lines.append("")
        lines.append(f"![{item['filename']} tahminleri](figures/{figure})")
        lines.append("")
        class_table = build_photo_class_table(item)
        if class_table:
            lines += [class_table, ""]
        inference_note = (
            " (ilk çalıştırma; model warmup dahil, ortalama hız için benchmark geçerlidir)"
            if item["filename"] == "foto1.jpg"
            else ""
        )
        lines += [f"Tek görüntü çıkarım süresi: {fmt(item.get('inference_ms'), 2)} ms{inference_note}.",
                  ""]
    if comments:
        lines += [comments, ""]
    else:
        lines += [
            "> Yorumlar taslak halindedir; teslim öncesi kontrol edilecektir.",
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    metrics = load("test_metrics.json")
    train_time = load("train_time.json")
    smoke_time = load("smoke_time.json")
    inference = load("inference_time.json")
    real_summary = load("real_photos_summary.json")

    if not metrics:
        raise FileNotFoundError("results/test_metrics.json not found; run src/evaluate.py first.")
    if not train_time:
        raise FileNotFoundError("results/train_time.json not found; finish training first.")

    fixed = metrics["fixed_threshold_metrics"]
    overall = fixed["overall"]
    ultra = metrics["ultralytics_metrics"]
    gpu = inference["results"]["gpu"] if inference else {}
    cpu = inference["results"]["cpu"] if inference else {}

    values = {
        "MAP50": fmt(ultra["mAP50"]),
        "MAP50_95": fmt(ultra["mAP50_95"]),
        "PRECISION": fmt(overall["precision"]),
        "RECALL": fmt(overall["recall"]),
        "F1": fmt(overall["f1"]),
        "TP": str(overall["tp"]),
        "FP": str(overall["fp"]),
        "FN": str(overall["fn"]),
        "MATCHED_BOXES": str(fixed["matched_boxes"]),
        "MEAN_IOU": fmt(fixed["mean_matched_iou"]),
        "MEDIAN_IOU": fmt(fixed["median_matched_iou"]),
        "ULTA_P": fmt(ultra["precision_best_f1"]),
        "ULTA_R": fmt(ultra["recall_best_f1"]),
        "CLASS_TABLE": build_class_table(metrics),
        "SMOKE_TIME": fmt(smoke_time["wall_clock_seconds"], 1) if smoke_time else "yok",
        "TRAIN_TIME": train_time["wall_clock_human"],
        "EPOCHS": str(train_time["epochs_completed"]),
        "EPOCH_TIME": fmt(train_time["wall_clock_seconds"] / 60.0 / max(train_time["epochs_completed"], 1), 2),
        "GPU_PRE": fmt(gpu.get("preprocess", {}).get("mean_ms"), 2),
        "GPU_INF": fmt(gpu.get("inference", {}).get("mean_ms"), 2),
        "GPU_POST": fmt(gpu.get("postprocess", {}).get("mean_ms"), 2),
        "GPU_TOTAL": fmt(gpu.get("end_to_end", {}).get("mean_ms"), 2),
        "GPU_FPS": fmt(gpu.get("fps_from_mean_end_to_end"), 1),
        "CPU_PRE": fmt(cpu.get("preprocess", {}).get("mean_ms"), 2),
        "CPU_INF": fmt(cpu.get("inference", {}).get("mean_ms"), 2),
        "CPU_POST": fmt(cpu.get("postprocess", {}).get("mean_ms"), 2),
        "CPU_TOTAL": fmt(cpu.get("end_to_end", {}).get("mean_ms"), 2),
        "CPU_FPS": fmt(cpu.get("fps_from_mean_end_to_end"), 1),
        "CLASS_EXAMPLES": class_examples(metrics),
        "REAL_PHOTOS": "Gerçek fotoğraf sonuçları için `results/real_photos_summary.json` dosyasına bakın.",
        "REAL_PHOTOS_REPORT": build_real_photos_section(real_summary),
        "EARLY_STOPPING": (
            "Hayır; 48 epoch tamamlandı ve erken durdurma tetiklenmedi."
            if train_time["epochs_completed"] >= 48
            else f"Evet; eğitim {train_time['epochs_completed']} epoch'ta erken durdu."
        ),
    }

    report_text = REPORT.read_text(encoding="utf-8")
    for key, value in values.items():
        report_text = report_text.replace("{{" + key + "}}", value)
    REPORT.write_text(report_text, encoding="utf-8")

    notes_text = NOTES.read_text(encoding="utf-8")
    for key, value in values.items():
        notes_text = notes_text.replace("{{" + key + "}}", value)
    NOTES.write_text(notes_text, encoding="utf-8")

    print("Filled placeholders in REPORT.md and NOTES_TR.md")
    leftovers = [token for token in ("{{", "}}") if token in report_text or token in notes_text]
    if leftovers:
        print("WARNING: placeholders remain:", leftovers)


if __name__ == "__main__":
    main()
