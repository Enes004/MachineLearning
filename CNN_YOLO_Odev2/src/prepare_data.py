#!/usr/bin/env python3
"""Prepare Pascal VOC 2007 + 2012 in YOLO format with a fixed split.

Builds:
  data/voc_yolo/images/{train,val,test}/*.jpg  (hard links)
  data/voc_yolo/labels/{train,val,test}/*.txt
  data/splits/{train,val,test}.txt
  configs/voc_split.yaml
  results/dataset_stats.json
  report/figures/dataset_class_distribution.png

Split rule (fixed by task):
  VOC2007 trainval + VOC2012 trainval -> 90% train / 10% val, seed=42
  VOC2007 test -> test (untouched until final evaluation)
Duplicate images across the two years are removed to avoid train/val leakage.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend (etkileşimsiz arka uç)
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_ROOT = REPO_ROOT / "data" / "voc_yolo"
SPLIT_DIR = REPO_ROOT / "data" / "splits"
CONFIG_DIR = REPO_ROOT / "configs"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "report" / "figures"

SEED = 42
TRAIN_RATIO = 0.90

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(VOC_CLASSES)}

ZIPS = {
    "2007_trainval": "VOCtrainval_06-Nov-2007.zip",
    "2007_test": "VOCtest_06-Nov-2007.zip",
    "2012_trainval": "VOCtrainval_11-May-2012.zip",
}


def extract_archives() -> None:
    """Extract the VOC archives if needed (Arşivleri gerekirse aç)."""
    vocdevkit = DATA_RAW / "VOCdevkit"
    for key, zip_name in ZIPS.items():
        src = DATA_RAW / zip_name
        if not src.exists():
            raise FileNotFoundError(f"Missing archive: {src}")
        marker = vocdevkit / (".extracted_" + key)
        if marker.exists():
            continue
        print(f"Extracting {zip_name} ...", flush=True)
        with zipfile.ZipFile(src) as zf:
            zf.extractall(DATA_RAW)
        marker.touch()
    if not vocdevkit.exists():
        raise FileNotFoundError(f"VOCdevkit not found after extraction: {vocdevkit}")


def sha1sum(path: Path, chunk: int = 1 << 20) -> str:
    """Hash image bytes to detect duplicates across years (Yıllar arası kopyaları bul)."""
    h = hashlib.sha1()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def read_image_ids(split_file: Path) -> list[str]:
    """Read VOC ImageSets/Main image ids (VOC görüntü kimliklerini oku)."""
    with split_file.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def parse_voc_xml(xml_path: Path) -> tuple[list[dict], int]:
    """Parse one VOC annotation XML into boxes (Bir VOC XML'ini kutulara çevir)."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = int(float(size.findtext("width")))
    height = int(float(size.findtext("height")))
    objects = []
    difficult_count = 0
    for obj in root.findall("object"):
        name = obj.findtext("name").strip()
        if name not in CLASS_TO_IDX:
            continue
        difficult = int(float(obj.findtext("difficult") or 0))
        difficult_count += difficult
        box = obj.find("bndbox")
        xmin = float(box.findtext("xmin"))
        ymin = float(box.findtext("ymin"))
        xmax = float(box.findtext("xmax"))
        ymax = float(box.findtext("ymax"))
        # Clamp boxes to image bounds (Kutuları görüntü sınırlarına kırp)
        xmin = max(0.0, min(xmin, width - 1.0))
        ymin = max(0.0, min(ymin, height - 1.0))
        xmax = max(0.0, min(xmax, width - 1.0))
        ymax = max(0.0, min(ymax, height - 1.0))
        if xmax - xmin < 1.0 or ymax - ymin < 1.0:
            continue
        objects.append(
            {
                "class_id": CLASS_TO_IDX[name],
                "cx": (xmin + xmax) / 2.0 / width,
                "cy": (ymin + ymax) / 2.0 / height,
                "w": (xmax - xmin) / width,
                "h": (ymax - ymin) / height,
            }
        )
    return objects, difficult_count


def collect_year(year: str, split_name: str) -> list[dict]:
    """Collect image/annotation pairs for one VOC year split (Bir yılın kayıtlarını topla)."""
    year_dir = DATA_RAW / "VOCdevkit" / f"VOC{year}"
    ids = read_image_ids(year_dir / "ImageSets" / "Main" / f"{split_name}.txt")
    records = []
    missing = 0
    for image_id in ids:
        image_path = year_dir / "JPEGImages" / f"{image_id}.jpg"
        xml_path = year_dir / "Annotations" / f"{image_id}.xml"
        if not image_path.exists() or not xml_path.exists():
            missing += 1
            continue
        records.append(
            {
                "source_year": year,
                "image_id": image_id,
                "image_path": image_path,
                "xml_path": xml_path,
            }
        )
    print(f"VOC{year} {split_name}: {len(records)} usable, {missing} missing", flush=True)
    return records


def deduplicate(records: list[dict]) -> tuple[list[dict], int]:
    """Drop byte-identical images keeping VOC2012 annotation (Kopyaları tekilleştir)."""
    # Prefer VOC2012 records because their annotations are newer (2012 öncelikli)
    ordered = sorted(records, key=lambda r: (r["source_year"] != "2012", r["image_id"]))
    seen = set()
    unique = []
    duplicates = 0
    for rec in ordered:
        digest = sha1sum(rec["image_path"])
        if digest in seen:
            duplicates += 1
            continue
        seen.add(digest)
        rec["sha1"] = digest
        unique.append(rec)
    return unique, duplicates


def yolo_label_text(record: dict) -> tuple[str, int, int]:
    """Return YOLO label text and counts for one record (YOLO etiket metni üret)."""
    objects, difficult_count = parse_voc_xml(record["xml_path"])
    lines = []
    for obj in objects:
        lines.append(
            f"{obj['class_id']} {obj['cx']:.6f} {obj['cy']:.6f} "
            f"{obj['w']:.6f} {obj['h']:.6f}"
        )
    return "\n".join(lines) + ("\n" if lines else ""), len(objects), difficult_count


def materialize(records: list[dict], split: str) -> list[str]:
    """Write hard-linked images and YOLO labels for a split (Split dosyalarını yaz)."""
    image_dir = DATA_ROOT / "images" / split
    label_dir = DATA_ROOT / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    image_paths = []
    for rec in records:
        out_name = f"voc{rec['source_year']}_{rec['image_id']}.jpg"
        image_out = image_dir / out_name
        label_out = label_dir / (out_name.replace(".jpg", ".txt"))
        if not image_out.exists():
            try:
                os.link(rec["image_path"], image_out)  # Hard link saves disk (disk tasarrufu)
            except OSError:
                shutil.copy2(rec["image_path"], image_out)
        text, _, _ = yolo_label_text(rec)
        label_out.write_text(text, encoding="utf-8")
        image_paths.append(str(image_out.resolve()))
    return image_paths


def count_split(records: list[dict], image_paths: list[str]) -> dict:
    """Compute per-split image/object/class statistics (Split istatistikleri)."""
    class_counts = Counter()
    total_objects = 0
    empty_images = 0
    difficult_total = 0
    for path, rec in zip(image_paths, records):
        text, n_obj, n_difficult = yolo_label_text(rec)
        total_objects += n_obj
        difficult_total += n_difficult
        if n_obj == 0:
            empty_images += 1
        for line in text.splitlines():
            class_counts[int(line.split()[0])] += 1
    return {
        "images": len(records),
        "objects": total_objects,
        "objects_per_image": round(total_objects / max(len(records), 1), 4),
        "empty_images": empty_images,
        "difficult_objects": difficult_total,
        "class_counts": {name: class_counts.get(idx, 0) for idx, name in enumerate(VOC_CLASSES)},
    }


def write_split_config() -> None:
    """Write the Ultralytics dataset yaml (Ultralytics veri yaml'ını yaz)."""
    lines = [
        "# Pascal VOC 2007+2012, custom train/val/test split (seed=42)",
        f"path: {DATA_ROOT.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "names:",
    ]
    lines += [f"  {idx}: {name}" for idx, name in enumerate(VOC_CLASSES)]
    (CONFIG_DIR / "voc_split.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_class_distribution(stats: dict) -> None:
    """Draw a grouped bar chart of class counts (Sınıf dağılımı grafiği)."""
    import numpy as np

    splits = ["train", "val", "test"]
    x = np.arange(len(VOC_CLASSES))
    width = 0.27
    plt.figure(figsize=(14, 5.5))
    colors = {"train": "#1f77b4", "val": "#ff7f0e", "test": "#2ca02c"}
    for offset, split in zip((-width, 0.0, width), splits):
        counts = [stats[split]["class_counts"][name] for name in VOC_CLASSES]
        plt.bar(x + offset, counts, width, label=split, color=colors[split])
    plt.xticks(x, VOC_CLASSES, rotation=45, ha="right", fontsize=9)
    plt.ylabel("Object count")
    plt.title("Pascal VOC class distribution per split")
    plt.legend()
    plt.tight_layout()
    out = FIGURES_DIR / "dataset_class_distribution.png"
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"Saved {out}", flush=True)


def main() -> None:
    random.seed(SEED)
    for directory in (DATA_ROOT, SPLIT_DIR, CONFIG_DIR, RESULTS_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    extract_archives()

    records_2007 = collect_year("2007", "trainval")
    records_2012 = collect_year("2012", "trainval")
    records_test = collect_year("2007", "test")
    test_ids = {rec["image_id"] for rec in records_test}

    all_trainval, duplicates = deduplicate(records_2007 + records_2012)

    # Never let a test image enter train/val (Test görüntüleri eğitime girmesin)
    overlap_with_test = [
        rec for rec in all_trainval
        if rec["source_year"] == "2007" and rec["image_id"] in test_ids
    ]
    if overlap_with_test:
        raise RuntimeError("VOC2007 test images found in trainval records; aborting.")

    rng = random.Random(SEED)
    rng.shuffle(all_trainval)
    n_train = int(len(all_trainval) * TRAIN_RATIO)
    train_records = sorted(all_trainval[:n_train], key=lambda r: r["image_id"])
    val_records = sorted(all_trainval[n_train:], key=lambda r: r["image_id"])
    test_records = sorted(records_test, key=lambda r: r["image_id"])

    # Remove any stale files from a previous run (Eski çıktıları temizle)
    for split in ("train", "val", "test"):
        for sub in ("images", "labels"):
            target = DATA_ROOT / sub / split
            if target.exists():
                shutil.rmtree(target)

    split_paths = {
        "train": materialize(train_records, "train"),
        "val": materialize(val_records, "val"),
        "test": materialize(test_records, "test"),
    }
    split_records = {"train": train_records, "val": val_records, "test": test_records}

    for split, paths in split_paths.items():
        (SPLIT_DIR / f"{split}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")

    write_split_config()

    stats = {
        "seed": SEED,
        "train_ratio": TRAIN_RATIO,
        "source": {
            "train_pool": "VOC2007 trainval + VOC2012 trainval",
            "test": "VOC2007 test",
        },
        "duplicates_removed": duplicates,
        "trainval_pool_unique": len(all_trainval),
        "splits": {split: count_split(split_records[split], split_paths[split]) for split in split_paths},
    }
    (RESULTS_DIR / "dataset_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    plot_class_distribution(stats["splits"])

    print(json.dumps({k: v["images"] for k, v in stats["splits"].items()}, indent=2))
    print(f"Duplicates removed: {duplicates}")
    print("Data preparation complete.")


if __name__ == "__main__":
    main()
