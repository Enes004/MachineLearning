#!/usr/bin/env python3
"""Train a YOLO nano detector on the custom Pascal VOC split.

Examples:
  python src/train.py --smoke
  python src/train.py --epochs 30
  nohup python src/train.py --epochs 30 > results/train.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend (etkileşimsiz arka uç)
import matplotlib.pyplot as plt
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "report" / "figures"
WEIGHTS_DIR = REPO_ROOT / "weights"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "train_config.yaml"))
    parser.add_argument("--smoke", action="store_true", help="1 epoch on 5%% of the data")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    parser.add_argument("--fraction", type=float, default=None, help="Dataset fraction")
    parser.add_argument("--name", default=None, help="Run directory name")
    parser.add_argument("--resume", action="store_true", help="Resume the last run")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def plot_training_curves(results_csv: Path, out_dir: Path) -> None:
    """Plot loss and validation metric curves from results.csv (Eğri grafikleri)."""
    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]
    epochs = df["epoch"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for col, label in (
        ("train/box_loss", "train box loss"),
        ("train/cls_loss", "train cls loss"),
        ("train/dfl_loss", "train dfl loss"),
        ("train/l1_loss", "train l1 loss (YOLO26)"),
    ):
        if col in df and (df[col].abs().sum() > 0 or col == "train/box_loss"):
            axes[0].plot(epochs, df[col], label=label.replace("train ", ""))
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Training losses")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    for col, label in (
        ("metrics/mAP50(B)", "mAP@0.5"),
        ("metrics/mAP50-95(B)", "mAP@0.5:0.95"),
    ):
        if col in df:
            axes[1].plot(epochs, df[col], label=label)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("metric")
    axes[1].set_title("Validation metrics")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    out = out_dir / "train_loss_curves.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for col, label in (("metrics/precision(B)", "precision"), ("metrics/recall(B)", "recall")):
        if col in df:
            axes[0].plot(epochs, df[col], label=label)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("score")
    axes[0].set_title("Validation precision / recall")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    if "lr/pg0" in df:
        axes[1].plot(epochs, df["lr/pg0"], label="lr/pg0", color="tab:red")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("learning rate")
    axes[1].set_title("Learning rate schedule")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "train_precision_recall_lr.png", dpi=180)
    plt.close(fig)


def copy_run_artifacts(run_dir: Path, is_full: bool, best_path: Path | None) -> None:
    """Copy Ultralytics figures and final weights to stable locations (Çıktıları kopyala)."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    mapping = {
        "results.png": "train_results.png",
        "BoxPR_curve.png": "pr_curve.png",
        "confusion_matrix.png": "confusion_matrix.png",
        "confusion_matrix_normalized.png": "confusion_matrix_normalized.png",
        "BoxF1_curve.png": "f1_curve.png",
        "BoxP_curve.png": "precision_curve.png",
        "BoxR_curve.png": "recall_curve.png",
    }
    for src_name, dst_name in mapping.items():
        src = run_dir / src_name
        if src.exists():
            shutil.copy2(src, FIGURES_DIR / dst_name)
    results_csv = run_dir / "results.csv"
    if results_csv.exists():
        plot_training_curves(results_csv, FIGURES_DIR)
    if is_full and best_path is not None and best_path.exists():
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_path, WEIGHTS_DIR / "best.pt")
        print(f"Copied best weights to {WEIGHTS_DIR / 'best.pt'}", flush=True)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    cfg = load_config(config_path)

    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch is not None:
        cfg["batch"] = args.batch
    if args.name is not None:
        cfg["name"] = args.name
    if args.smoke:
        cfg.update({"epochs": 1, "fraction": 0.05, "name": "voc_smoke", "patience": 1})
        print("SMOKE TEST: 1 epoch on 5% of the data", flush=True)
    elif args.fraction is not None:
        cfg["fraction"] = args.fraction

    if args.resume:
        cfg["resume"] = True

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = REPO_ROOT / cfg["project"] / cfg["name"]

    print("Training configuration:", flush=True)
    print(json.dumps(cfg, indent=2), flush=True)

    from ultralytics import YOLO

    model_name = cfg["model"]
    model = YOLO(cfg.pop("model"))
    data_yaml = cfg.get("data")
    project = cfg.pop("project")
    name = cfg.pop("name")
    resume = cfg.pop("resume", False)
    exist_ok = bool(cfg.pop("exist_ok", False))

    start = time.perf_counter()
    model.train(project=str(REPO_ROOT / project), name=name, exist_ok=exist_ok, resume=resume, **cfg)
    elapsed = time.perf_counter() - start

    run_dir = Path(model.trainer.save_dir)
    best_path = run_dir / "weights" / "best.pt"
    copy_run_artifacts(run_dir, is_full=not args.smoke, best_path=best_path)
    args_yaml = run_dir / "args.yaml"
    if args_yaml.exists():
        shutil.copy2(args_yaml, RESULTS_DIR / "train_args.yaml")

    # Persist measured wall-clock time and the exact configuration (Gerçek süreyi kaydet)
    out_name = "smoke_time.json" if args.smoke else "train_time.json"
    payload = {
        "wall_clock_seconds": round(elapsed, 2),
        "wall_clock_human": time.strftime("%H:%M:%S", time.gmtime(elapsed)),
        "epochs_completed": int(model.trainer.epoch + 1),
        "run_dir": str(run_dir.relative_to(REPO_ROOT)),
        "best_weights": str(best_path.relative_to(REPO_ROOT)) if best_path.exists() else None,
        "model": model_name,
        "data": data_yaml,
        "config": cfg,
        "smoke": bool(args.smoke),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (RESULTS_DIR / out_name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nTraining finished in {payload['wall_clock_human']} (wall clock).", flush=True)
    print(f"Saved {RESULTS_DIR / out_name}", flush=True)


if __name__ == "__main__":
    main()
