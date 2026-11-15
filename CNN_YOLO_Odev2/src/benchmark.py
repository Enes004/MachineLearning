#!/usr/bin/env python3
"""Measure per-image inference latency and FPS on GPU and CPU.

Preprocess / inference / postprocess timings come from Ultralytics' internal
per-image profiler (batch=1). Wall-clock time is measured independently.
At least 200 test images are used after warm-up.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=str(REPO_ROOT / "weights" / "best.pt"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--images", type=int, default=200, help="At least 200 test images")
    parser.add_argument("--warmup", type=int, default=10)
    return parser.parse_args()


def benchmark_device(model, image_paths: list[str], device: str, warmup: int, imgsz: int, conf: float) -> dict:
    """Run latency benchmark for one device (Bir cihaz için gecikme ölçümü)."""
    print(f"\nBenchmark on {device}: {len(image_paths)} images (+{warmup} warm-up)", flush=True)
    for path in image_paths[:warmup]:
        model.predict(source=path, imgsz=imgsz, conf=conf, device=device, verbose=False)

    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()

    pre, inf, post, totals = [], [], [], []
    wall_start = time.perf_counter()
    for i, path in enumerate(image_paths, start=1):
        t0 = time.perf_counter()
        result = model.predict(source=path, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
        t1 = time.perf_counter()
        speed = result.speed
        pre.append(float(speed["preprocess"]))
        inf.append(float(speed["inference"]))
        post.append(float(speed["postprocess"]))
        totals.append((t1 - t0) * 1000.0)
        if i % 50 == 0:
            print(f"  {i}/{len(image_paths)} done", flush=True)
    wall_elapsed = time.perf_counter() - wall_start

    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()

    def stats(values: list[float]) -> dict:
        arr = np.array(values, dtype=np.float64)
        return {
            "mean_ms": round(float(arr.mean()), 3),
            "median_ms": round(float(np.median(arr)), 3),
            "std_ms": round(float(arr.std()), 3),
            "min_ms": round(float(arr.min()), 3),
            "max_ms": round(float(arr.max()), 3),
        }

    total_ms = np.array(totals)
    return {
        "device": device,
        "device_name": torch.cuda.get_device_name(0) if device != "cpu" and torch.cuda.is_available() else platform.processor(),
        "images": len(image_paths),
        "warmup_images": warmup,
        "preprocess": stats(pre),
        "inference": stats(inf),
        "postprocess": stats(post),
        "end_to_end": stats(totals),
        "fps_from_mean_end_to_end": round(1000.0 / float(total_ms.mean()), 3),
        "fps_from_wall_clock": round(len(image_paths) / wall_elapsed, 3),
        "wall_clock_seconds": round(wall_elapsed, 3),
    }


def main() -> None:
    args = parse_args()
    if args.images < 200:
        raise ValueError("At least 200 images are required by the task.")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    test_list = (REPO_ROOT / "data" / "splits" / "test.txt").read_text(encoding="utf-8")
    all_paths = [line.strip() for line in test_list.splitlines() if line.strip()]
    image_paths = all_paths[: args.images]
    if len(image_paths) < args.images:
        raise RuntimeError(f"Only {len(image_paths)} test images available, need {args.images}")

    from ultralytics import YOLO

    model = YOLO(args.weights)
    payload = {
        "weights": str(Path(args.weights).relative_to(REPO_ROOT)),
        "imgsz": args.imgsz,
        "confidence_threshold": args.conf,
        "batch_size": 1,
        "sample": f"first {len(image_paths)} images of the VOC2007 test split list",
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": {},
    }

    payload["results"]["gpu"] = benchmark_device(
        model, image_paths, "0", args.warmup, args.imgsz, args.conf
    )
    payload["results"]["cpu"] = benchmark_device(
        model, image_paths, "cpu", args.warmup, args.imgsz, args.conf
    )

    (RESULTS_DIR / "inference_time.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2), flush=True)
    print(f"Saved {RESULTS_DIR / 'inference_time.json'}", flush=True)


if __name__ == "__main__":
    main()
