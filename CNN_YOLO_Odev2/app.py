#!/usr/bin/env python3
"""Gradio demo for the fine-tuned YOLO VOC detector.

Run:
  .venv/bin/python app.py
Then open http://127.0.0.1:7860
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

REPO_ROOT = Path(__file__).resolve().parent
WEIGHTS = REPO_ROOT / "weights" / "best.pt"
IMGSZ = 640

register_heif_opener()  # HEIC/HEIF support for phone photos (telefon fotoğrafları)

from ultralytics import YOLO  # noqa: E402  (import after HEIF registration)

MODEL = YOLO(str(WEIGHTS))
# Warm up CUDA kernels so the first user request is not slow (ilk istek yavaş olmasın)
MODEL.predict(source=np.zeros((640, 640, 3), dtype=np.uint8), imgsz=IMGSZ, verbose=False)
COLUMNS = ["class", "confidence", "x1", "y1", "x2", "y2"]


def prepare_image(image: Image.Image) -> Image.Image:
    """Fix EXIF orientation and force RGB (EXIF yönünü düzelt, RGB'ye çevir)."""
    return ImageOps.exif_transpose(image).convert("RGB")


def detect(image: Image.Image, conf_threshold: float):
    """Run detection and return annotated image, table and timing (Tahmin ve çıktı)."""
    if image is None:
        raise gr.Error("Lütfen bir görüntü yükleyin / Please upload an image.")
    image = prepare_image(image)

    t0 = time.perf_counter()
    result = MODEL.predict(source=image, imgsz=IMGSZ, conf=float(conf_threshold), verbose=False)[0]
    wall_ms = (time.perf_counter() - t0) * 1000.0

    annotated = cv2.cvtColor(result.plot(line_width=2, font_size=11), cv2.COLOR_BGR2RGB)
    annotated_pil = Image.fromarray(annotated)

    rows = []
    if result.boxes is not None and len(result.boxes) > 0:
        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        for box, conf, cls_id in zip(xyxy, confs, classes):
            rows.append(
                [
                    MODEL.names[int(cls_id)],
                    round(float(conf), 4),
                    round(float(box[0]), 1),
                    round(float(box[1]), 1),
                    round(float(box[2]), 1),
                    round(float(box[3]), 1),
                ]
            )
    table = pd.DataFrame(rows, columns=COLUMNS)
    table = table.sort_values("confidence", ascending=False).reset_index(drop=True)

    speed = result.speed
    total_ms = float(sum(speed.values()))
    timing = (
        f"**{len(rows)} nesne / objects** | "
        f"Toplam {total_ms:.1f} ms "
        f"(preprocess {speed['preprocess']:.1f} ms, "
        f"inference {speed['inference']:.1f} ms, "
        f"postprocess {speed['postprocess']:.1f} ms) | "
        f"wall-clock {wall_ms:.1f} ms | "
        f"~{1000.0 / max(wall_ms, 1e-6):.1f} FPS"
    )
    return annotated_pil, table, timing


def build_demo() -> gr.Blocks:
    """Build the Gradio interface (Gradio arayüzünü kur)."""
    with gr.Blocks(title="VOC Object Detection - YOLO") as demo:
        gr.Markdown(
            "# Pascal VOC Nesne Tespiti\n"
            "Fine-tuned YOLO modeli ile 20 VOC sınıfında nesne tespiti. "
            "Bir görüntü yükleyin, confidence eşiğini ayarlayın ve **Tespit Et**'e basın.\n\n"
            "Desteklenen formatlar: JPG, PNG, WEBP, HEIC/HEIF (telefon fotoğrafları). "
            "EXIF yönü otomatik düzeltilir."
        )
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Görüntü / Image", sources=["upload", "clipboard"])
                conf_slider = gr.Slider(
                    minimum=0.05, maximum=0.95, value=0.25, step=0.05,
                    label="Confidence threshold (güven eşiği)",
                )
                detect_button = gr.Button("Tespit Et / Detect", variant="primary")
            with gr.Column():
                image_output = gr.Image(type="pil", label="Tespitler / Detections")
                timing_output = gr.Markdown("Inference süresi burada gösterilecek.")
                table_output = gr.Dataframe(
                    headers=COLUMNS,
                    datatype=["str", "number", "number", "number", "number", "number"],
                    label="Tespit tablosu / Detection table",
                    interactive=False,
                    wrap=True,
                )
        detect_button.click(
            fn=detect,
            inputs=[image_input, conf_slider],
            outputs=[image_output, table_output, timing_output],
        )
        image_input.change(fn=lambda: "", outputs=timing_output)
    return demo


if __name__ == "__main__":
    demo = build_demo()
    port = int(os.environ.get("APP_PORT", "7860"))
    demo.launch(server_name="127.0.0.1", server_port=port, share=False, show_error=True)
