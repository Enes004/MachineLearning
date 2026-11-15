#!/usr/bin/env python3
"""Launch the Gradio app headlessly, upload one validation image, and save a screenshot.

Usage:
  env -u PYTHONPATH .venv/bin/python src/screenshot_app.py
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "report" / "figures" / "app_screenshot.png"
PORT = 7861


def wait_until_ready(url: str, timeout: float = 180.0) -> None:
    """Wait for the Gradio server to answer (Sunucunun hazır olmasını bekle)."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(2)
    raise TimeoutError(f"Gradio app did not start within {timeout} seconds")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=None, help="Image to upload (default: first val image)")
    args = parser.parse_args()
    if args.image:
        image_path = str(Path(args.image).resolve())
    else:
        val_list = (REPO_ROOT / "data" / "splits" / "val.txt").read_text(encoding="utf-8")
        image_path = val_list.splitlines()[0].strip()
    if not Path(image_path).exists():
        raise FileNotFoundError(image_path)
    print(f"Uploading {image_path}", flush=True)

    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(
        {
            "APP_PORT": str(PORT),
            "GRADIO_ANALYTICS_ENABLED": "False",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }
    )
    log = open(REPO_ROOT / "results" / "app_server.log", "w", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, str(REPO_ROOT / "app.py")],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    try:
        wait_until_ready(f"http://127.0.0.1:{PORT}/")
        time.sleep(3)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1300}, device_scale_factor=1.5)
            page.goto(f"http://127.0.0.1:{PORT}/", wait_until="networkidle", timeout=120000)
            page.wait_for_timeout(3000)

            file_inputs = page.locator("input[type=file]")
            print(f"file inputs: {file_inputs.count()}", flush=True)
            file_inputs.first.set_input_files(image_path)
            page.wait_for_timeout(4000)

            button = page.get_by_role("button", name=re.compile("Tespit|Detect", re.IGNORECASE))
            button.first.click()
            page.wait_for_timeout(10000)
            page.screenshot(path=str(OUTPUT), full_page=True)
            browser.close()
        print(f"Saved {OUTPUT}", flush=True)
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
        log.close()


if __name__ == "__main__":
    main()
