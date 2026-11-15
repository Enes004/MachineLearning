#!/usr/bin/env python3
"""Record the software/hardware environment into results/env.json."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"


def nvidia_query(field: str) -> str | None:
    """Query nvidia-smi for one field (nvidia-smi'den alan sorgula)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return out.splitlines()[0] if out else None
    except Exception:
        return None


def driver_cuda_version() -> str | None:
    """Parse the CUDA version reported by the driver (Sürücünün CUDA sürümü)."""
    try:
        out = subprocess.check_output(["nvidia-smi"], text=True, stderr=subprocess.DEVNULL)
        import re

        match = re.search(r"CUDA Version:\s*([\d.]+)", out)
        return match.group(1) if match else None
    except Exception:
        return None


def version_of(module_name: str) -> str | None:
    """Return installed module version without importing heavy code (Sürüm oku)."""
    try:
        module = __import__(module_name)
        return getattr(module, "__version__", None)
    except Exception:
        return None


def main() -> None:
    import torch

    total_ram_gb = None
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total_ram_gb = round(int(line.split()[1]) / 1024 / 1024, 2)
                    break
    except Exception:
        pass

    payload = {
        "os": platform.platform(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "cpu": platform.processor(),
        "cpu_count": torch.get_num_threads(),
        "ram_gb": total_ram_gb,
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "gpu": {
            "name": nvidia_query("name"),
            "vram_mib": nvidia_query("memory.total"),
            "driver_version": nvidia_query("driver_version"),
            "cuda_version_from_driver": driver_cuda_version(),
            "cuda_version_torch": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_name_torch": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "device_count": torch.cuda.device_count(),
        },
        "packages": {
            "torch": torch.__version__,
            "torchvision": version_of("torchvision"),
            "ultralytics": version_of("ultralytics"),
            "gradio": version_of("gradio"),
            "numpy": version_of("numpy"),
            "opencv_python": version_of("cv2"),
            "pandas": version_of("pandas"),
            "matplotlib": version_of("matplotlib"),
            "scikit_learn": version_of("sklearn"),
            "pillow": version_of("PIL"),
            "pillow_heif": version_of("pillow_heif"),
            "python_docx": version_of("docx"),
        },
        "disk_free_gb": round(shutil.disk_usage(REPO_ROOT).free / 1024**3, 2),
        "command": " ".join(sys.argv),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "env.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
