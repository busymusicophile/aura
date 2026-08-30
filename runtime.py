"""
AURA — process bootstrap.

`bootstrap()` must be the first thing every entry point calls, before importing
onnxruntime or any GPU library.

Why this exists: no CUDA toolkit is installed on this machine. PyTorch ships the
CUDA runtime DLLs inside its own package, and onnxruntime cannot find them on its
own. Without this, `onnxruntime.get_available_providers()` still cheerfully lists
CUDAExecutionProvider, sessions silently bind to CPU instead, and face
recognition runs ~10x slower with no error anywhere. That failure mode cost real
debugging time during setup - do not remove this.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from loguru import logger

from aura import config

_BOOTSTRAPPED = False


def enable_cuda_dlls() -> bool:
    """Add PyTorch's bundled CUDA/cuDNN DLLs to the loader search path."""
    torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
    if not torch_lib.exists():
        logger.warning("torch/lib not found at {} - GPU providers will fail", torch_lib)
        return False
    os.add_dll_directory(str(torch_lib))
    os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")
    return True


def setup_logging(name: str = "aura", level: str = "INFO") -> None:
    """Console logging plus a rotating file log under the data directory."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<dim>{time:HH:mm:ss}</dim> <level>{level: <7}</level> "
        "<cyan>{name}</cyan> | {message}",
    )
    logger.add(
        config.LOG_DIR / f"{name}.log",
        level="DEBUG",
        rotation="10 MB",
        retention="30 days",
        encoding="utf-8",
    )


def bootstrap(name: str = "aura", level: str = "INFO") -> None:
    """Idempotent one-time process setup."""
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return

    config.ensure_dirs()
    setup_logging(name, level)

    if enable_cuda_dlls():
        logger.debug("CUDA DLLs exposed from torch/lib")

    _BOOTSTRAPPED = True


def gpu_report() -> dict[str, object]:
    """Describe GPU availability. Used by diagnostics and the control panel."""
    info: dict[str, object] = {"torch_cuda": False, "onnx_cuda": False}
    try:
        import torch

        info["torch_cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["device"] = props.name
            info["vram_gb"] = round(props.total_memory / 1024**3, 1)
            info["vram_free_gb"] = round(
                torch.cuda.mem_get_info()[0] / 1024**3, 1
            )
    except Exception as exc:  # noqa: BLE001
        info["torch_error"] = str(exc)

    try:
        import onnxruntime as ort

        info["onnx_cuda"] = "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception as exc:  # noqa: BLE001
        info["onnx_error"] = str(exc)

    return info
