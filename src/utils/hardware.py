"""Hardware capture. Every value here is read from the machine, never assumed.

The manuscript's reproducibility section requires a hardware description, and
The reproducibility protocol requires it to be a measurement. Anything this module cannot
actually determine is reported as the string "UNKNOWN", never as a plausible guess.
"""
from __future__ import annotations

import importlib
import platform
import shutil
import subprocess
import sys

__all__ = ["capture_hardware", "format_hardware"]

_UNKNOWN = "UNKNOWN"


def _cpu_name() -> str:
    if platform.system() == "Windows":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor | Select-Object -First 1).Name"],
                capture_output=True, text=True, timeout=30,
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        except Exception:
            pass
    return platform.processor() or _UNKNOWN


def _total_ram_bytes() -> int | str:
    try:
        import psutil
        return int(psutil.virtual_memory().total)
    except Exception:
        pass
    if platform.system() == "Windows":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"],
                capture_output=True, text=True, timeout=30,
            )
            if out.returncode == 0 and out.stdout.strip().isdigit():
                return int(out.stdout.strip())
        except Exception:
            pass
    return _UNKNOWN


def _nvidia_driver() -> str:
    if shutil.which("nvidia-smi") is None:
        return _UNKNOWN
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip().splitlines()[0].strip()
    except Exception:
        pass
    return _UNKNOWN


def capture_hardware() -> dict:
    """Return a dict describing the machine. Goes verbatim into every run manifest."""
    hw: dict = {
        "os": f"{platform.system()} {platform.release()} {platform.version()}",
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "cpu": _cpu_name(),
        "cpu_count_logical": _safe_cpu_count(),
        "ram_total_bytes": _total_ram_bytes(),
        "nvidia_driver": _nvidia_driver(),
    }
    try:
        import torch
        hw["torch"] = torch.__version__
        hw["torch_cuda_build"] = torch.version.cuda or _UNKNOWN
        hw["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            hw["gpu_name"] = props.name
            hw["gpu_total_memory_bytes"] = int(props.total_memory)
            hw["gpu_compute_capability"] = f"{props.major}.{props.minor}"
            hw["gpu_multiprocessors"] = int(props.multi_processor_count)
        else:
            hw["gpu_name"] = "NONE (CPU only)"
    except ImportError:
        hw["torch"] = "NOT INSTALLED"
        hw["cuda_available"] = False
        hw["gpu_name"] = _UNKNOWN
    hw["packages"] = _package_versions()
    hw["executable"] = sys.executable
    return hw


#: Packages whose version can change a reported number. Recorded per run, because "it reproduces on
#: my machine" is not a claim anyone can check without knowing which versions produced the numbers.
#: This project was run from two interpreters at different times, which is exactly the situation
#: that makes the record worth having.
_TRACKED_PACKAGES = ("numpy", "pandas", "scipy", "sklearn", "xgboost", "lightgbm",
                     "river", "lifelines", "matplotlib")


def _package_versions() -> dict:
    out: dict = {}
    for name in _TRACKED_PACKAGES:
        try:
            mod = importlib.import_module(name)
            out[name] = str(getattr(mod, "__version__", _UNKNOWN))
        except Exception:
            out[name] = "NOT INSTALLED"
    return out


def _safe_cpu_count() -> int | str:
    try:
        import os
        n = os.cpu_count()
        return int(n) if n else _UNKNOWN
    except Exception:
        return _UNKNOWN


def format_hardware(hw: dict | None = None) -> str:
    """One-line human-readable summary, for logs and the manuscript's setup section."""
    hw = hw if hw is not None else capture_hardware()
    ram = hw.get("ram_total_bytes")
    ram_s = f"{ram / 1024**3:.2f} GiB" if isinstance(ram, int) else str(ram)
    gpu = hw.get("gpu_name", _UNKNOWN)
    vram = hw.get("gpu_total_memory_bytes")
    if isinstance(vram, int):
        gpu = f"{gpu} ({vram / 1024**3:.2f} GiB)"
    return (f"{hw.get('cpu', _UNKNOWN)} | {hw.get('cpu_count_logical', _UNKNOWN)} logical cores | "
            f"RAM {ram_s} | GPU {gpu} | {hw.get('os', _UNKNOWN)} | "
            f"python {hw.get('python', _UNKNOWN)} | torch {hw.get('torch', _UNKNOWN)}")
