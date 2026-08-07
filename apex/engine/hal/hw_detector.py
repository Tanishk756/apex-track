"""
Hardware Abstraction Layer — Auto-Detector
==========================================
Probes the current system and returns a fully-populated HWProfile.

Detection order (fast → comprehensive):
1. Read /proc/cpuinfo, /proc/meminfo for CPU and RAM
2. Probe PyTorch for CUDA devices and compute capability
3. Attempt TensorRT import to confirm availability
4. Check for Jetson-specific files (/etc/nv_tegra_release)
5. Check for Raspberry Pi (/proc/device-tree/model)
6. Probe OpenCL via pyopencl (optional)
7. Check NVDEC via subprocess (ffmpeg -hwaccels)
8. Populate HWCapabilities flags and HWProfile

All probes are wrapped in try/except — detection NEVER raises.
If a probe fails, that capability is simply marked absent.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import structlog

from apex.engine.hal.hw_profile import Capability, HWCapabilities, HWProfile

log = structlog.get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Internal probe helpers
# ──────────────────────────────────────────────────────────────────────────────

def _probe_cpu() -> dict:
    """Return basic CPU info."""
    result = {
        "arch": platform.machine(),
        "cores": os.cpu_count() or 1,
        "threads": os.cpu_count() or 1,
    }
    # Try reading logical vs physical core count on Linux
    try:
        import psutil
        result["cores"]   = psutil.cpu_count(logical=False) or 1
        result["threads"] = psutil.cpu_count(logical=True)  or 1
    except Exception:
        pass
    return result


def _probe_ram() -> dict:
    """Return total and available RAM in MB."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return {
            "total_mb":     vm.total     // (1024 * 1024),
            "available_mb": vm.available // (1024 * 1024),
        }
    except Exception:
        return {"total_mb": 0, "available_mb": 0}


def _probe_cuda() -> dict:
    """Probe CUDA / GPU info via PyTorch."""
    result: dict = {"available": False, "gpus": [], "compute": (0, 0)}
    try:
        import torch
        if not torch.cuda.is_available():
            return result
        result["available"] = True
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            result["gpus"].append({
                "name":    props.name,
                "vram_mb": props.total_memory // (1024 * 1024),
                "cc":      (props.major, props.minor),
            })
        if result["gpus"]:
            result["compute"] = result["gpus"][0]["cc"]
    except Exception as exc:
        log.debug("cuda_probe_failed", reason=str(exc))
    return result


def _probe_fp16(compute: tuple[int, int]) -> bool:
    """FP16 Tensor Cores require compute capability >= 7.0 (Volta+)."""
    return compute >= (7, 0)


def _probe_int8(compute: tuple[int, int]) -> bool:
    """INT8 Tensor Cores require compute capability >= 6.1 (Pascal+)."""
    return compute >= (6, 1)


def _probe_tensorrt() -> tuple[bool, str]:
    """Try importing TensorRT and return (available, version_str)."""
    try:
        import tensorrt as trt  # noqa: F401
        return True, trt.__version__
    except ImportError:
        pass
    # Try trtexec in PATH as fallback
    if shutil.which("trtexec"):
        return True, "unknown (trtexec found)"
    return False, ""


def _probe_openvino() -> bool:
    try:
        from openvino.runtime import Core  # noqa: F401
        return True
    except ImportError:
        return False


def _probe_opencl() -> bool:
    try:
        import pyopencl  # noqa: F401
        platforms = pyopencl.get_platforms()
        return len(platforms) > 0
    except Exception:
        return False


def _probe_hwdec() -> dict[str, bool]:
    """Check hardware video decoder availability via ffmpeg."""
    result = {"nvdec": False, "nvenc": False, "v4l2m2m": False, "vaapi": False, "qsv": False}
    if not shutil.which("ffmpeg"):
        return result
    try:
        out = subprocess.check_output(
            ["ffmpeg", "-hide_banner", "-hwaccels"],
            stderr=subprocess.DEVNULL, text=True, timeout=5
        )
        lower = out.lower()
        result["nvdec"]   = "cuda"   in lower or "nvdec" in lower
        result["nvenc"]   = "cuda"   in lower
        result["v4l2m2m"] = "v4l2m2m" in lower
        result["vaapi"]   = "vaapi"  in lower
        result["qsv"]     = "qsv"    in lower
    except Exception:
        pass
    return result


def _probe_jetson() -> tuple[bool, str]:
    """Detect Jetson platform by checking NVIDIA Tegra release file."""
    p = Path("/etc/nv_tegra_release")
    if p.exists():
        try:
            return True, p.read_text().strip().split("\n")[0]
        except Exception:
            return True, "Jetson (version unknown)"
    # Also check /proc/device-tree/compatible
    p2 = Path("/proc/device-tree/compatible")
    if p2.exists():
        try:
            content = p2.read_bytes().decode("ascii", errors="ignore").lower()
            if "nvidia" in content or "tegra" in content:
                return True, "Jetson (via device-tree)"
        except Exception:
            pass
    return False, ""


def _probe_rpi() -> tuple[bool, str]:
    """Detect Raspberry Pi by model string."""
    p = Path("/proc/device-tree/model")
    if p.exists():
        try:
            model = p.read_bytes().decode("ascii", errors="ignore").strip("\x00")
            if "raspberry pi" in model.lower():
                return True, model
        except Exception:
            pass
    p2 = Path("/proc/cpuinfo")
    if p2.exists():
        try:
            content = p2.read_text()
            if "raspberry pi" in content.lower() or "bcm27" in content.lower():
                return True, "Raspberry Pi (via cpuinfo)"
        except Exception:
            pass
    return False, ""


def _probe_avx() -> tuple[bool, bool]:
    """Check AVX2 / AVX-512 support from /proc/cpuinfo."""
    try:
        flags_line = ""
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags"):
                    flags_line = line
                    break
        avx2   = "avx2"   in flags_line
        avx512 = "avx512f" in flags_line
        return avx2, avx512
    except Exception:
        return False, False


def _probe_neon() -> bool:
    """Check ARM NEON from /proc/cpuinfo."""
    try:
        with open("/proc/cpuinfo") as f:
            content = f.read()
        return "neon" in content.lower() or "asimd" in content.lower()
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def detect() -> HWProfile:
    """
    Auto-detect all hardware capabilities and return an HWProfile.

    This is the ONLY entry point external code should call.
    Safe to call at any time — never raises.
    """
    notes: list[str] = []
    flags = Capability.NONE

    # ── CPU ──────────────────────────────────────────────────────────────────
    cpu = _probe_cpu()
    ram = _probe_ram()
    arch = cpu["arch"]

    avx2, avx512 = _probe_avx()
    neon = _probe_neon()
    if avx2:   flags |= Capability.AVX2
    if avx512: flags |= Capability.AVX512
    if neon:   flags |= Capability.ARM_NEON

    # ── Platform ─────────────────────────────────────────────────────────────
    is_jetson, jetson_note = _probe_jetson()
    is_rpi,    rpi_note    = _probe_rpi()
    if is_jetson:
        flags |= Capability.JETSON
        notes.append(f"Jetson: {jetson_note}")
    if is_rpi:
        flags |= Capability.RPI
        notes.append(f"RPi: {rpi_note}")

    # ── CUDA / GPU ────────────────────────────────────────────────────────────
    cuda_info = _probe_cuda()
    gpu_name = "CPU-only"
    gpu_count = 0
    gpu_vram_mb = 0
    compute = (0, 0)

    if cuda_info["available"] and cuda_info["gpus"]:
        flags |= Capability.CUDA
        gpu_count = len(cuda_info["gpus"])
        gpu_name    = cuda_info["gpus"][0]["name"]
        gpu_vram_mb = cuda_info["gpus"][0]["vram_mb"]
        compute     = cuda_info["compute"]
        notes.append(f"GPU[0]: {gpu_name} ({gpu_vram_mb} MB VRAM, CC {compute[0]}.{compute[1]})")

        if _probe_fp16(compute):
            flags |= Capability.FP16
        if _probe_int8(compute):
            flags |= Capability.INT8

        # Zero-copy (unified memory) on Jetson
        if is_jetson:
            flags |= Capability.ZERO_COPY
            flags |= Capability.DMA
            flags |= Capability.PINNED_MEM

    # ── TensorRT ─────────────────────────────────────────────────────────────
    if cuda_info["available"]:
        trt_ok, trt_ver = _probe_tensorrt()
        if trt_ok:
            flags |= Capability.TENSORRT
            notes.append(f"TensorRT {trt_ver}")

    # ── Hardware video decode ─────────────────────────────────────────────────
    hwdec = _probe_hwdec()
    if hwdec["nvdec"] and cuda_info["available"]:
        flags |= Capability.NVDEC
        flags |= Capability.NVENC
    if hwdec["v4l2m2m"]:  flags |= Capability.V4L2M2M
    if hwdec["vaapi"]:    flags |= Capability.VAAPI
    if hwdec["qsv"]:      flags |= Capability.QUICKSYNC

    # ── OpenVINO / OpenCL ─────────────────────────────────────────────────────
    if _probe_openvino():
        flags |= Capability.OPENVINO
        notes.append("OpenVINO runtime found")
    if _probe_opencl():
        flags |= Capability.OPENCL

    # ── Recommended precision ─────────────────────────────────────────────────
    if (flags & Capability.TENSORRT) and (flags & Capability.FP16):
        rec_precision = "fp16"
    elif flags & Capability.INT8:
        rec_precision = "int8"
    elif flags & Capability.CUDA:
        rec_precision = "fp32_cuda"
    else:
        rec_precision = "fp32"

    # ── Stream / batch budget ─────────────────────────────────────────────────
    cuda_streams = min(4, gpu_count * 2) if gpu_count else 1
    max_batch    = 4 if flags & Capability.TENSORRT else 1
    dec_threads  = min(cpu["threads"], 4)

    # ── Profile name ──────────────────────────────────────────────────────────
    if is_jetson:
        if gpu_vram_mb >= 8000:
            profile_name = "jetson_agx_orin"
        elif gpu_vram_mb >= 4000:
            profile_name = "jetson_orin_nx"
        else:
            profile_name = "jetson_nano"
    elif is_rpi:
        profile_name = "raspberry_pi"
    elif flags & Capability.TENSORRT:
        profile_name = "desktop_rtx"
    elif flags & Capability.CUDA:
        profile_name = "desktop_cuda"
    else:
        profile_name = "cpu_only"

    caps = HWCapabilities(
        flags=flags,
        gpu_name=gpu_name,
        gpu_count=gpu_count,
        gpu_vram_mb=gpu_vram_mb,
        gpu_compute_capability=compute,
        cpu_cores=cpu["cores"],
        cpu_threads=cpu["threads"],
        cpu_arch=arch,
        ram_total_mb=ram["total_mb"],
        ram_available_mb=ram["available_mb"],
        max_inference_batch=max_batch,
        recommended_fp_precision=rec_precision,
        max_decode_threads=dec_threads,
        cuda_stream_count=cuda_streams,
    )

    profile = HWProfile(
        capabilities=caps,
        profile_name=profile_name,
        platform_notes=notes,
    )

    log.info(
        "hw_detection_complete",
        profile=profile_name,
        gpu=gpu_name,
        precision=rec_precision,
        caps=[c.name for c in Capability if c != Capability.NONE and caps.has(c)],
    )
    return profile
