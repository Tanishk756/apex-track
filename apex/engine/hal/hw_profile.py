"""
Hardware Abstraction Layer — Capability Model
=============================================
Exposes WHAT the hardware can do, not WHAT it is.
Every downstream module queries HWCapabilities flags, never device names.

Design:
- HWCapabilities is a flat frozen dataclass of boolean flags + numeric limits.
- HWProfile bundles capabilities with memory/compute budget info.
- Modules use .supports(Capability.TENSORRT) not `if 'jetson' in device_name`.
- This allows seamless operation across RTX, Jetson, RPi, and CPU-only systems.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Flag, auto


class Capability(Flag):
    """
    Hardware capability flags — use bitwise OR/AND for multi-capability checks.

    Example:
        if hw.has(Capability.TENSORRT | Capability.FP16):
            use_trt_fp16()
    """
    NONE         = 0

    # Compute backends
    CUDA         = auto()   # NVIDIA CUDA available
    TENSORRT     = auto()   # TensorRT engine available
    CUDNN        = auto()   # cuDNN acceleration
    OPENCL       = auto()   # OpenCL (AMD/Intel/ARM)
    VULKAN       = auto()   # Vulkan Compute
    METAL        = auto()   # Apple Metal (macOS/iOS)
    AVX2         = auto()   # x86 AVX2 SIMD
    AVX512       = auto()   # x86 AVX-512
    ARM_NEON     = auto()   # ARM NEON SIMD
    OPENVINO     = auto()   # Intel OpenVINO runtime

    # Precision
    FP16         = auto()   # Native FP16 tensor cores
    INT8         = auto()   # INT8 quantized inference
    BF16         = auto()   # BFloat16 support

    # Video decode / encode
    NVDEC        = auto()   # NVIDIA hardware video decode (NVDEC/NVMEDIA)
    NVENC        = auto()   # NVIDIA hardware video encode
    V4L2M2M     = auto()   # V4L2 memory-to-memory (RPi / ARM)
    VAAPI        = auto()   # VA-API (Intel/AMD Linux)
    QUICKSYNC    = auto()   # Intel Quick Sync

    # Memory
    ZERO_COPY    = auto()   # Unified memory / zero-copy GPU↔CPU
    PINNED_MEM   = auto()   # CUDA pinned (page-locked) memory
    DMA          = auto()   # DMA transfer support (Jetson)

    # Special hardware
    JETSON       = auto()   # Running on NVIDIA Jetson platform
    RPI          = auto()   # Running on Raspberry Pi
    NPU          = auto()   # Dedicated Neural Processing Unit


@dataclass(frozen=True, slots=True)
class HWCapabilities:
    """
    Resolved capability set for the current hardware.
    Produced by hw_detector.py; consumed by engine_factory and optimizer.
    """

    flags: Capability = Capability.NONE
    """Combined capability flags."""

    # GPU info
    gpu_name: str = "CPU-only"
    gpu_count: int = 0
    gpu_vram_mb: int = 0
    gpu_compute_capability: tuple[int, int] = (0, 0)  # e.g. (8, 6) for Ampere

    # CPU info
    cpu_cores: int = 1
    cpu_threads: int = 1
    cpu_arch: str = "unknown"   # 'x86_64' | 'aarch64' | 'armv7l'

    # System memory
    ram_total_mb: int = 0
    ram_available_mb: int = 0

    # Performance budget (filled by Optimizer)
    max_inference_batch: int = 1
    recommended_fp_precision: str = "fp32"  # 'fp32' | 'fp16' | 'int8'
    max_decode_threads: int = 2
    cuda_stream_count: int = 1

    def has(self, cap: Capability) -> bool:
        """Check if ALL supplied capability flags are present (AND semantics)."""
        return (self.flags & cap) == cap

    def __repr__(self) -> str:
        active = [c.name for c in Capability if c != Capability.NONE and self.has(c)]
        return f"HWCapabilities(gpu={self.gpu_name!r}, caps=[{', '.join(active)}])"


@dataclass
class HWProfile:
    """
    The authoritative hardware context object passed through the entire engine.
    Created once at startup by hw_detector.detect() and injected everywhere.
    """

    capabilities: HWCapabilities
    profile_name: str = "unknown"
    """Human-readable label: 'desktop_rtx', 'jetson_orin_nx', 'rpi5', 'cpu_only'."""

    platform_notes: list[str] = field(default_factory=list)
    """Informational notes from the detector (e.g. 'TensorRT 8.6.1 found')."""

    def has(self, cap: Capability) -> bool:
        return self.capabilities.has(cap)

    @property
    def is_gpu(self) -> bool:
        return self.has(Capability.CUDA)

    @property
    def is_jetson(self) -> bool:
        return self.has(Capability.JETSON)

    @property
    def is_rpi(self) -> bool:
        return self.has(Capability.RPI)

    def __repr__(self) -> str:
        return (
            f"HWProfile(name={self.profile_name!r}, "
            f"gpu={self.capabilities.gpu_name!r})"
        )
