"""
Unit Tests — Hardware Detector
Tests that hw_detector.detect() returns a valid HWProfile on any machine,
even without CUDA/GPU installed.
"""

import pytest
from apex.engine.hal import hw_detector
from apex.engine.hal.hw_profile import Capability, HWCapabilities, HWProfile


class TestHWDetector:

    def test_detect_returns_hw_profile(self):
        """detect() must return HWProfile regardless of hardware."""
        profile = hw_detector.detect()
        assert isinstance(profile, HWProfile)

    def test_profile_name_is_set(self):
        profile = hw_detector.detect()
        assert profile.profile_name != ""
        assert profile.profile_name is not None

    def test_capabilities_instance(self):
        profile = hw_detector.detect()
        assert isinstance(profile.capabilities, HWCapabilities)

    def test_cpu_cores_positive(self):
        profile = hw_detector.detect()
        assert profile.capabilities.cpu_cores >= 1
        assert profile.capabilities.cpu_threads >= 1

    def test_ram_detected(self):
        profile = hw_detector.detect()
        assert profile.capabilities.ram_total_mb > 0

    def test_has_method(self):
        profile = hw_detector.detect()
        # has() should not raise for any capability
        for cap in Capability:
            result = profile.has(cap)
            assert isinstance(result, bool)

    def test_cpu_arch_set(self):
        profile = hw_detector.detect()
        assert profile.capabilities.cpu_arch in ("x86_64", "aarch64", "armv7l", "arm64", "AMD64")

    def test_recommended_precision_valid(self):
        profile = hw_detector.detect()
        assert profile.capabilities.recommended_fp_precision in (
            "fp32", "fp16", "int8", "fp32_cuda"
        )

    def test_no_cuda_on_cpu_only(self, monkeypatch):
        """When torch reports CUDA unavailable, CUDA flag must not be set."""
        import apex.engine.hal.hw_detector as det_mod

        def mock_probe_cuda():
            return {"available": False, "gpus": [], "compute": (0, 0)}

        monkeypatch.setattr(det_mod, "_probe_cuda", mock_probe_cuda)
        profile = hw_detector.detect()
        assert not profile.has(Capability.CUDA)
        assert not profile.has(Capability.TENSORRT)

    def test_cuda_flag_set_with_gpu(self, monkeypatch):
        """When torch reports a GPU, CUDA flag must be set."""
        import apex.engine.hal.hw_detector as det_mod

        def mock_probe_cuda():
            return {
                "available": True,
                "gpus": [{"name": "RTX 4090", "vram_mb": 24576, "cc": (8, 9)}],
                "compute": (8, 9),
            }

        monkeypatch.setattr(det_mod, "_probe_cuda", mock_probe_cuda)
        profile = hw_detector.detect()
        assert profile.has(Capability.CUDA)
        assert profile.has(Capability.FP16)
        assert profile.has(Capability.INT8)


class TestCapabilityFlags:

    def test_flag_combination(self):
        combined = Capability.CUDA | Capability.FP16 | Capability.TENSORRT
        caps = HWCapabilities(flags=combined, gpu_name="RTX")
        assert caps.has(Capability.CUDA)
        assert caps.has(Capability.FP16)
        assert caps.has(Capability.TENSORRT)
        assert not caps.has(Capability.OPENCL)

    def test_no_flags(self):
        caps = HWCapabilities(flags=Capability.NONE)
        assert not caps.has(Capability.CUDA)

    def test_multi_flag_check(self):
        caps = HWCapabilities(flags=Capability.CUDA | Capability.FP16)
        # has() with single flags
        assert caps.has(Capability.CUDA)
        assert caps.has(Capability.FP16)
        assert not caps.has(Capability.TENSORRT)
        # has() with combined flags checks ALL must be present
        assert caps.has(Capability.CUDA | Capability.FP16)      # both present
        assert not caps.has(Capability.CUDA | Capability.TENSORRT)  # TENSORRT missing
