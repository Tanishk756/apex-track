"""
Config System — Pydantic Schemas
=================================
Single-source schema definitions for all configuration keys.
Every field has a default so partial configs are always valid.

Profile inheritance chain:
    default.yaml
        ↓ merged by
    desktop_rtx.yaml / jetson_orin_nx.yaml / cpu_only.yaml
        ↓ merged by
    user-provided config file (--config flag)
        ↓ overridden by
    environment variables (APEX_* prefix)
        ↓ overridden by
    runtime API calls (hot-reload)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ─── Sub-schemas ──────────────────────────────────────────────────────────────

class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: Literal["console", "json"] = "console"
    file: Optional[Path] = None
    max_bytes: int = 10 * 1024 * 1024  # 10 MB
    backup_count: int = 5


class HardwareConfig(BaseModel):
    force_cpu: bool = False
    """Override: disable GPU even if available."""
    cuda_device: int = 0
    """CUDA device index."""
    tensorrt_workspace_mb: int = 1024
    """TensorRT builder workspace size in MB."""
    accept_agpl_plugins: bool = False
    """Accept AGPL-3.0 plugin licenses without interactive prompt."""
    pinned_memory: bool = True
    cuda_streams: int = 2


class DetectorConfig(BaseModel):
    plugin: str = "rtdetr"
    """Name of the detector plugin to load from the registry."""
    model_name: str = "rtdetr_r50vd_coco"
    use_real_ai: bool = True
    confidence_threshold: float = Field(default=0.45, ge=0.0, le=1.0)

    nms_iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    max_detections: int = Field(default=300, ge=1)
    input_size: tuple[int, int] = (640, 640)
    fp_precision: Literal["fp32", "fp16", "int8"] = "fp16"
    class_filter: list[str] = []
    """If non-empty, only these class names are forwarded to the tracker."""


class TrackerConfig(BaseModel):
    plugin: str = "bytetrack"
    """Name of the tracker plugin."""
    min_hits: int = 1
    """Frames before a TENTATIVE track becomes CONFIRMED."""
    max_misses: int = 30
    """Frames of no detection before a CONFIRMED track becomes LOST."""
    iou_threshold: float = 0.3
    track_high_thresh: float = 0.20   # ByteTrack: high-score detection threshold
    track_low_thresh: float = 0.1    # ByteTrack: low-score association threshold
    match_thresh: float = 0.8
    reid_enabled: bool = True
    """Enable Re-ID appearance embedding (BoT-SORT / StrongSORT)."""


class AdaptiveSchedulerConfig(BaseModel):
    detect_every: int = Field(default=3, ge=1, le=30)
    """Run detector every N frames. Tracker runs on all other frames."""
    min_detect_every: int = 1
    """Minimum interval (fastest detection rate)."""
    max_detect_every: int = 10
    """Maximum interval (slowest detection rate, pure tracking)."""
    gpu_load_high_pct: float = 80.0
    """GPU % above which detect_every is increased."""
    confidence_low_thresh: float = 0.4
    """Track confidence below which detect_every is decreased."""
    optical_flow_trigger: bool = True
    """Trigger a detection frame when large untracked motion is detected."""


class TelemetryConfig(BaseModel):
    enabled: bool = True
    plugin: str = "mavlink_udp"
    connection: str = "udp:0.0.0.0:14550"
    """MAVLink connection string. Format: udp:host:port | tcp:host:port | /dev/ttyUSBx:57600"""
    timeout_s: float = 5.0
    reconnect_interval_s: float = 3.0
    source_system: int = 255
    source_component: int = 0


class CameraConfig(BaseModel):
    plugin: str = "usb_camera"
    source: str = "0"
    """Device index (USB), RTSP URL, GStreamer pipeline string, etc."""
    width: int = 1280
    height: int = 720
    fps: float = 30.0
    buffer_size: int = 1
    hw_decode: bool = True
    """Use hardware-accelerated decoder when available."""
    reconnect_on_failure: bool = True


class CameraManagerConfig(BaseModel):
    cameras: list[CameraConfig] = Field(default_factory=lambda: [CameraConfig()])
    sync_tolerance_ms: float = 50.0
    """Max timestamp difference between cameras to be considered synchronized."""


class RecordingConfig(BaseModel):
    enabled: bool = False
    output_dir: Path = Path("recordings")
    format: Literal["mp4", "mkv"] = "mp4"
    codec: str = "h264"
    fps: float = 30.0
    record_raw: bool = True
    record_hud: bool = True
    record_ros_bag: bool = False
    record_telemetry: bool = True
    telemetry_format: Literal["jsonl", "csv", "both"] = "both"


class MissionConfig(BaseModel):
    profile: str = "road_vehicles"
    """Active mission profile name (maps to configs/missions/*.yaml)."""
    auto_lock: bool = True
    lock_mode: Literal["AUTO", "NEAREST", "LARGEST", "MANUAL"] = "AUTO"
    reacquisition_enabled: bool = True
    coast_max_frames: int = 30
    """Frames to coast on UKF prediction before entering LOST."""


class APIConfig(BaseModel):
    rest_enabled: bool = True
    rest_host: str = "0.0.0.0"
    rest_port: int = 8000
    ws_enabled: bool = True
    ws_port: int = 8001
    ws_publish_rate_hz: float = 10.0
    cors_origins: list[str] = ["*"]


class DiagnosticsConfig(BaseModel):
    health_interval_s: float = 1.0
    metrics_interval_s: float = 0.5
    gpu_warn_pct: float = 85.0
    cpu_warn_pct: float = 90.0
    ram_warn_pct: float = 85.0


# ─── Root schema ─────────────────────────────────────────────────────────────

class ApexConfig(BaseModel):
    """Root configuration schema for the entire APEX-Track engine."""

    version: str = "1.0"

    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    scheduler: AdaptiveSchedulerConfig = Field(default_factory=AdaptiveSchedulerConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    camera_manager: CameraManagerConfig = Field(default_factory=CameraManagerConfig)
    recording: RecordingConfig = Field(default_factory=RecordingConfig)
    mission: MissionConfig = Field(default_factory=MissionConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    diagnostics: DiagnosticsConfig = Field(default_factory=DiagnosticsConfig)

    model_config = {"extra": "allow"}  # allow plugin-specific config keys

    @field_validator("version")
    @classmethod
    def check_version(cls, v: str) -> str:
        if not v:
            raise ValueError("Config version must not be empty")
        return v
