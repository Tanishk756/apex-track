"""
Health Monitor
==============
Periodically samples system health (GPU, CPU, RAM, temperatures)
and publishes snapshots on Ch.HEALTH.
Also monitors plugin health and emits events when thresholds are crossed.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

import structlog

from apex.engine.bus.channels import Ch

log = structlog.get_logger(__name__)


@dataclass
class HealthSnapshot:
    """A point-in-time system health snapshot."""
    timestamp: float = field(default_factory=time.time)

    # CPU
    cpu_pct: float = 0.0
    cpu_temp_c: Optional[float] = None

    # RAM
    ram_total_mb: float = 0.0
    ram_used_mb: float = 0.0
    ram_pct: float = 0.0

    # GPU (None if no GPU)
    gpu_pct: Optional[float] = None
    gpu_mem_used_mb: Optional[float] = None
    gpu_mem_total_mb: Optional[float] = None
    gpu_temp_c: Optional[float] = None

    # Disk
    disk_free_gb: Optional[float] = None

    # Plugin health
    plugin_statuses: dict[str, str] = field(default_factory=dict)

    @property
    def gpu_mem_pct(self) -> Optional[float]:
        if self.gpu_mem_used_mb and self.gpu_mem_total_mb:
            return 100.0 * self.gpu_mem_used_mb / self.gpu_mem_total_mb
        return None


def _sample_cpu() -> tuple[float, Optional[float]]:
    try:
        import psutil
        cpu_pct = psutil.cpu_percent(interval=None)
        temps = psutil.sensors_temperatures() if hasattr(psutil, "sensors_temperatures") else {}
        temp = None
        for label in ("coretemp", "cpu_thermal", "k10temp"):
            entries = temps.get(label, [])
            if entries:
                temp = entries[0].current
                break
        return cpu_pct, temp
    except Exception:
        return 0.0, None


def _sample_ram() -> tuple[float, float, float]:
    try:
        import psutil
        vm = psutil.virtual_memory()
        total = vm.total / (1024 * 1024)
        used  = vm.used  / (1024 * 1024)
        return total, used, vm.percent
    except Exception:
        return 0.0, 0.0, 0.0


def _sample_gpu() -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Returns (gpu_pct, mem_used_mb, mem_total_mb, temp_c)."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util   = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem    = pynvml.nvmlDeviceGetMemoryInfo(handle)
        try:
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        except Exception:
            temp = None
        pynvml.nvmlShutdown()
        return (
            float(util.gpu),
            mem.used  / (1024 * 1024),
            mem.total / (1024 * 1024),
            float(temp) if temp is not None else None,
        )
    except Exception:
        pass
    # Fallback: try GPUtil
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            g = gpus[0]
            return g.load * 100, g.memoryUsed, g.memoryTotal, g.temperature
    except Exception:
        pass
    return None, None, None, None


def _sample_disk() -> Optional[float]:
    try:
        import psutil
        usage = psutil.disk_usage("/")
        return usage.free / (1024 ** 3)
    except Exception:
        return None


class HealthMonitor:
    """
    Periodic health sampler. Publishes HealthSnapshot on Ch.HEALTH.
    Emits GPU_OVERLOADED / CPU_OVERLOADED / MEMORY_PRESSURE events
    when configurable thresholds are crossed.
    """

    def __init__(
        self,
        bus=None,
        event_engine=None,
        plugin_registry=None,
        interval_s: float = 1.0,
        gpu_warn_pct: float = 85.0,
        cpu_warn_pct: float = 90.0,
        ram_warn_pct: float = 85.0,
    ) -> None:
        self._bus = bus
        self._events = event_engine
        self._registry = plugin_registry
        self._interval = interval_s
        self._gpu_warn = gpu_warn_pct
        self._cpu_warn = cpu_warn_pct
        self._ram_warn = ram_warn_pct
        self._last: Optional[HealthSnapshot] = None
        self._running = False

    async def start(self) -> None:
        self._running = True
        log.info("health_monitor_started", interval_s=self._interval)
        while self._running:
            try:
                snapshot = self._sample()
                self._last = snapshot
                if self._bus:
                    await self._bus.publish(Ch.HEALTH, snapshot)
                await self._check_thresholds(snapshot)
            except Exception as exc:
                log.warning("health_sample_error", error=str(exc))
            await asyncio.sleep(self._interval)

    def stop(self) -> None:
        self._running = False

    def _sample(self) -> HealthSnapshot:
        cpu_pct, cpu_temp = _sample_cpu()
        ram_total, ram_used, ram_pct = _sample_ram()
        gpu_pct, gpu_mem_used, gpu_mem_total, gpu_temp = _sample_gpu()
        disk_free = _sample_disk()

        plugin_statuses: dict[str, str] = {}
        if self._registry:
            for name, health in self._registry.health_report().items():
                plugin_statuses[name] = health.status.name

        return HealthSnapshot(
            cpu_pct=cpu_pct,
            cpu_temp_c=cpu_temp,
            ram_total_mb=ram_total,
            ram_used_mb=ram_used,
            ram_pct=ram_pct,
            gpu_pct=gpu_pct,
            gpu_mem_used_mb=gpu_mem_used,
            gpu_mem_total_mb=gpu_mem_total,
            gpu_temp_c=gpu_temp,
            disk_free_gb=disk_free,
            plugin_statuses=plugin_statuses,
        )

    async def _check_thresholds(self, s: HealthSnapshot) -> None:
        if self._events is None:
            return
        from apex.engine.contracts.event import ApexEvent, EventSeverity, EventType
        if s.gpu_pct and s.gpu_pct > self._gpu_warn:
            await self._events.emit(ApexEvent(
                EventType.GPU_OVERLOADED, source="health_monitor",
                severity=EventSeverity.WARNING,
                payload={"gpu_pct": s.gpu_pct, "threshold": self._gpu_warn},
            ))
        if s.cpu_pct > self._cpu_warn:
            await self._events.emit(ApexEvent(
                EventType.CPU_OVERLOADED, source="health_monitor",
                severity=EventSeverity.WARNING,
                payload={"cpu_pct": s.cpu_pct},
            ))
        if s.ram_pct > self._ram_warn:
            await self._events.emit(ApexEvent(
                EventType.MEMORY_PRESSURE, source="health_monitor",
                severity=EventSeverity.WARNING,
                payload={"ram_pct": s.ram_pct},
            ))

    @property
    def last_snapshot(self) -> Optional[HealthSnapshot]:
        return self._last
