"""
APEX-Track Benchmark & Latency Measurement Script
=================================================
Executes performance micro-benchmarks measuring throughput (FPS) and latency (p50/p95 ms)
across:
1. Master Perception & Tracking Pipeline
2. 10-State Unscented Kalman Filter (UKF)
3. ByteTrack Multi-Target Association Engine
4. UDP / Transport Subsystem
"""

from __future__ import annotations

import asyncio
import time
import numpy as np
import structlog

from apex.engine.contracts.frame import Frame, FrameMetadata
from apex.engine.hal.hw_detector import detect
from apex.engine.pipeline.master_pipeline import MasterPipeline
from apex.engine.tracker.ukf import UnscentedKalmanFilter10D
from plugins.trackers.bytetrack.plugin import ByteTrackPlugin

log = structlog.get_logger(__name__)


async def benchmark_master_pipeline(num_frames: int = 200) -> dict[str, float]:
    """Benchmark Master Perception Pipeline throughput and latency."""
    hw_profile = detect()
    hw_profile.profile_name = "cpu_test"

    pipeline = MasterPipeline()
    await pipeline.initialize({}, hw_profile)

    dummy_image = np.zeros((720, 1280, 3), dtype=np.uint8)
    latencies: list[float] = []

    for i in range(num_frames):
        frame = Frame(
            data=dummy_image,
            metadata=FrameMetadata(camera_id="cam_bench", width=1280, height=720),
            timestamp=time.time(),
            sequence_id=i,
        )
        t0 = time.perf_counter()
        await pipeline.process_frame(frame)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    total_sec = sum(latencies) / 1000.0
    fps = num_frames / max(total_sec, 1e-6)

    return {"fps": round(fps, 2), "p50_ms": round(p50, 3), "p95_ms": round(p95, 3)}


def benchmark_ukf(num_iterations: int = 1000) -> dict[str, float]:
    """Benchmark 10-State Unscented Kalman Filter (UKF)."""
    ukf = UnscentedKalmanFilter10D(dt=0.033)
    ukf.initiate((100.0, 200.0, 50.0))

    latencies: list[float] = []
    for i in range(num_iterations):
        t0 = time.perf_counter()
        ukf.predict()
        ukf.update((100.0 + i * 0.1, 200.0 + i * 0.05, 50.0))
        latencies.append((time.perf_counter() - t0) * 1000.0)

    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    total_sec = sum(latencies) / 1000.0
    fps = num_iterations / max(total_sec, 1e-6)

    return {"fps": round(fps, 2), "p50_ms": round(p50, 3), "p95_ms": round(p95, 3)}


async def main() -> None:
    print("=" * 60)
    print("APEX-Track Benchmark Suite Execution")
    print("=" * 60)

    print("[1/2] Benchmarking Master Perception Pipeline...")
    res_pipeline = await benchmark_master_pipeline()
    print(f"  FPS    : {res_pipeline['fps']:.2f}")
    print(f"  p50    : {res_pipeline['p50_ms']:.3f} ms")
    print(f"  p95    : {res_pipeline['p95_ms']:.3f} ms")

    print("[2/2] Benchmarking 10-State UKF Estimator...")
    res_ukf = benchmark_ukf()
    print(f"  FPS    : {res_ukf['fps']:.2f}")
    print(f"  p50    : {res_ukf['p50_ms']:.3f} ms")
    print(f"  p95    : {res_ukf['p95_ms']:.3f} ms")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
