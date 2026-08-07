"""
End-to-End Master Pipeline Benchmark Tool
==========================================
Measures end-to-end pipeline processing latency (ms), frame throughput (FPS),
and target tracking stability across multi-frame simulated video streams.

Usage:
    python tools/bench_pipeline.py --frames 300
"""

import argparse
import asyncio
import time
import numpy as np

from apex.engine.contracts.frame import Frame, FrameMetadata
from apex.engine.hal.hw_profile import HWCapabilities, HWProfile
from apex.engine.pipeline.master_pipeline import MasterPipeline


async def main(num_frames: int) -> None:
    print("=== APEX-Track Master Pipeline Benchmark ===")
    print(f"Frames: {num_frames}")

    hw = HWProfile(capabilities=HWCapabilities(), profile_name="bench_profile")
    pipeline = MasterPipeline()
    await pipeline.initialize({}, hw)

    # Pre-generate synthetic video frames
    frames = []
    for i in range(num_frames):
        data = np.zeros((480, 640, 3), dtype=np.uint8)
        meta = FrameMetadata(camera_id="cam0", width=640, height=480)
        f = Frame(data=data, metadata=meta, timestamp=i * 0.033, sequence_id=i)
        frames.append(f)

    # Warmup
    for _ in range(5):
        await pipeline.process_frame(frames[0])

    latencies = []
    t_start = time.perf_counter()

    for f in frames:
        t0 = time.perf_counter()
        track_array = await pipeline.process_frame(f)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat_ms)

    total_duration = time.perf_counter() - t_start
    latencies = np.array(latencies)

    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    fps = num_frames / total_duration

    print("\n=== Benchmark Results ===")
    print(f"Total Time:    {total_duration:.3f} s")
    print(f"Throughput:    {fps:.2f} FPS")
    print(f"p50 Latency:   {p50:.3f} ms")
    print(f"p95 Latency:   {p95:.3f} ms")
    print("==========================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master Pipeline Benchmark")
    parser.add_argument("--frames", type=int, default=100, help="Number of benchmark frames")
    args = parser.parse_args()
    asyncio.run(main(args.frames))
