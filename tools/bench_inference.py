"""
Inference Benchmark Tool
========================
Measures latency (p50, p95, p99), throughput (FPS), and memory consumption of models across inference backends.

Usage:
    python tools/bench_inference.py --model models/rtdetr.onnx --iterations 200
"""

import argparse
import time
import numpy as np

from apex.engine.hal import hw_detector
from apex.engine.inference.factory import EngineFactory


def run_benchmark(model_path: str, iterations: int = 100, batch_size: int = 1, precision: str = "fp16") -> None:
    print(f"=== APEX-Track Inference Benchmark ===")
    print(f"Model: {model_path}")
    print(f"Iterations: {iterations} | Batch size: {batch_size} | Precision: {precision}")

    hw = hw_detector.detect()
    print(f"Hardware Profile: {hw.profile_name} (GPU: {hw.capabilities.gpu_name})")

    engine = EngineFactory.create(model_path, hw, preferred_precision=precision)

    if not engine.is_loaded:
        print("Engine failed to load. Exiting.")
        return

    # Warmup
    engine.warmup(warmup_runs=10)

    # Benchmark loop
    input_data = np.random.randn(batch_size, 3, 640, 640).astype(np.float32)
    latencies = []

    print("\nRunning benchmark iterations...")
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = engine.infer(input_data)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)

    latencies = np.array(latencies)
    avg_latency = np.mean(latencies)
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0

    print("\n=== Benchmark Results ===")
    print(f"FPS:          {fps:.2f} frames/sec")
    print(f"Avg Latency:  {avg_latency:.2f} ms")
    print(f"p50 Latency:  {p50:.2f} ms")
    print(f"p95 Latency:  {p95:.2f} ms")
    print(f"p99 Latency:  {p99:.2f} ms")
    print("==========================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="APEX-Track Inference Benchmark")
    parser.add_argument("--model", type=str, default="dummy.onnx", help="Path to model file")
    parser.add_argument("--iterations", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--precision", type=str, default="fp16", help="Precision (fp32, fp16, int8)")
    args = parser.parse_args()
    run_benchmark(args.model, iterations=args.iterations, precision=args.precision)
