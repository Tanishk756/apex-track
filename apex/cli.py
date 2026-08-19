"""
APEX-Track Unified Command Line Interface (CLI)
==============================================
Provides production command utilities for APEX-Track:
- serve: Launch FastAPI C4ISR REST & WebSocket Operational Command Console
- bench: Execute sub-millisecond pipeline latency and throughput benchmarks
- ros2: Launch ROS2 target tracker node adapter
- convert: Download or convert neural detection models to ONNX/TensorRT
- info: Display system hardware profile and execution providers
"""

from __future__ import annotations

import argparse
import sys
import uvicorn

from apex import __version__
from apex.engine.hal import hw_detector, hw_profile


def serve_command(args: argparse.Namespace) -> None:
    """Launch the FastAPI operational dashboard server."""
    if args.camera_source:
        import os
        os.environ["APEX_CAMERA_SOURCE"] = args.camera_source
        print(f"  Camera Source override set to: {args.camera_source}")
    if args.camera_plugin:
        import os
        os.environ["APEX_CAMERA_PLUGIN"] = args.camera_plugin
        print(f"  Camera Plugin override set to: {args.camera_plugin}")
    if args.mission:
        import os
        os.environ["APEX_MISSION_PROFILE"] = args.mission
        print(f"  Mission Profile override set to: {args.mission}")

    print(f"===========================================================")
    print(f"  APEX-Track C4ISR Tactical Command Server v{__version__}")
    print(f"  Listening on http://{args.host}:{args.port}")
    print(f"===========================================================")
    uvicorn.run("apex.api.server:app", host=args.host, port=args.port, reload=args.reload)




def bench_command(args: argparse.Namespace) -> None:
    """Run performance and latency benchmarks."""
    if args.target == "pipeline":
        from tools.bench_pipeline import run_benchmark
        run_benchmark(iterations=args.iterations)
    elif args.target == "tracker":
        from tools.bench_tracker import run_benchmark
        run_benchmark(tracker_name="bytetrack", num_frames=args.iterations)
    elif args.target == "inference":
        from tools.bench_inference import run_benchmark
        run_benchmark(model_path=args.model, iterations=args.iterations, precision=args.precision)



def ros2_command(args: argparse.Namespace) -> None:
    """Launch ROS2 Node Adapter."""
    try:
        import rclpy
        from ros2.nodes.detection_node import ROS2DetectionNodeAdapter
        print("[INFO] ROS2 rclpy environment detected. Initializing ROS2 Perception Node...")
        adapter = ROS2DetectionNodeAdapter()
        print("[SUCCESS] ROS2 Node Adapter active. Subscribed to /camera/image_raw.")
        _ = adapter
    except ImportError:
        print("[NOTICE] Standard rclpy bindings not active in current Python interpreter.")
        print("[INFO] Initializing ROS2 Standalone Adapter Mode...")
        from ros2.nodes.detection_node import ROS2DetectionNodeAdapter
        adapter = ROS2DetectionNodeAdapter()
        print("[SUCCESS] ROS2 Standalone Perception Adapter initialized successfully.")
        _ = adapter



def train_command(args: argparse.Namespace) -> None:
    """Execute model fine-tuning."""
    from tools.dataset_manager import DatasetManager
    from tools.train_detector import TacticalModelTrainer

    ds_mgr = DatasetManager.instance()
    yaml_path = args.data
    if not yaml_path:
        ds_res = ds_mgr.synthesize_dataset(dataset_name=args.dataset, num_samples=args.samples)
        yaml_path = ds_res["data_yaml"]
        print(f"[SUCCESS] Synthesized dataset '{args.dataset}' ({args.samples} samples): {yaml_path}")

    trainer = TacticalModelTrainer()
    print(f"===========================================================")
    print(f"  APEX-Track Neural Model Fine-Tuning Engine")
    print(f"  Dataset YAML:  {yaml_path}")
    print(f"  Backbone:      {args.backbone}")
    print(f"  Epochs:        {args.epochs} | Batch Size: {args.batch_size}")
    print(f"  Device:        {args.device}")
    print(f"===========================================================")

    res = trainer.train_model(
        dataset_yaml=yaml_path,
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        device=args.device,
        output_name=args.output,
    )
    print(f"[RESULT] Training {res.get('status')}: mAP50 = {res.get('mAP50')} | Saved to: {res.get('weights_saved')}")


def dataset_command(args: argparse.Namespace) -> None:
    """Manage and synthesize datasets."""
    from tools.dataset_manager import DatasetManager
    ds_mgr = DatasetManager.instance()

    if args.action == "list":
        catalog = ds_mgr.list_datasets()
        print("===========================================================")
        print("             APEX-Track Tactical Dataset Catalog           ")
        print("===========================================================")
        for ds in catalog:
            inst = "INSTALLED" if ds["installed"] else "AVAILABLE"
            print(f" - {ds['name']:<22} [{inst}] | Samples: {ds['sample_count']} | Classes: {', '.join(ds['classes'])}")
            print(f"   {ds['description']}")
        print("===========================================================")
    elif args.action == "synthesize":
        res = ds_mgr.synthesize_dataset(dataset_name=args.name, num_samples=args.count)
        print(f"[SUCCESS] Synthesized dataset '{args.name}' with {args.count} samples.")
        print(f" Data YAML: {res['data_yaml']}")


def info_command(args: argparse.Namespace) -> None:
    """Print hardware profile and environment state."""
    hw = hw_detector.detect()
    print("===========================================================")
    print("           APEX-Track Hardware & Perception Profile        ")
    print("===========================================================")
    print(f" Profile Name:        {hw.profile_name}")
    print(f" CPU Cores:           {hw.capabilities.cpu_cores}")
    print(f" System RAM:          {hw.capabilities.ram_total_mb / 1024.0:.1f} GB")
    print(f" CUDA Acceleration:   {'ENABLED' if hw.is_gpu else 'DISABLED'}")
    print(f" GPU Device:          {hw.capabilities.gpu_name or 'N/A'}")
    print(f" Rec. Precision:      {hw.capabilities.recommended_fp_precision}")
    print(f" Active Flags:        {', '.join([c.name for c in hw_profile.Capability if c != hw_profile.Capability.NONE and hw.has(c)])}")
    print("===========================================================")



def main() -> None:
    parser = argparse.ArgumentParser(
        prog="apex-track",
        description="APEX-Track Industrial AI Detection & Target Tracking Platform",
    )
    parser.add_value_name = "COMMAND"
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # serve command
    serve_parser = subparsers.add_parser("serve", help="Start FastAPI Operational Server & Dashboard")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")
    serve_parser.add_argument("--camera-source", type=str, default=None, help="Camera video stream source (URL or device index, e.g., http://192.168.1.50:4747/video or 0)")
    serve_parser.add_argument("--camera-plugin", type=str, default=None, choices=["usb_camera", "rtsp_camera", "file_camera", "gstreamer_source"], help="Camera plugin type")
    serve_parser.add_argument("--mission", type=str, default=None, help="Mission profile name (e.g. drone_tracking, road_vehicles, battlefield)")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload for dev")
    serve_parser.set_defaults(func=serve_command)

    # train command
    train_parser = subparsers.add_parser("train", help="Fine-tune neural object detector")
    train_parser.add_argument("--dataset", type=str, default="aod_4", choices=["drone_uav", "aod_4", "battlefield_vehicles", "thermal_ir", "coco_defense"], help="Preset dataset name")
    train_parser.add_argument("--data", type=str, default=None, help="Custom dataset data.yaml path")
    train_parser.add_argument("--backbone", type=str, default="yolov8s.pt", help="Pre-trained backbone weights file")
    train_parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    train_parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution")
    train_parser.add_argument("--device", type=str, default="0", help="Device ID (0 for cuda:0 or cpu)")
    train_parser.add_argument("--samples", type=int, default=100, help="Number of synthetic samples if generating")
    train_parser.add_argument("--output", type=str, default="apex_tactical_v12.pt", help="Output checkpoint file name")
    train_parser.set_defaults(func=train_command)

    # dataset command
    dataset_parser = subparsers.add_parser("dataset", help="Manage and synthesize datasets")
    dataset_parser.add_argument("action", choices=["list", "synthesize"], help="Dataset action")
    dataset_parser.add_argument("--name", type=str, default="drone_uav", help="Dataset name")
    dataset_parser.add_argument("--count", type=int, default=100, help="Number of synthetic samples to render")
    dataset_parser.set_defaults(func=dataset_command)

    # bench command
    bench_parser = subparsers.add_parser("bench", help="Run Latency and FPS Benchmarks")
    bench_parser.add_argument(
        "--target", type=str, choices=["pipeline", "tracker", "inference"], default="pipeline",
        help="Benchmark target (pipeline, tracker, inference)"
    )
    bench_parser.add_argument("--iterations", type=int, default=100, help="Number of benchmark iterations")
    bench_parser.add_argument("--model", type=str, default="dummy.onnx", help="Model path for inference benchmark")
    bench_parser.add_argument("--precision", type=str, default="fp16", help="Precision (fp32, fp16, int8)")
    bench_parser.set_defaults(func=bench_command)

    # ros2 command
    ros2_parser = subparsers.add_parser("ros2", help="Launch ROS2 Perception Adapter Node")
    ros2_parser.set_defaults(func=ros2_command)

    # info command
    info_parser = subparsers.add_parser("info", help="Display hardware profile and system info")
    info_parser.set_defaults(func=info_command)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

