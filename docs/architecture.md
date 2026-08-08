# APEX-Track Architecture Specification

## System Overview
APEX-Track is an industrial-grade, zero-latency AI perception and target tracking platform designed for UAV and autonomous defense operations. It provides high-speed multi-spectral object detection, velocity-adaptive target tracking (ByteTrack / BoT-SORT), spatial geofencing, and telemetry synchronization over ROS2 and FastAPI/WebSockets.

```
                  +-----------------------------------+
                  |  Multi-Spectral Frame Ingestion   |
                  |     (File / RTSP / USB Camera)    |
                  +-----------------+-----------------+
                                    |
                                    v
                  +-----------------+-----------------+
                  | Hardware Abstraction Layer (HAL) |
                  |    (HWProfile / Capabilities)     |
                  +-----------------+-----------------+
                                    |
                                    v
                  +-----------------+-----------------+
                  |   Neural Detection & Ensemble    |
                  |    (RT-DETR / RTMDet / YOLO11)    |
                  +-----------------+-----------------+
                                    |
                                    v
                  +-----------------+-----------------+
                  |    Adaptive Multi-Target Tracker  |
                  |      (ByteTrack / BoT-SORT CMC)   |
                  +-----------------+-----------------+
                                    |
                                    v
                  +-----------------+-----------------+
                  |    Spatial Engine & Target DB     |
                  |  (3D Projection / Geofencing)    |
                  +--------+----------------+---------+
                           |                |
                           v                v
         +-----------------+--+   +---------+-----------------+
         | ROS2 Node Adapter  |   | FastAPI Operational Server|
         | (apex_track_ros)   |   | (C4ISR Glassmorphic UI)   |
         +--------------------+   +---------------------------+
```

## Key Subsystems

### 1. Hardware Abstraction Layer (HAL)
- **Capability Model**: Feature flags query capabilities (e.g., `Capability.CUDA`, `Capability.TENSORRT`, `Capability.FP16`), decoupling code from specific GPU names or SBC platforms.
- **Hardware Profile**: Automatically detects available compute backends, thread counts, RAM limits, and optimal inference precision.

### 2. Neural Detection & Inference Engine
- **Pluggable Architecture**: Modular loading of open-source neural detectors (RT-DETR, RTMDet, YOLOv11).
- **Ensemble Detector**: Weighted Non-Maximum Suppression (NMS) for multi-model decision fusion.
- **Engine Factory**: Dynamic fallback strategy across TensorRT, ONNX Runtime (CUDA/CPU), and PyTorch.

### 3. Adaptive Multi-Target Tracker
- **ByteTrack**: High-throughput target identification and motion association.
- **BoT-SORT CMC**: Camera Motion Compensation for high-speed pitch/yaw UAV maneuvers.
- **Constant Acceleration Kalman Filter**: 9-state state vector tracking position, velocity, and acceleration.

### 4. Spatial Engine & Geofencing
- **3D World Projection**: Maps 2D pixel coordinates to 3D spatial points based on sensor telemetry.
- **Geofence Engine**: Real-time evaluation of perimeter breaches, target arrival/departure events, and threat score updates.
- **Target Registry**: In-memory database with history tracking and state persistence.

### 5. API & C4ISR Command Console
- **FastAPI Backend**: Operational REST API endpoints (`/api/v1/health`, `/api/v1/targets`, `/api/v1/missions/switch`).
- **WebSocket Streaming**: 30 Hz live target HUD position and spatial telemetry feed.
- **Glassmorphic Tactical Dashboard**: High-contrast UI featuring live video reticle, radar sweep, target registry, and mission selector.
