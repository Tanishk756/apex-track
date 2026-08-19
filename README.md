# 🛡️ APEX-Track v15.0 Master Enterprise C4ISR Platform

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![ROS2 Compatible](https://img.shields.io/badge/ROS2-Humble%2FSDK-orange.svg)](https://docs.ros.org/)
[![Defense Standard](https://img.shields.io/badge/STANAG-4609%20KLV-green.svg)](https://www.nso.nato.int/)
[![License](https://img.shields.io/badge/license-MIT-purple.svg)](LICENSE)

**APEX-Track v15.0 Master Enterprise Edition** is a defense-grade, ultra-low-latency Command, Control, Communications, Computers, Intelligence, Surveillance, and Reconnaissance (**C4ISR**) perception and persistent target tracking platform. 

Engineered for edge autonomous systems (UAVs, UGVs, USVs) and fixed tactical observation posts, APEX-Track seamlessly fuses multi-spectral visual perception, continuous 3D spatial analytics, autonomous reinforcement learning policy control, and tactical AI copilot reasoning into a zero-downtime, production-ready operational suite.

---

## 🌟 Key Platform Capabilities

### 🧠 1. Multi-Model Perception & WBF Consensus Ensemble
- **Multi-Backend Inference**: Dynamic engine abstraction supporting YOLOv8-seg, RT-DETR 2XL, YOLO11, and Roboflow Universe custom fine-tuned weights.
- **Weighted Boxes Fusion (WBF)**: Blends bounding box predictions from concurrent neural models using spatial overlap consensus ($IoU \ge 0.55$) for maximum acquisition precision.
- **Optical Pre-Processing Enhancer**: Adaptive CLAHE (Contrast Limited Adaptive Histogram Equalization) combined with unsharp sharpening filters for all-weather clarity.

### 🌐 2. Multi-Spectral Thermal IR Shaders
Real-time GPU/CPU shader pipeline supporting instantaneous thermal palette switching:
1. `EO`: Electro-Optical High-Definition Daylight Mode
2. `FLIR_IRONBOW`: Ironbow Thermal Infrared Palette
3. `FLIR_WHITE_HOT`: Standard White-Hot FLIR Palette
4. `FLIR_BLACK_HOT`: Black-Hot Inverted Infrared Palette
5. `NVG_GREEN`: Night Vision Goggles Tactical Phosphor Green Shader

### 🎯 3. Kinematic Tracking & 10D Unscented Kalman Filtering (UKF)
- **Tracking Core**: Integrated BoT-SORT & ByteTrack algorithms with Camera Motion Compensation (CMC).
- **10D UKF Dynamics**: State vector $[x, y, z, v_x, v_y, v_z, a_x, a_y, \psi, \dot{\psi}]$ models high-speed evasive maneuvering targets (60–120+ km/h) across coordinate systems.
- **REMIND Visual Re-ID**: Fuses visual feature embeddings with kinematic motion vectors to maintain persistent target IDs across line-of-sight occlusions.

### 🤖 4. Tactical ReAct AI Copilot (OpenAI GPT-4o RAG Agent)
- **Retrieval-Augmented Generation**: Ingests live telemetry, target track histories, and system health status.
- **ReAct Autonomous Tool Calling**: Executes dynamic tools (`query_history_db`, `get_active_telemetry`, `calculate_intercept`) to deliver instant tactical summaries.

### ☁️ 5. Third-Party Defense Integrations & Telemetry
- **Roboflow Universe Engine**: One-click model switching and cloud-offloaded hosted inference fallback.
- **OpenWeatherMap Integration**: Calculates meteorological atmospheric attenuation and crosswind vector compensation.
- **STANAG 4609 & MAVLink v2**: Native parsing of MISB ST 0601 KLV metadata and MAVLink UDP streams (`GLOBAL_POSITION_INT`, `ATTITUDE`).

### 🎮 6. Reinforcement Learning Policy Engine (D3QN)
- **Dueling Double Deep Q-Network (D3QN)**: Dynamically adjusts camera frame rates, resolution, and sector focus based on target threat density and hardware compute budget.
- **Prioritized Experience Replay (PER)**: Accelerates policy adaptation during high-priority threat events.

---

## 🔒 Security & Credential Isolation Architecture

APEX-Track implements strict zero-leak credential isolation:
- **Environment Isolation**: Third-party API keys (`ROBOFLOW_API_KEY`, `OPENAI_API_KEY`, `OPENWEATHER_API_KEY`) are stored strictly inside `.env` files (never committed to Git).
- **Log & REST API Sanitization**: The internal `SecurityManager` automatically masks keys in logs and API outputs (`sk-proj-...***MASKED***`).
- **Security Audit Endpoint**: Real-time posture verification via `GET /api/v1/security/audit`.

---

## 📁 Repository Structure

```text
apex-track/
├── apex/
│   ├── api/                  # FastAPI Web Server, REST API & Web Dashboard Static Assets
│   │   ├── static/           # Tactical HUD (index.html), Admin Dashboard (admin.html), CSS & JS
│   │   └── server.py         # Main C4ISR Command Center API routes
│   ├── cli.py                # Command Line Interface launcher
│   └── engine/
│       ├── agent/            # ReAct Tactical AI Copilot RAG Agent
│       ├── bus/              # Async Message Bus & Multi-Channel Publisher
│       ├── camera/           # Universal Camera HAL & Frame Synchronizer
│       ├── config/           # Schema validation & SecurityManager key masking
│       ├── db/               # Target Track SQLite Database & History Logger
│       ├── detector/         # Base Detector, WBF Ensemble & Roboflow Integrator
│       ├── hal/              # MAVLink v2 & STANAG 4609 Telemetry Decoders
│       ├── mission/          # PID Gimbal Pursuit & Countermeasures Engine
│       ├── pipeline/         # Master Pipeline Processor & Video Recorder
│       ├── rl/               # D3QN Reinforcement Learning Agent & PER Buffer
│       ├── spatial/          # 3D Geolocation, Geofencing & Trajectory Predictor
│       ├── telemetry/        # Weather Engine & MAVLink Receiver
│       └── tracker/          # BoT-SORT, ByteTrack, 10D UKF & REMIND Visual Re-ID
├── configs/                  # System YAML Configuration files
├── docs/                     # System Architecture & Benchmark Documentation
├── plugins/                  # Modular Camera, Detector & Tracker Plugin Hub
├── tests/unit/               # Complete 137-test Automated Verification Suite
└── tools/                    # Benchmarking, Fine-Tuning & Dataset Harvester Tools
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- **Python**: 3.10 or higher
- **System Packages**: `ffmpeg`, `libv4l-dev` (Linux)

### 2. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/Tanishk756/apex-track.git
cd apex-track
pip install -r requirements.txt
```

### 3. Environment Configuration (.env)
Create a `.env` file in the root directory to supply optional third-party API credentials:
```env
ROBOFLOW_API_KEY=your_roboflow_private_or_publishable_key
OPENAI_API_KEY=sk-proj-your_openai_api_key
OPENWEATHER_API_KEY=your_openweather_api_key
```

### 4. Launching the Tactical Command Server
Start the APEX-Track server:
```bash
python3 -m apex.cli serve --host 0.0.0.0 --port 8000
```

- **Tactical HUD Dashboard**: [http://localhost:8000](http://localhost:8000)
- **Admin Control Console**: [http://localhost:8000/admin](http://localhost:8000/admin)
- **API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🧪 Automated Testing & Verification

APEX-Track includes a comprehensive automated test suite covering all modules:

```bash
python3 -m pytest tests/unit/ -v
```

> **Verification Status**: 137 / 137 unit tests passing (100% pass rate).

---

## 📡 REST API Reference Summary

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/` | `GET` | Main Tactical C4ISR Interactive HUD |
| `/admin` | `GET` | Admin Diagnostics & Detector Tuning Dashboard |
| `/api/v1/health` | `GET` | System health check & component status |
| `/api/v1/targets` | `GET` | Active target tracks registry |
| `/api/v1/video_feed` | `GET` | Real-time MJPEG visual stream feed |
| `/api/v1/security/audit` | `GET` | Sanitized security audit & masked key report |
| `/api/v1/copilot/query` | `POST` | Route custom queries to Tactical AI Copilot RAG Agent |
| `/api/v1/vision/mode` | `POST` | Change vision shader mode (`EO`, `FLIR_IRONBOW`, `NVG_GREEN`) |
| `/api/v1/roboflow/model` | `POST` | Hot-swap active neural detector weights |
| `/api/v1/weather/telemetry` | `GET` | Fetch OpenWeatherMap atmospheric metrics |

---

## 👤 Author & Owner

**Tanishk Singhal**  
- **GitHub**: [@Tanishk756](https://github.com/Tanishk756)
- **Email**: [tanishksinghal6285@gmail.com](mailto:tanishksinghal6285@gmail.com)

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.  
Copyright (c) 2026 **Tanishk Singhal**. All rights reserved.

---
*APEX-Track v15.0 Master Enterprise Edition — Autonomous Tactical Perception & Tracking.*
