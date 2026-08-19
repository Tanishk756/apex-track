# APEX-Track v15.0 Master Enterprise C4ISR Defense Platform
## Comprehensive System Features, Architecture & Operational Guide

---

## 🛡️ Executive Summary

**APEX-Track v15.0 Master Enterprise Edition** is a defense-grade, ultra-low-latency Command, Control, Communications, Computers, Intelligence, Surveillance, and Reconnaissance (**C4ISR**) perception and target tracking platform. Engineered for edge autonomous vehicles (UAVs, UGVs, USVs) and stationary surveillance nodes, APEX-Track fuses multi-spectral visual perception, continuous spatial tracking, reinforcement learning policy control, and multi-cloud AI integrations into a single, high-reliability system.

---

## 🔒 1. Security & Credential Isolation Layer

- **Encrypted `.env` Credential Isolation**: All sensitive third-party API keys (Roboflow, OpenAI, OpenWeatherMap) are isolated inside a protected `.env` environment file outside version control.
- **Git & Leak Protection**: `.env`, `*.pem`, `*.key`, and credential stores are registered in `.gitignore` and `.antigravityignore` to guarantee zero source code credential exposure.
- **Log & REST API Sanitization (`SecurityManager`)**: All internal logging (`structlog`) and REST API endpoints automatically run keys through masking filters (e.g. `sk-proj-...***MASKED***` or `JeocFH...***MASKED***`).
- **Automated Security Audit Endpoint**: Real-time verification via `GET /api/v1/security/audit`.

---

## 🧠 2. Deep Dive: Core Perception & AI Subsystems

### A. Multi-Backend Neural Object Detection & WBF Ensemble
- **Supported Models**: YOLOv8-seg, RT-DETR 2XL, YOLO11 (AGPL compliant), and Roboflow Universe custom weights.
- **Weighted Boxes Fusion (WBF)**: Blends bounding box predictions from multiple concurrent neural detectors using spatial overlap consensus ($IoU \ge 0.55$).
- **Optical Pre-Processing Enhancer**: Adaptive CLAHE (Contrast Limited Adaptive Histogram Equalization) combined with unsharp sharpening filters for all-weather clarity.

### B. Multi-Spectral Thermal IR Vision Shaders
- **Supported Modes**:
  1. `EO`: Electro-Optical Daylight Mode
  2. `FLIR_IRONBOW`: Ironbow Pseudo-Color Thermal Palette
  3. `FLIR_WHITE_HOT`: Standard White-Hot FLIR Infrared Palette
  4. `FLIR_BLACK_HOT`: Black-Hot Inverted Infrared Palette
  5. `NVG_GREEN`: Night Vision Goggles Tactical Phosphor Green Shader

### C. Persistent Target Tracking Core
- **Algorithms**: BoT-SORT & ByteTrack integration.
- **10D Unscented Kalman Filter (UKF)**: Models coordinated turn dynamics $[x, y, z, v_x, v_y, v_z, a_x, a_y, \psi, \dot{\psi}]$ for high-speed evasive target kinematics (60 – 120+ km/h).
- **Camera Motion Compensation (CMC)**: Optical flow matrix adjustment to maintain track locks during camera panning/slew.

### D. REMIND Memory Visual Re-Identification (Re-ID)
- **Fused Spatial-Visual Kinematic Coherence Matching**: Extracts visual feature embeddings and fuses them with velocity vector continuity.
- **Exponential Moving Average (EMA) Memory Buffer**: Maintains persistent Target UIDs across temporary line-of-sight occlusions and thermal flares.

### E. Dueling Double Deep Q-Network (D3QN) RL Policy Agent
- **Autonomous Sector Scheduling**: Dynamically shifts detection frame rates and sector focus based on target threat density and GPU load.
- **Prioritized Experience Replay (PER)**: Accelerates reinforcement learning convergence on high-priority threat events.

### F. OpenAI GPT-4o Tactical AI Copilot RAG Agent
- **Retrieval-Augmented Generation (RAG)**: Ingests live telemetry, target track histories, and system capabilities to feed OpenAI `gpt-4o-mini`.
- **ReAct Reasoning Loop**: Executes tool calling (`query_history_db`, `get_active_telemetry`, `calculate_intercept`) to deliver instant battlefield intelligence summaries.

### G. Roboflow Universe Dynamic Weight Engine
- **One-Click Model Downloads**: Hot-swaps active detector model weights directly from Roboflow Universe workspaces.
- **Hosted Inference Fallback**: Leverages Roboflow Cloud API for high-precision offloaded inference when required.

### H. OpenWeatherMap Atmospheric Telemetry Engine
- **Meteorological Attenuation**: Automatically computes optical attenuation factors to adjust CLAHE contrast gain during fog, rain, or haze.
- **Crosswind Vector Compensation**: Ingests live wind speed ($m/s$) and direction ($\text{deg}$) to refine trajectory predictions.

### I. STANAG 4609 KLV & MAVLink v2 Telemetry Interoperability
- **STANAG 4609 Metadata**: Ingests MISB ST 0601 telemetry packets.
- **MAVLink v2 UDP**: Receives flight controller telemetry (`GLOBAL_POSITION_INT`, `ATTITUDE`) for coordinate georeferencing.

### J. Target Intercept & Time-To-Intercept (TTI) Calculator
- **Lead-Angle Calculation**: Computes lead angles in degrees ($\text{deg}$) and Time-To-Intercept ($\text{seconds}$) for anti-drone and countermeasure engagement.

---

## 📊 3. Interactive Web Dashboards

### Main Tactical C4ISR Dashboard (`/`)
- **HUD Video Stream**: Live MJPEG / WebRTC feed with dynamic bounding boxes, target classification labels, and velocity vectors.
- **DEFCON Threat Matrix**: Dynamic alert badges (DEFCON 5 Normal to DEFCON 1 Alpha Critical).
- **PPI Radar Sweep**: 2D sector radar showing target range, bearing, and heading vectors.
- **Interactive AI Copilot Console**: Voice-assisted AI chat prompt interface with instant action chips (`AUDIT`, `THREATS`, `INTERCEPT`, `REMIND`).
- **Live Weather Badge**: Top header weather telemetry widget (`🌤️ CLEAR | 24.5°C | 3.5m/s`).

### Admin Control Console (`/admin`)
- **Real-Time System Diagnostics**: Live CPU usage %, RAM %, FPS, and inference latency graphs powered by `psutil`.
- **Dynamic Detector Tuning**: Live sliders to adjust confidence threshold ($0.05 - 0.95$), inference resolution ($320 - 1280$), and target class filtering.
- **Roboflow Model Selector**: Dropdown to switch model weights on the fly.
- **Telemetry Data Exporter**: One-click SQL-to-CSV target track exporter for post-mission analysis.

---

## 📡 4. Master REST API Endpoint Catalog

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/` | `GET` | Main Tactical C4ISR Dashboard |
| `/admin` | `GET` | Admin Control Console |
| `/api/v1/health` | `GET` | System health check & uptime status |
| `/api/v1/targets` | `GET` | Active target tracks registry |
| `/api/v1/video_feed` | `GET` | MJPEG video stream feed |
| `/api/v1/security/audit` | `GET` | Sanitized security audit & masked key report |
| `/api/v1/roboflow/status` | `GET` | Roboflow API client status & model info |
| `/api/v1/roboflow/model` | `POST` | Switch active detector model weights |
| `/api/v1/weather/telemetry` | `GET` | Live OpenWeatherMap atmospheric metrics |
| `/api/v1/vision/mode` | `POST` | Change vision shader (`EO`, `FLIR_IRONBOW`, `NVG_GREEN`) |
| `/api/v1/admin/system_stats` | `GET` | Real-time CPU, RAM, FPS, and latency metrics |
| `/api/v1/admin/detector_config` | `POST` | Hot-reload detector confidence & resolution settings |
| `/api/v1/admin/export_csv` | `GET` | Export historical track logs as a CSV file |
| `/api/v1/copilot/query` | `POST` | Submit query to Tactical AI Copilot RAG Agent |

---

## 🚀 5. Deployment & Execution Instructions

### A. Quick Start Server Launch
To start the APEX-Track server:
```bash
python3 -m apex.cli serve --host 0.0.0.0 --port 8000
```
- Access Main Dashboard: **[http://localhost:8000](http://localhost:8000)**
- Access Admin Console: **[http://localhost:8000/admin](http://localhost:8000/admin)**

### B. Running Automated Unit Verification
To execute the complete 137-test automated verification suite:
```bash
python3 -m pytest tests/unit/ -v
```

---
*APEX-Track v15.0 Master Enterprise Edition — Production Defense Perception System.*
