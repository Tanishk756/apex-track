"""
FastAPI Operational REST & WebSocket Command Server
===================================================
Tactical API server exposing endpoints for:
- System state & hardware profile metrics (/api/v1/health, /api/v1/status)
- Active target tracks (/api/v1/targets)
- Mission profile switching (/api/v1/missions)
- Live MJPEG Video Feed with AI Detection Overlay (/video_feed)
- Live WebSocket stream for real-time telemetry (/ws/telemetry)
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import structlog

from apex.engine.config.config_manager import ConfigManager
from apex.engine.contracts.frame import Frame, FrameMetadata
from apex.engine.hal import hw_detector
from apex.engine.pipeline.master_pipeline import MasterPipeline
from apex.engine.spatial.trajectory_predictor import TrajectoryPredictor
from apex.engine.mission.threat_matrix import ThreatMatrixEngine
from apex.engine.pipeline.thermal_fusion import ThermalFusionShader, ThermalVisionMode
from apex.engine.fusion.sensor_fusion import SensorFusionEngine
from apex.engine.spatial.swarm_defense import SwarmDefenseGrid
from apex.engine.analytics.anomaly_detector import AnomalyDetector
from apex.engine.recording.blackbox_recorder import BlackboxRecorder
from apex.engine.spatial.intercept_calculator import InterceptCalculator
from apex.engine.mission.countermeasures import CountermeasureEngine
from plugins.cameras.rtsp_camera.plugin import RTSPCameraPlugin
from plugins.cameras.usb_camera.plugin import USBCameraPlugin

log = structlog.get_logger(__name__)

app = FastAPI(
    title="APEX-Track Perception Platform API",
    version="4.0.0",
    description="Ultra-low latency defense-grade detection, thermal fusion & RF-DETR 2XL tracking REST/WS API",
)

# Enable CORS for C4ISR tactical dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

pipeline_instance: MasterPipeline | None = None
camera_instance = None
latest_jpeg_bytes: bytes | None = None
camera_status_msg: str = "INITIALIZING OPTICAL FEED"

# Advanced v4.0 Subsystem Engines
trajectory_predictor = TrajectoryPredictor()
threat_matrix = ThreatMatrixEngine()
thermal_shader = ThermalFusionShader(mode=ThermalVisionMode.EO)
sensor_fusion = SensorFusionEngine()
swarm_grid = SwarmDefenseGrid()
anomaly_detector = AnomalyDetector()
blackbox_recorder = BlackboxRecorder()
intercept_calculator = InterceptCalculator()
countermeasure_engine = CountermeasureEngine()



def get_pipeline() -> MasterPipeline:
    global pipeline_instance
    if pipeline_instance is None:
        pipeline_instance = MasterPipeline()
    return pipeline_instance


def draw_hud_overlay(img: np.ndarray, tracks: list) -> np.ndarray:
    # 1. Apply Thermal Vision Shader
    annotated = thermal_shader.apply_fusion(img).copy()
    h, w = annotated.shape[:2]

    # 2. Compute Trajectory & Threat Analysis
    traj_data = trajectory_predictor.update_and_predict(tracks)
    threat_data = threat_matrix.evaluate_threats(tracks, traj_data, frame_w=w, frame_h=h)
    swarm_data = swarm_grid.analyze_swarms(tracks)
    anomalies = anomaly_detector.detect_anomalies(tracks, traj_data)

    # Record to Blackbox Logger
    blackbox_recorder.record_frame_event(
        frame_id=int(time.time() * 30),
        tracks=tracks,
        threat_data=threat_data,
        thermal_mode=thermal_shader.mode,
    )

    # Center Reticle
    cx, cy = w // 2, h // 2
    reticle_color = (248, 189, 56) if thermal_shader.mode == ThermalVisionMode.EO else (0, 255, 255)
    cv2.circle(annotated, (cx, cy), 30, reticle_color, 1)
    cv2.line(annotated, (cx - 45, cy), (cx - 15, cy), reticle_color, 1)
    cv2.line(annotated, (cx + 15, cy), (cx + 45, cy), reticle_color, 1)
    cv2.line(annotated, (cx, cy - 45), (cx, cy - 15), reticle_color, 1)
    cv2.line(annotated, (cx, cy + 15), (cx, cy + 45), reticle_color, 1)

    # Swarm Alert Banner
    if swarm_data.get("swarm_detected", False):
        cv2.putText(
            annotated,
            f"SWARM ALERT: {swarm_data['drone_count']} DRONES DETECTED",
            (40, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    # Active Detections & Tracks
    for tr in tracks:
        x1, y1, x2, y2 = int(tr.bbox.x1), int(tr.bbox.y1), int(tr.bbox.x2), int(tr.bbox.y2)
        tid = tr.track_id
        t_info = threat_data.get("threat_matrix", {}).get(tid, {})
        t_level = t_info.get("level", "CHARLIE")
        t_score = t_info.get("score", 0.0)

        color = (129, 185, 16)
        if t_level == "ALPHA":
            color = (0, 0, 255)  # Bright Red for ALPHA Threat
        elif t_level == "BRAVO":
            color = (0, 165, 255)  # Orange for BRAVO Threat

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Tactical corner brackets
        corner = min(15, max(5, (x2 - x1) // 3))
        cv2.line(annotated, (x1, y1), (x1 + corner, y1), color, 3)
        cv2.line(annotated, (x1, y1), (x1, y1 + corner), color, 3)
        cv2.line(annotated, (x2, y1), (x2 - corner, y1), color, 3)
        cv2.line(annotated, (x2, y1), (x2, y1 + corner), color, 3)

        label = f"TRK #{tr.track_id:02d} {tr.class_name.upper()} [{int(tr.confidence * 100)}%] {t_level}"
        cv2.putText(annotated, label, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        speed = getattr(tr, "speed_kmh", 0.0) if getattr(tr, "speed_kmh", None) is not None else 0.0
        cv2.putText(annotated, f"VEL: {speed:.1f} km/h | T: {t_score:.0f}%", (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # 3. Draw Future Flight Trajectory Vector Dotted Line (+1s, +2s, +3s, +5s)
        t_traj = traj_data.get(tid, {}).get("future_points", [])
        prev_p = (int(tr.bbox.cx), int(tr.bbox.cy))
        for fp in t_traj:
            next_p = (int(fp[0]), int(fp[1]))
            cv2.line(annotated, prev_p, next_p, color, 1, cv2.LINE_AA)
            cv2.circle(annotated, next_p, 3, color, -1)
            prev_p = next_p

    return annotated



    return annotated


def create_synthetic_frame(status_text: str = "SEARCHING FOR OPTICAL FEED") -> bytes:
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    for x in range(0, 1280, 40):
        cv2.line(img, (x, 0), (x, 720), (30, 25, 15), 1)
    for y in range(0, 720, 40):
        cv2.line(img, (0, y), (1280, y), (30, 25, 15), 1)

    cx, cy = 640, 360
    cv2.circle(img, (cx, cy), 40, (248, 189, 56), 1)
    cv2.line(img, (cx - 60, cy), (cx - 20, cy), (248, 189, 56), 1)
    cv2.line(img, (cx + 20, cy), (cx + 60, cy), (248, 189, 56), 1)
    cv2.line(img, (cx, cy - 60), (cx, cy - 20), (248, 189, 56), 1)
    cv2.line(img, (cx, cy + 20), (cx, cy + 60), (248, 189, 56), 1)

    cv2.putText(img, "APEX-TRACK C4ISR OPTICAL FEED", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (248, 189, 56), 2)
    cv2.putText(img, f"STATUS: {status_text}", (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (16, 185, 129), 2)

    _, jpeg = cv2.imencode(".jpg", img)
    return jpeg.tobytes()


async def camera_pipeline_worker():
    global camera_instance, latest_jpeg_bytes, camera_status_msg

    pipeline = get_pipeline()
    hw = hw_detector.detect()
    cfg_mgr = ConfigManager.instance()
    config = cfg_mgr.load(hw_profile_name=hw.profile_name)

    # Initialize detector and tracker plugins to active state
    await pipeline.initialize(config.model_dump(), hw)

    camera_src = os.environ.get("APEX_CAMERA_SOURCE") or config.camera_manager.cameras[0].source
    camera_plug = os.environ.get("APEX_CAMERA_PLUGIN") or config.camera_manager.cameras[0].plugin


    log.info("starting_camera_worker", source=camera_src, plugin=camera_plug)

    if camera_plug == "rtsp_camera" or str(camera_src).startswith("http://") or str(camera_src).startswith("rtsp://"):
        camera_instance = RTSPCameraPlugin(camera_id="cam_0")
    else:
        camera_instance = USBCameraPlugin(camera_id="cam_0")

    camera_instance.config = {
        "source": camera_src,
        "width": 1280,
        "height": 720,
        "fps": 30,
    }

    try:
        connected = await asyncio.wait_for(camera_instance._connect(), timeout=3.0)
    except Exception:
        connected = False

    if not connected:
        log.warning("camera_connection_failed", source=camera_src)
        camera_status_msg = f"CAMERA DISCONNECTED ({camera_src})"

    seq_id = 0
    while True:
        try:
            if not connected or camera_instance._cap is None or not camera_instance._cap.isOpened():
                camera_status_msg = f"CONNECTING TO {camera_src}..."
                try:
                    connected = await asyncio.wait_for(camera_instance._connect(), timeout=3.0)
                except Exception:
                    connected = False

                if not connected:
                    latest_jpeg_bytes = create_synthetic_frame(f"CONNECTING TO {camera_src}...")
                    await asyncio.sleep(1.0)
                    continue


            res = await camera_instance._grab_frame()
            if res is None:
                await asyncio.sleep(0.01)
                continue

            img, ts = res
            seq_id += 1
            h, w = img.shape[:2]

            frame_obj = Frame(
                data=img,
                metadata=FrameMetadata(camera_id="cam_0", width=w, height=h),
                timestamp=ts,
                sequence_id=seq_id,
            )

            track_array = await pipeline.process_frame(frame_obj)

            def render_hud_bytes(raw_image, active_tracks):
                annotated_img = draw_hud_overlay(raw_image, active_tracks)
                _, encoded_buf = cv2.imencode(".jpg", annotated_img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                return encoded_buf.tobytes()

            latest_jpeg_bytes = await asyncio.to_thread(render_hud_bytes, img, track_array.tracks)
            camera_status_msg = "STREAM ACTIVE"


        except Exception as exc:
            log.error("camera_worker_error", error=str(exc))
            latest_jpeg_bytes = create_synthetic_frame(f"ERROR: {str(exc)}")
            await asyncio.sleep(1.0)


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(camera_pipeline_worker())


@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    index_file = static_dir / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>APEX-Track Perception API Online</h1>")


@app.get("/video_feed")
async def video_feed():
    """MJPEG Live Optical Feed with target detection overlay."""
    async def frame_generator():
        while True:
            frame_bytes = latest_jpeg_bytes or create_synthetic_frame(camera_status_msg)
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            await asyncio.sleep(0.033)

    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/health")
@app.get("/api/v1/health")
async def get_health() -> Dict[str, Any]:
    return {"status": "HEALTHY", "engine": "APEX-Track", "version": "1.0.0"}


@app.get("/api/v1/targets")
async def get_active_targets() -> Dict[str, Any]:
    pipeline = get_pipeline()
    targets = pipeline.target_db.get_active_targets()
    return {
        "count": len(targets),
        "targets": [
            {
                "track_id": t.track_id,
                "class_name": t.class_name,
                "state": t.state.name,
                "confidence": t.confidence,
                "bbox": [t.bbox.x1, t.bbox.y1, t.bbox.x2, t.bbox.y2],
                "world_point": t.world_point,
                "speed_kmh": t.speed_kmh,
            }
            for t in targets
        ],
    }


@app.post("/api/v1/missions/switch")
async def switch_mission(profile_name: str) -> Dict[str, Any]:
    pipeline = get_pipeline()
    path = f"configs/missions/{profile_name}.yaml"
    try:
        profile = pipeline.mission_mgr.load_mission_profile(path)
        return {"status": "SUCCESS", "active_profile": profile.name}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/v1/roboflow/model")
async def switch_roboflow_model(model_id: str):
    """Dynamically switch active neural detector to Roboflow Universe model."""
    pipeline = get_pipeline()
    if hasattr(pipeline.detector, "model_id"):
        pipeline.detector.model_id = model_id
    log.info("switched_roboflow_model", model_id=model_id)
    return {"status": "success", "active_roboflow_model": model_id}


@app.post("/api/v1/vision/mode")

async def set_vision_mode(mode: str):
    """Sets multi-spectral thermal IR vision mode (EO, FLIR_IRONBOW, FLIR_WHITE_HOT, FLIR_BLACK_HOT, NVG_GREEN)."""
    valid_modes = [ThermalVisionMode.EO, ThermalVisionMode.FLIR_IRONBOW, ThermalVisionMode.FLIR_WHITE_HOT, ThermalVisionMode.FLIR_BLACK_HOT, ThermalVisionMode.NVG_GREEN]
    mode_upper = mode.upper().strip()
    if mode_upper not in valid_modes:
        raise HTTPException(status_code=400, detail=f"Invalid vision mode. Must be one of {valid_modes}")
    thermal_shader.mode = mode_upper
    return {"status": "success", "vision_mode": mode_upper}


@app.post("/api/v1/targets/lock")
async def lock_target(track_id: int):
    """Manually lock optical tracking gimbal onto specific track ID."""
    threat_matrix.primary_lock_id = track_id
    return {"status": "success", "primary_lock_id": track_id}


@app.get("/api/v1/threats")
async def get_threat_status():
    """Returns real-time threat matrix and drone swarm analytics."""
    pipeline = get_pipeline()
    tracks = pipeline.active_tracks.tracks
    traj_data = trajectory_predictor.update_and_predict(tracks)
    t_eval = threat_matrix.evaluate_threats(tracks, traj_data)
    swarm_eval = swarm_grid.analyze_swarms(tracks)
    return {
        "threat_assessment": t_eval,
        "swarm_analytics": swarm_eval,
        "active_vision_mode": thermal_shader.mode,
    }


@app.get("/api/v1/telemetry/advanced")
async def get_advanced_telemetry():
    """Returns spatial trajectories, acoustic bearings, and kinematic anomalies."""
    pipeline = get_pipeline()
    tracks = pipeline.active_tracks.tracks
    traj_data = trajectory_predictor.update_and_predict(tracks)
    acoustic_fused = sensor_fusion.correlate_tracks(tracks)
    anomalies = anomaly_detector.detect_anomalies(tracks, traj_data)
    return {
        "trajectories": traj_data,
        "rf_acoustic_fusion": acoustic_fused,
        "anomalies": anomalies,
    }


@app.post("/api/v1/countermeasures/jam")
async def trigger_jamming(target_id: int):
    """Manually trigger directional RF Jamming soft-kill on target ID."""
    return countermeasure_engine.trigger_manual_jamming(target_id)


@app.post("/api/v1/countermeasures/intercept")
async def trigger_intercept(target_id: int):
    """Manually engage kinetic intercept pursuit hard-kill on target ID."""
    return countermeasure_engine.trigger_manual_intercept(target_id)


@app.get("/api/v1/intercept/vectors")
async def get_intercept_vectors():
    """Returns 3D intercept bearing, elevation, slant range, and TTI calculations."""
    pipeline = get_pipeline()
    tracks = pipeline.active_tracks.tracks
    vectors = [intercept_calculator.compute_intercept(tr) for tr in tracks]
    traj_data = trajectory_predictor.update_and_predict(tracks)
    t_eval = threat_matrix.evaluate_threats(tracks, traj_data)
    cm_eval = countermeasure_engine.evaluate_countermeasures(t_eval, vectors)
    return {
        "intercept_vectors": vectors,
        "countermeasure_status": cm_eval,
    }


@app.websocket("/ws/telemetry")


async def telemetry_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    log.info("ws_client_connected", client=str(websocket.client))
    try:
        pipeline = get_pipeline()
        while True:
            targets = pipeline.target_db.get_active_targets()
            data = {
                "timestamp": targets[0].frame_timestamp if targets else 0.0,
                "active_targets_count": len(targets),
                "targets": [
                    {
                        "id": t.track_id,
                        "class": t.class_name,
                        "state": t.state.name,
                        "cx": t.bbox.cx,
                        "cy": t.bbox.cy,
                    }
                    for t in targets
                ],
            }
            await websocket.send_json(data)
            await asyncio.sleep(0.033)
    except WebSocketDisconnect:
        log.info("ws_client_disconnected")
