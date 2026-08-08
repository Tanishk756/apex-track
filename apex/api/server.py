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
from plugins.cameras.rtsp_camera.plugin import RTSPCameraPlugin
from plugins.cameras.usb_camera.plugin import USBCameraPlugin

log = structlog.get_logger(__name__)

app = FastAPI(
    title="APEX-Track Perception Platform API",
    version="1.0.0",
    description="Ultra-low latency defense-grade detection & tracking REST/WS API",
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


def get_pipeline() -> MasterPipeline:
    global pipeline_instance
    if pipeline_instance is None:
        pipeline_instance = MasterPipeline()
    return pipeline_instance


def draw_hud_overlay(img: np.ndarray, tracks: list) -> np.ndarray:
    annotated = img.copy()
    h, w = annotated.shape[:2]

    # Center Reticle
    cx, cy = w // 2, h // 2
    cv2.circle(annotated, (cx, cy), 30, (248, 189, 56), 1)
    cv2.line(annotated, (cx - 45, cy), (cx - 15, cy), (248, 189, 56), 1)
    cv2.line(annotated, (cx + 15, cy), (cx + 45, cy), (248, 189, 56), 1)
    cv2.line(annotated, (cx, cy - 45), (cx, cy - 15), (248, 189, 56), 1)
    cv2.line(annotated, (cx, cy + 15), (cx, cy + 45), (248, 189, 56), 1)

    # Active Detections & Tracks
    for tr in tracks:
        x1, y1, x2, y2 = int(tr.bbox.x1), int(tr.bbox.y1), int(tr.bbox.x2), int(tr.bbox.y2)
        color = (129, 185, 16) if tr.state.name == "CONFIRMED" else (11, 158, 245)

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Tactical corner brackets
        corner = min(15, max(5, (x2 - x1) // 3))
        cv2.line(annotated, (x1, y1), (x1 + corner, y1), color, 3)
        cv2.line(annotated, (x1, y1), (x1, y1 + corner), color, 3)
        cv2.line(annotated, (x2, y1), (x2 - corner, y1), color, 3)
        cv2.line(annotated, (x2, y1), (x2, y1 + corner), color, 3)

        label = f"TRK #{tr.track_id:02d} {tr.class_name.upper()} [{int(tr.confidence * 100)}%]"
        cv2.putText(annotated, label, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        speed = getattr(tr, 'speed_kmh', 100.0)
        cv2.putText(annotated, f"VEL: {speed:.1f} km/h", (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

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

    connected = await camera_instance._connect()
    if not connected:
        log.warning("camera_connection_failed", source=camera_src)
        camera_status_msg = f"CAMERA DISCONNECTED ({camera_src})"

    seq_id = 0
    while True:
        try:
            if not connected or camera_instance._cap is None or not camera_instance._cap.isOpened():
                camera_status_msg = f"CONNECTING TO {camera_src}..."
                connected = await camera_instance._connect()
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
            annotated = draw_hud_overlay(img, track_array.tracks)

            _, jpeg = cv2.imencode(".jpg", annotated)
            latest_jpeg_bytes = jpeg.tobytes()
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
