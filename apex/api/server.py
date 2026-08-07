"""
FastAPI Operational REST & WebSocket Command Server
===================================================
Tactical API server exposing endpoints for:
- System state & hardware profile metrics (/api/v1/health, /api/v1/status)
- Active target tracks (/api/v1/targets)
- Mission profile switching (/api/v1/missions)
- Live WebSocket stream for real-time telemetry and target HUD overlays (/ws/telemetry)
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import structlog

from apex.engine.pipeline.master_pipeline import MasterPipeline

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


@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Return C4ISR Glassmorphic Tactical Command Dashboard."""
    index_file = static_dir / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>APEX-Track Perception API Online</h1>")

pipeline_instance: MasterPipeline | None = None


def get_pipeline() -> MasterPipeline:
    global pipeline_instance
    if pipeline_instance is None:
        pipeline_instance = MasterPipeline()
    return pipeline_instance


@app.get("/health")
@app.get("/api/v1/health")
async def get_health() -> Dict[str, Any]:
    """Health check endpoint for Kubernetes / Docker liveness probes."""
    return {"status": "HEALTHY", "engine": "APEX-Track", "version": "1.0.0"}


@app.get("/api/v1/targets")
async def get_active_targets() -> Dict[str, Any]:
    """Return currently active confirmed and coasting target tracks."""
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
    """Switch active mission profile at runtime (road_vehicles, drone_tracking, battlefield)."""
    pipeline = get_pipeline()
    path = f"configs/missions/{profile_name}.yaml"
    try:
        profile = pipeline.mission_mgr.load_mission_profile(path)
        return {"status": "SUCCESS", "active_profile": profile.name}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.websocket("/ws/telemetry")
async def telemetry_websocket(websocket: WebSocket) -> None:
    """Live WebSocket channel streaming active targets and telemetry."""
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
            await asyncio.sleep(0.033)  # ~30 Hz stream rate
    except WebSocketDisconnect:
        log.info("ws_client_disconnected")
