"""
Dual-Channel Video Recorder & HUD Overlay Generator
===================================================
Asynchronous OpenCV VideoWriter wrapper for recording raw input feeds and HUD annotated track feeds.
"""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import structlog

from apex.engine.contracts.frame import Frame
from apex.engine.contracts.track import TrackArray

log = structlog.get_logger(__name__)


class VideoRecorder:
    """Records raw and HUD-annotated video channels to MP4/AVI files."""

    def __init__(self, output_dir: str = "recordings", fps: float = 30.0) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self._raw_writer: cv2.VideoWriter | None = None
        self._hud_writer: cv2.VideoWriter | None = None
        self._is_recording = False

    def start_recording(self, width: int = 640, height: int = 480, session_name: str = "mission_rec") -> None:
        raw_path = str(self.output_dir / f"{session_name}_raw.mp4")
        hud_path = str(self.output_dir / f"{session_name}_hud.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._raw_writer = cv2.VideoWriter(raw_path, fourcc, self.fps, (width, height))
        self._hud_writer = cv2.VideoWriter(hud_path, fourcc, self.fps, (width, height))
        self._is_recording = True

        log.info("video_recording_started", raw_file=raw_path, hud_file=hud_path)

    def write_frame(self, frame: Frame, track_array: TrackArray) -> None:
        """Write raw frame and render HUD overlays."""
        if not self._is_recording or frame.data is None:
            return

        # Write raw
        if self._raw_writer is not None:
            self._raw_writer.write(frame.data)

        # Render HUD annotations
        hud_img = frame.data.copy()
        for track in track_array.tracks:
            box = track.bbox
            pt1 = (int(box.x1), int(box.y1))
            pt2 = (int(box.x2), int(box.y2))
            cv2.rectangle(hud_img, pt1, pt2, (0, 255, 0), 2)

            label = f"#{track.track_id} {track.class_name} {track.confidence:.2f}"
            cv2.putText(hud_img, label, (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if self._hud_writer is not None:
            self._hud_writer.write(hud_img)

    def stop_recording(self) -> None:
        self._is_recording = False
        if self._raw_writer is not None:
            self._raw_writer.release()
            self._raw_writer = None
        if self._hud_writer is not None:
            self._hud_writer.release()
            self._hud_writer = None
        log.info("video_recording_stopped")
