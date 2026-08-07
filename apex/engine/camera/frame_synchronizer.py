"""
Frame Synchronizer
==================
Buffers and synchronizes frames across multiple cameras (e.g., EO/IR dual sensors).
Emits synchronized frame bundles when timestamps across registered cameras match within tolerance.

Design:
- Maintains deque ring buffers per camera.
- Sliding time-window matching within sync_tolerance_ms.
- Drop-oldest strategy for un-matched stragglers to maintain low latency.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Callable, Optional

import structlog

from apex.engine.contracts.frame import Frame

log = structlog.get_logger(__name__)


class FrameSynchronizer:
    """
    Multi-camera frame synchronizer.

    Usage::
        sync = FrameSynchronizer(camera_ids=["eo_cam", "ir_cam"], tolerance_ms=50.0)
        sync.push(frame_eo)
        sync.push(frame_ir)
        bundle = sync.pop_synced_bundle()  # dict[str, Frame] or None
    """

    def __init__(self, camera_ids: list[str], tolerance_ms: float = 50.0, max_buffer_size: int = 30) -> None:
        self.camera_ids = list(camera_ids)
        self.tolerance_s = tolerance_ms / 1000.0
        self.max_buffer_size = max_buffer_size
        self._buffers: dict[str, deque[Frame]] = {cid: deque(maxlen=max_buffer_size) for cid in camera_ids}
        self._synced_count = 0
        self._dropped_count = 0

    def push(self, frame: Frame) -> None:
        """Push an incoming frame into its camera buffer."""
        cid = frame.metadata.camera_id
        if cid in self._buffers:
            self._buffers[cid].append(frame)
        else:
            log.warning("unregistered_camera_frame_ignored", camera_id=cid)

    def pop_synced_bundle(self) -> Optional[dict[str, Frame]]:
        """
        Find and return the oldest timestamp-aligned frame bundle containing
        one frame per registered camera within tolerance_ms.
        Returns None if no matching bundle is available yet.
        """
        if not self.camera_ids or any(len(b) == 0 for b in self._buffers.values()):
            return None

        # Pick pivot camera (the one with the oldest front frame)
        pivot_cid = min(self.camera_ids, key=lambda c: self._buffers[c][0].timestamp)
        pivot_frame = self._buffers[pivot_cid][0]
        pivot_ts = pivot_frame.timestamp

        bundle: dict[str, Frame] = {pivot_cid: pivot_frame}
        matches_found = True

        for cid in self.camera_ids:
            if cid == pivot_cid:
                continue

            buf = self._buffers[cid]
            best_match: Optional[Frame] = None
            min_diff = float("inf")

            # Search buffer for frame closest to pivot timestamp
            for frame in buf:
                diff = abs(frame.timestamp - pivot_ts)
                if diff <= self.tolerance_s and diff < min_diff:
                    min_diff = diff
                    best_match = frame

            if best_match is not None:
                bundle[cid] = best_match
            else:
                matches_found = False
                break

        if matches_found:
            # Pop matched frames from all buffers, plus any older stragglers before matched frame
            for cid, matched_frame in bundle.items():
                buf = self._buffers[cid]
                while buf and buf[0].timestamp <= matched_frame.timestamp:
                    popped = buf.popleft()
                    if popped.sequence_id != matched_frame.sequence_id:
                        self._dropped_count += 1
            self._synced_count += 1
            return bundle

        # If pivot frame cannot be matched by all other cameras and its timestamp is too old, drop it
        now = time.time()
        if now - pivot_ts > (self.tolerance_s * 5):
            self._buffers[pivot_cid].popleft()
            self._dropped_count += 1

        return None

    @property
    def synced_count(self) -> int:
        return self._synced_count

    @property
    def dropped_count(self) -> int:
        return self._dropped_count
