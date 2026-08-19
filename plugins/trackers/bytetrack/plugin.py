"""
ByteTrack Plugin
================
MIT Licensed ByteTrack Multi-Target Tracker Plugin.
Implements two-stage association matching high and low-confidence detections
to prevent track drops during target occlusions and motion blur.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
import structlog

from apex.engine.contracts.detection import BoundingBox, Detection
from apex.engine.contracts.frame import Frame
from apex.engine.contracts.track import Track, TrackState
from apex.engine.plugins.plugin_base import PluginMetadata, PluginType
from apex.engine.tracker.kalman import KalmanFilterTarget
from apex.engine.tracker.tracker_base import TrackerBase

log = structlog.get_logger(__name__)


class SingleByteTrack:
    """Internal track state representation for ByteTrack."""

    def __init__(self, track_id: int, det: Detection, kalman_filter: KalmanFilterTarget) -> None:
        self.track_id = track_id
        self.kf = kalman_filter
        self.mean, self.covariance = self.kf.initiate(self.kf.bbox_to_z(det.bbox))

        self.state = TrackState.TENTATIVE
        self.class_id = det.class_id
        self.class_name = det.class_name
        self.confidence = det.confidence
        self.camera_id = det.camera_id

        self.age = 0
        self.hits = 1
        self.misses = 0

    def predict(self) -> BoundingBox:
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
        self.age += 1
        self.misses += 1
        return self.kf.z_to_bbox(self.mean)

    def update(self, det: Detection) -> None:
        self.mean, self.covariance = self.kf.update(self.mean, self.covariance, self.kf.bbox_to_z(det.bbox))
        if det.confidence >= self.confidence * 0.9:
            self.class_name = det.class_name
            self.class_id = det.class_id
        self.confidence = det.confidence
        self.hits += 1
        self.misses = 0
        if self.state == TrackState.TENTATIVE and self.hits >= 1:
            self.state = TrackState.CONFIRMED
        elif self.state == TrackState.COASTING:
            self.state = TrackState.CONFIRMED


    @property
    def current_bbox(self) -> BoundingBox:
        return self.kf.z_to_bbox(self.mean)

    def predict_next_bbox(self) -> BoundingBox:
        pred_mean, _ = self.kf.predict(self.mean.copy(), self.covariance.copy())
        return self.kf.z_to_bbox(pred_mean)


class ByteTrackPlugin(TrackerBase):
    """ByteTrack Multi-Object Tracker Plugin."""

    metadata = PluginMetadata(
        name="bytetrack",
        version="1.0.0",
        plugin_type=PluginType.TRACKER,
        license="MIT",
        author="APEX-Track",
        description="ByteTrack Multi-Target Tracker Plugin",
    )

    def __init__(self) -> None:
        super().__init__()
        self.kf = KalmanFilterTarget()
        self.tracked_objects: list[SingleByteTrack] = []

    def update(self, detections: list[Detection], frame: Frame) -> list[Track]:
        # Split detections into high score and low score groups
        det_high: list[Detection] = []
        det_low: list[Detection] = []

        for d in detections:
            if d.confidence >= self.track_high_thresh:
                det_high.append(d)
            elif d.confidence >= self.track_low_thresh:
                det_low.append(d)

        # 1. Predict all existing track positions via Kalman filter
        for track in self.tracked_objects:
            track.predict()

        # 2. Stage 1 Association: High-confidence detections vs Active Tracks
        unmatched_tracks_1, unmatched_dets_1, matched_1 = self._associate(
            self.tracked_objects, det_high, self.match_thresh
        )

        from apex.engine.tracker.reid_db import PersistentReIDDatabase
        reid_db = PersistentReIDDatabase.instance()

        for track_idx, det_idx in matched_1:
            det = det_high[det_idx]
            track = self.tracked_objects[track_idx]
            track.update(det)
            reid_db.update_feature_for_uid(track.track_id, frame.data if frame else None, det.bbox)

        # 3. Stage 2 Association: Unmatched active tracks vs Low-confidence detections
        rem_tracks = [self.tracked_objects[i] for i in unmatched_tracks_1]
        unmatched_tracks_2, _, matched_2 = self._associate(
            rem_tracks, det_low, max(0.5, self.match_thresh - 0.2)
        )

        for rel_track_idx, det_idx in matched_2:
            det = det_low[det_idx]
            track = rem_tracks[rel_track_idx]
            track.update(det)
            reid_db.update_feature_for_uid(track.track_id, frame.data if frame else None, det.bbox)

        # Mark un-matched tracks as COASTING / LOST / DELETED
        final_unmatched = [rem_tracks[i] for i in unmatched_tracks_2]
        for track in final_unmatched:
            if track.misses > self.max_time_lost:
                track.state = TrackState.DELETED
            else:
                track.state = TrackState.COASTING

        # 4. Initiate new tentative tracks from unmatched high-confidence detections (preventing duplicate UIDs)
        existing_uids = {t.track_id for t in self.tracked_objects if t.state != TrackState.DELETED}
        for i in unmatched_dets_1:
            det = det_high[i]
            if det.confidence >= self.new_track_thresh:
                # Query Persistent Re-ID database to recover previous UID or assign new UID
                persistent_uid, is_reid = reid_db.match_or_create_uid(
                    frame.data if frame else None, det.bbox, det.class_name
                )
                if persistent_uid not in existing_uids:
                    new_t = SingleByteTrack(persistent_uid, det, self.kf)
                    self.tracked_objects.append(new_t)
                    existing_uids.add(persistent_uid)

        # Purge deleted tracks and deduplicate by track_id (guarantees 1 bounding box per target)
        unique_tracks_map: dict[int, SingleByteTrack] = {}
        for t in self.tracked_objects:
            if t.state != TrackState.DELETED:
                if t.track_id not in unique_tracks_map or t.confidence > unique_tracks_map[t.track_id].confidence:
                    unique_tracks_map[t.track_id] = t
        self.tracked_objects = list(unique_tracks_map.values())

        # Convert to Track contract objects for message bus output
        output_tracks: list[Track] = []
        for t in self.tracked_objects:
            # Calculate pixel velocity from Kalman state vector
            vx, vy = float(t.mean[4]), float(t.mean[5])
            tr = Track(
                track_id=t.track_id,
                state=t.state,
                bbox=t.current_bbox,
                predicted_bbox=t.predict_next_bbox(),
                confidence=t.confidence,
                class_id=t.class_id,
                class_name=t.class_name,
                frame_timestamp=frame.timestamp,
                camera_id=frame.metadata.camera_id,
                velocity_px=(vx, vy),
                age_frames=t.age,
                hits=t.hits,
                misses=t.misses,
            )
            output_tracks.append(tr)

        return output_tracks

    def _associate(
        self,
        tracks: list[SingleByteTrack],
        detections: list[Detection],
        iou_threshold: float,
    ) -> tuple[list[int], list[int], list[tuple[int, int]]]:
        if not tracks or not detections:
            return list(range(len(tracks))), list(range(len(detections))), []

        # Construct IoU Cost Matrix
        cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        for i, t in enumerate(tracks):
            for j, d in enumerate(detections):
                iou = t.current_bbox.iou(d.bbox)
                cost_matrix[i, j] = 1.0 - iou

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched: list[tuple[int, int]] = []
        unmatched_tracks = set(range(len(tracks)))
        unmatched_dets = set(range(len(detections)))

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] <= (1.0 - iou_threshold):
                matched.append((r, c))
                unmatched_tracks.remove(r)
                unmatched_dets.remove(c)

        return list(unmatched_tracks), list(unmatched_dets), matched
