"""
Unit Tests — Persistent Target Re-Identification (Re-ID) Subsystem
"""

import numpy as np
import pytest

from apex.engine.contracts.detection import BoundingBox, Detection
from apex.engine.contracts.frame import Frame, FrameMetadata
from apex.engine.tracker.reid_db import PersistentReIDDatabase
from apex.engine.tracker.reid_extractor import ReIDFeatureExtractor
from plugins.trackers.bytetrack.plugin import ByteTrackPlugin


class TestReIDSubsystem:

    def setup_method(self):
        PersistentReIDDatabase.instance().reset()

    def test_feature_extraction_and_cosine_similarity(self):
        extractor = ReIDFeatureExtractor(feature_dim=128)
        img = np.ones((720, 1280, 3), dtype=np.uint8) * 128
        # Add distinct color patch
        img[100:200, 100:200] = [255, 0, 0]

        bbox1 = BoundingBox(x1=100, y1=100, x2=200, y2=200)
        bbox2 = BoundingBox(x1=105, y1=105, x2=205, y2=205) # Similar region
        bbox3 = BoundingBox(x1=500, y1=500, x2=600, y2=600) # Different region

        feat1 = extractor.extract_crop_feature(img, bbox1)
        feat2 = extractor.extract_crop_feature(img, bbox2)
        feat3 = extractor.extract_crop_feature(img, bbox3)

        sim_1_2 = extractor.compute_cosine_similarity(feat1, feat2)
        sim_1_3 = extractor.compute_cosine_similarity(feat1, feat3)

        assert sim_1_2 > 0.85
        assert sim_1_3 < sim_1_2

    def test_persistent_reid_db_match_and_recovery(self):
        reid_db = PersistentReIDDatabase.instance()
        img = np.random.randint(0, 50, (720, 1280, 3), dtype=np.uint8)
        img[100:200, 100:200] = [0, 255, 0] # Green target box

        bbox = BoundingBox(x1=100, y1=100, x2=200, y2=200)

        # 1. First time target appears: gets assigned UID #1
        uid1, is_reid1 = reid_db.match_or_create_uid(img, bbox, "person")
        assert uid1 == 1
        assert not is_reid1

        # 2. Target leaves frame and reappears: gets re-assigned UID #1 via Cosine Similarity!
        bbox_reappear = BoundingBox(x1=102, y1=102, x2=202, y2=202)
        uid2, is_reid2 = reid_db.match_or_create_uid(img, bbox_reappear, "person")
        assert uid2 == 1
        assert is_reid2

        # 3. Completely different target appears: gets assigned new UID #2
        img_diff = np.random.randint(0, 50, (720, 1280, 3), dtype=np.uint8)
        img_diff[400:550, 400:550] = [255, 0, 150] # Bright magenta distinct box
        bbox_diff = BoundingBox(x1=400, y1=400, x2=550, y2=550)

        uid3, is_reid3 = reid_db.match_or_create_uid(img_diff, bbox_diff, "person")
        assert uid3 == 2
        assert not is_reid3

    def test_bytetrack_persistent_uid_integration(self):
        tracker = ByteTrackPlugin()
        img = np.random.randint(0, 50, (720, 1280, 3), dtype=np.uint8)
        img[50:150, 50:150] = [200, 100, 50]
        frame = Frame(data=img, metadata=FrameMetadata(camera_id="cam_0", width=1280, height=720))

        det1 = Detection(bbox=BoundingBox(x1=50, y1=50, x2=150, y2=150), confidence=0.9, class_id=0, class_name="person")

        # Frame 1: Target gets track ID
        tracks1 = tracker.update([det1], frame)
        assert len(tracks1) == 1
        assigned_id = tracks1[0].track_id

        # Simulate track loss (target disappears for 40 frames)
        empty_frame = Frame(data=img, metadata=FrameMetadata(camera_id="cam_0", width=1280, height=720))
        for _ in range(40):
            tracker.update([], empty_frame)

        # Target reappears at slightly moved position: recovers original assigned_id!
        det1_reappear = Detection(bbox=BoundingBox(x1=55, y1=55, x2=155, y2=155), confidence=0.9, class_id=0, class_name="person")
        tracks_reappear = tracker.update([det1_reappear], frame)
        assert len(tracks_reappear) == 1
        assert tracks_reappear[0].track_id == assigned_id
