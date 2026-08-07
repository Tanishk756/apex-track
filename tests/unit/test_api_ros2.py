"""
Unit Tests — API Server & ROS2 (Phase 11 & 12)
"""

import pytest
import numpy as np

from apex.api.server import get_health, get_active_targets, switch_mission
from ros2.nodes.detection_node import ROS2DetectionNodeAdapter


class TestAPIServer:

    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        res = await get_health()
        assert res["status"] == "HEALTHY"
        assert res["engine"] == "APEX-Track"

    @pytest.mark.asyncio
    async def test_targets_endpoint(self):
        res = await get_active_targets()
        assert "targets" in res
        assert "count" in res

    @pytest.mark.asyncio
    async def test_switch_mission_endpoint(self):
        res = await switch_mission("road_vehicles")
        assert res["status"] == "SUCCESS"
        assert res["active_profile"] == "road_vehicles"


class TestROS2Adapter:

    def test_ros2_node_adapter(self):
        adapter = ROS2DetectionNodeAdapter()
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = adapter.process_ros_image(img)
        assert isinstance(dets, list)
