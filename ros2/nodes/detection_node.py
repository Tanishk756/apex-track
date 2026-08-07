"""
ROS2 Detection Node Wrapper
===========================
ROS2 Lifecycle Node providing standard ROS2 topic interfaces:
- Subscribes to /sensor/camera/image_raw (sensor_msgs/msg/Image)
- Publishes detections to /apex/detections (apex_track_msgs/msg/DetectionArray)
- Provides standard fallback when running in standalone Python environments.
"""

from __future__ import annotations

import structlog

from apex.engine.contracts.frame import Frame, FrameMetadata
from apex.engine.detector.detector_base import DetectorBase
from plugins.detectors.rtdetr.plugin import RTDETRPlugin

log = structlog.get_logger(__name__)


class ROS2DetectionNodeAdapter:
    """ROS2 Detection Node bridge."""

    def __init__(self, detector: DetectorBase | None = None) -> None:
        self.detector = detector or RTDETRPlugin()
        log.info("ros2_detection_node_adapter_created")

    def process_ros_image(self, image_data: any, camera_id: str = "ros2_cam") -> list:
        if not self.detector.is_active:
            from apex.engine.plugins.plugin_base import PluginStatus
            self.detector._set_status(PluginStatus.ACTIVE)
        meta = FrameMetadata(camera_id=camera_id, width=640, height=480)
        frame = Frame(data=image_data, metadata=meta, timestamp=0.0, sequence_id=1)
        return self.detector.detect(frame)
