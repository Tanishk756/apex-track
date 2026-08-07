"""
World Coordinate Geolocation System
===================================
Pinhole camera projection model and ray-plane terrain intersection engine.

Converts 2D image pixel coordinates (x, y) into real-world 3D WGS84 GPS coordinates (lat, lon, alt)
using UAV telemetry inputs (UAV lat/lon/alt, camera pitch/roll/yaw, focal length, sensor dimensions).
"""

from __future__ import annotations

import math
import numpy as np

EARTH_RADIUS_METERS = 6378137.0  # WGS-84 equatorial radius


class WorldCoordinateSystem:
    """Target 3D geolocation transformation engine."""

    def __init__(
        self,
        focal_length_mm: float = 35.0,
        sensor_width_mm: float = 36.0,
        sensor_height_mm: float = 24.0,
    ) -> None:
        self.focal_length_mm = focal_length_mm
        self.sensor_w_mm = sensor_width_mm
        self.sensor_h_mm = sensor_height_mm

    def pixel_to_world(
        self,
        pixel_x: float,
        pixel_y: float,
        image_w: int,
        image_h: int,
        uav_lat: float,
        uav_lon: float,
        uav_alt_m: float,
        gimbal_pitch_deg: float = -45.0,  # Pitch down
        gimbal_yaw_deg: float = 0.0,      # Relative to North
        gimbal_roll_deg: float = 0.0,
    ) -> tuple[float, float, float]:
        """
        Project pixel coordinate (pixel_x, pixel_y) to (lat, lon, alt) WGS84 position.
        Assumes flat-earth local terrain at alt = 0.0m.
        """
        # Calculate focal length in pixels
        fx = (self.focal_length_mm / self.sensor_w_mm) * image_w
        fy = (self.focal_length_mm / self.sensor_h_mm) * image_h

        # Convert pixel coords to normalized camera optical frame vector
        cx, cy = image_w / 2.0, image_h / 2.0
        x_cam = (pixel_x - cx) / fx
        y_cam = (pixel_y - cy) / fy
        z_cam = 1.0  # Forward direction along optical axis

        ray_cam = np.array([x_cam, y_cam, z_cam], dtype=np.float64)
        ray_cam /= np.linalg.norm(ray_cam)

        # Convert pitch/roll/yaw to Rotation Matrix
        pitch = math.radians(gimbal_pitch_deg)
        yaw = math.radians(gimbal_yaw_deg)
        roll = math.radians(gimbal_roll_deg)

        # Pitch rotation matrix (around X axis)
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(pitch), -math.sin(pitch)],
            [0, math.sin(pitch), math.cos(pitch)]
        ])

        # Yaw rotation matrix (around Z axis)
        Rz = np.array([
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw), math.cos(yaw), 0],
            [0, 0, 1]
        ])

        R = np.dot(Rz, Rx)
        ray_world = np.dot(R, ray_cam)

        # Ray-plane intersection with ground plane (z = 0)
        # ray_world = [dx, dy, dz]
        if abs(ray_world[2]) < 1e-6 or ray_world[2] >= 0:
            # Ray pointing skywards or horizon: fallback to UAV nadir coordinates
            return uav_lat, uav_lon, 0.0

        scale = -uav_alt_m / ray_world[2]
        north_m = ray_world[1] * scale
        east_m = ray_world[0] * scale

        # Convert North/East offsets to Delta Lat/Lon
        d_lat = (north_m / EARTH_RADIUS_METERS) * (180.0 / math.pi)
        d_lon = (east_m / (EARTH_RADIUS_METERS * math.cos(math.radians(uav_lat)))) * (180.0 / math.pi)

        target_lat = uav_lat + d_lat
        target_lon = uav_lon + d_lon

        return float(target_lat), float(target_lon), 0.0
