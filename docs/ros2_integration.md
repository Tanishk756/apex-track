# APEX-Track ROS2 Integration Guide

## Overview
APEX-Track includes a full ROS2 adapter node (`apex_track_ros`) designed for seamless deployment on autonomous UAV hardware running ROS2 Humble / Iron / Rolling.

## Published Topics & Subscriptions

### Subscriptions
- `/camera/image_raw` (`sensor_msgs/msg/Image`): Raw multi-spectral optical video feed.
- `/uav/telemetry` (`geometry_msgs/msg/PoseStamped`): Real-time flight controller positioning data for 3D coordinate projection.

### Publications
- `/apex/target_tracks` (`apex_track_msgs/msg/TrackArray`): Confirmed active target tracks with bounding boxes, velocity vectors, and threat classifications.
- `/apex/system_status` (`std_msgs/msg/String`): Hardware profile and engine operational state.

## Launching ROS2 Adapter

### Command Line Interface
```bash
# Sourcing ROS2 installation
source /opt/ros/humble/setup.bash

# Launching via apex-track CLI
apex-track ros2
```

### ROS2 Launch File
```bash
ros2 launch ros2/launch/apex_track.launch.py
```
