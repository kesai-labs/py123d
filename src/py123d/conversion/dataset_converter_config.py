from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class DatasetConverterConfig:
    force_log_conversion: bool = False
    force_map_conversion: bool = False

    # Map
    include_map: bool = False

    # Ego
    include_ego: bool = False

    # Box Detections
    include_box_detections: bool = False
    include_box_lidar_points: bool = False

    # Traffic Lights
    include_traffic_lights: bool = False

    # Pinhole Cameras
    include_pinhole_cameras: bool = False
    pinhole_camera_store_option: Literal["path", "jpeg_binary", "png_binary"] = "path"

    # Fisheye MEI Cameras
    include_fisheye_mei_cameras: bool = False
    fisheye_mei_camera_store_option: Literal["path", "jpeg_binary", "png_binary"] = "path"

    # Lidars
    include_lidars: bool = False
    lidar_store_option: Literal["path", "binary"] = "path"
    lidar_point_cloud_codec: Optional[Literal["laz", "draco", "ipc_zstd", "ipc_lz4", "ipc"]] = None
    lidar_point_feature_codec: Optional[Literal["ipc_zstd", "ipc_lz4", "ipc"]] = None  # None drops features.

    # Scenario tag / Route
    # NOTE: These are only supported for nuPlan. Consider removing or expanding support.
    include_scenario_tags: bool = False
    include_route: bool = False

    def __post_init__(self):
        assert self.pinhole_camera_store_option in {
            "path",
            "jpeg_binary",
            "png_binary",
        }, f"Invalid Pinhole camera store option, got {self.pinhole_camera_store_option}."

        assert self.fisheye_mei_camera_store_option in {
            "path",
            "jpeg_binary",
            "png_binary",
        }, f"Invalid Fisheye MEI camera store option, got {self.fisheye_mei_camera_store_option}."

        assert self.lidar_store_option in {
            "path",
            "binary",
        }, f"Invalid Lidar store option, got {self.lidar_store_option}."

        if self.lidar_store_option == "binary":
            assert self.lidar_point_cloud_codec in {
                "laz",
                "draco",
                "ipc_zstd",
                "ipc_lz4",
                "ipc",
            }, f"Invalid Lidar point cloud codec, got {self.lidar_point_cloud_codec}."
            assert self.lidar_point_feature_codec is None or self.lidar_point_feature_codec in {
                "ipc_zstd",
                "ipc_lz4",
                "ipc",
            }, f"Invalid Lidar point feature codec, got {self.lidar_point_feature_codec}."
