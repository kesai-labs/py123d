from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from py123d.common.io.lidar.point_cloud_codec_config import PointCloudCodecConfig


@dataclass
class LogWriterConfig:
    force_log_conversion: bool = False
    force_map_conversion: bool = False
    async_conversion: bool = False

    exclude_modality_keys: set[str] = field(default_factory=set)
    exclude_modality_types: set[str] = field(default_factory=set)

    # Cameras
    camera_store_option: Literal["path", "jpeg_binary", "png_binary", "mp4"] = "path"

    # Lidars
    lidar_store_option: Literal["path", "binary"] = "path"
    lidar_codec: Optional[Literal["laz", "draco", "ipc_zstd", "ipc_lz4", "ipc"]] = None
    lidar_codec_config: PointCloudCodecConfig = field(default_factory=PointCloudCodecConfig)

    # Radars
    radar_store_option: Literal["path", "binary"] = "path"
    radar_codec: Optional[Literal["laz", "draco", "ipc_zstd", "ipc_lz4", "ipc"]] = None
    radar_codec_config: PointCloudCodecConfig = field(default_factory=PointCloudCodecConfig)

    # IPC write options
    # Maximum buffered size, in bytes, before a modality writer flushes a record batch. None keeps
    # each writer's built-in default (DEFAULT_MAX_BATCH_BYTES). Lower this when converting many logs
    # in parallel, since the budget applies per open modality writer.
    ipc_max_batch_bytes: Optional[int] = None

    # Ego
    infer_ego_dynamics: bool = False

    # Route (driven-route polyline + per-frame progress, derived from ego odometry at close time)
    write_route: bool = True
    route_resolution_m: float = 1.0

    # Boxes
    infer_box_dynamics: bool = False

    def __post_init__(self):
        assert self.camera_store_option in {
            "path",
            "jpeg_binary",
            "png_binary",
            "mp4",
        }, f"Invalid camera store option, got {self.camera_store_option}."

        assert self.lidar_store_option in {
            "path",
            "binary",
        }, f"Invalid Lidar store option, got {self.lidar_store_option}."

        if self.lidar_store_option == "binary":
            assert self.lidar_codec in {
                "laz",
                "draco",
                "ipc_zstd",
                "ipc_lz4",
                "ipc",
            }, f"Invalid Lidar codec, got {self.lidar_codec}."

        assert self.radar_store_option in {
            "path",
            "binary",
        }, f"Invalid Radar store option, got {self.radar_store_option}."

        if self.radar_store_option == "binary":
            assert self.radar_codec in {
                "laz",
                "draco",
                "ipc_zstd",
                "ipc_lz4",
                "ipc",
            }, f"Invalid Radar codec, got {self.radar_codec}."

        assert self.route_resolution_m > 0.0, f"route_resolution_m must be > 0, got {self.route_resolution_m}."
