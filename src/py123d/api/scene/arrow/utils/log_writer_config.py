from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from py123d.common.io.lidar.point_cloud_codec_config import PointCloudCodecConfig
from py123d.common.io.lidar.point_cloud_codec_registry import point_cloud_codec_names


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
    # A name of py123d.common.io.lidar.point_cloud_codec_registry: "laz", "draco", "ipc_zstd", "ipc_lz4", "ipc",
    # "kedec".
    lidar_codec: Optional[str] = None
    lidar_codec_config: PointCloudCodecConfig = field(default_factory=PointCloudCodecConfig)
    # Threads per lidar writer that encode binary blobs in parallel (rows keep their order); 1 encodes in the
    # converting thread.
    encode_threads: int = 1

    # Radars
    radar_store_option: Literal["path", "binary"] = "path"
    # A registered codec that radars may use: "laz", "draco", "ipc_zstd", "ipc_lz4", "ipc".
    radar_codec: Optional[str] = None
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
            assert self.lidar_codec in point_cloud_codec_names("lidar"), f"Invalid Lidar codec, got {self.lidar_codec}."

        assert self.radar_store_option in {
            "path",
            "binary",
        }, f"Invalid Radar store option, got {self.radar_store_option}."

        if self.radar_store_option == "binary":
            assert self.radar_codec in point_cloud_codec_names("radar"), f"Invalid Radar codec, got {self.radar_codec}."

        # Hydra passes nested settings without a _target_ as plain mappings.
        if isinstance(self.lidar_codec_config, dict):
            self.lidar_codec_config = PointCloudCodecConfig(**_tuple_values(self.lidar_codec_config))
        if isinstance(self.radar_codec_config, dict):
            self.radar_codec_config = PointCloudCodecConfig(**_tuple_values(self.radar_codec_config))
        assert self.encode_threads >= 1, f"encode_threads must be >= 1, got {self.encode_threads}."

        assert self.route_resolution_m > 0.0, f"route_resolution_m must be > 0, got {self.route_resolution_m}."


def _tuple_values(settings: dict) -> dict:
    """The settings with list values as tuples (YAML gives lists, the frozen codec config takes tuples)."""
    return {key: tuple(value) if isinstance(value, list) else value for key, value in settings.items()}
