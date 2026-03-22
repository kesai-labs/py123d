from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


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

    # IPC write options
    ipc_max_batch_size: Optional[int] = None

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
