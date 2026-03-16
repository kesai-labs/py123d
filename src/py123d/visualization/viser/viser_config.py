from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple, Union

from py123d.common.utils.enums import SerialIntEnum
from py123d.datatypes.sensors.base_camera import ALL_FISHEYE_MEI_CAMERA_IDS, ALL_PINHOLE_CAMERA_IDS, CameraID
from py123d.datatypes.sensors.lidar import LidarID
from py123d.visualization.color.color import ELLIS_5


@dataclass
class ViserConfig:
    # Server
    server_host: str = "localhost"
    server_port: int = 8080
    server_label: str = "123D Viser Server"
    server_verbose: bool = True

    # Theme
    theme_control_layout: Literal["floating", "collapsible", "fixed"] = "floating"
    theme_control_width: Literal["small", "medium", "large"] = "large"
    theme_dark_mode: bool = False
    theme_show_logo: bool = True
    theme_show_share_button: bool = True
    theme_brand_color: Optional[Tuple[int, int, int]] = ELLIS_5[4].rgb

    # Play Controls
    is_playing: bool = False
    playback_speed: float = 1.0  # Multiplier for real-time speed

    # Map
    map_visible: bool = True
    map_radius: float = 200.0  # [m]
    map_non_road_z_offset: float = 0.1  # small z-translation to place crosswalks, parking, etc. on top of the road
    map_requery: bool = True  # Re-query map when ego vehicle moves out of current map bounds

    # Bounding boxes
    bounding_box_visible: bool = True
    bounding_box_type: Literal["mesh", "lines"] = "mesh"
    bounding_box_line_width: float = 4.0

    # Pinhole Cameras
    # -> Frustum
    camera_frustum_visible: bool = True
    camera_frustum_types: List[CameraID] = field(default_factory=lambda: ALL_PINHOLE_CAMERA_IDS.copy())
    camera_frustum_scale: float = 1.0
    camera_frustum_image_scale: int = 4  # Downscale denominator for the camera image

    # -> GUI
    camera_gui_visible: bool = True
    camera_gui_types: List[CameraID] = field(default_factory=lambda: [CameraID.PCAM_F0].copy())
    camera_gui_image_scale: int = 4  # Downscale denominator for the camera image shown in the GUI

    # Fisheye MEI Cameras
    # -> Frustum
    fisheye_frustum_visible: bool = True
    fisheye_mei_camera_frustum_visible: bool = True
    fisheye_mei_camera_frustum_types: List[CameraID] = field(default_factory=lambda: ALL_FISHEYE_MEI_CAMERA_IDS.copy())
    fisheye_frustum_scale: float = 1.0
    fisheye_frustum_image_scale: int = 4  # Downscale denominator for the camera image shown on the frustum

    # Lidar
    lidar_visible: bool = True
    lidar_ids: List[LidarID] = field(default_factory=lambda: [LidarID.LIDAR_MERGED])
    lidar_point_size: float = 0.05
    lidar_point_shape: Literal["square", "diamond", "circle", "rounded", "sparkle"] = "circle"
    lidar_point_color: Literal["none", "distance", "intensity", "channel", "timestamps", "range", "elongation"] = "none"

    # internal use
    _force_map_update: bool = False

    def __post_init__(self):
        def _resolve_enum_arguments(
            serial_enum_cls: SerialIntEnum, input: Optional[List[Union[int, str, SerialIntEnum]]]
        ) -> List[SerialIntEnum]:
            if input is None:
                return None
            assert isinstance(input, list), f"input must be a list of {serial_enum_cls.__name__}"
            return [serial_enum_cls.from_arbitrary(value) for value in input]

        self.camera_frustum_types = _resolve_enum_arguments(
            CameraID,
            self.camera_frustum_types,
        )
        self.camera_gui_types = _resolve_enum_arguments(
            CameraID,
            self.camera_gui_types,
        )
        self.fisheye_mei_camera_frustum_types = _resolve_enum_arguments(
            CameraID,
            self.fisheye_mei_camera_frustum_types,
        )
        self.lidar_ids = _resolve_enum_arguments(
            LidarID,
            self.lidar_ids,
        )

        self.camera_gui_image_scale = int(self.camera_gui_image_scale)
        self.camera_frustum_image_scale = int(self.camera_frustum_image_scale)
        self.fisheye_frustum_image_scale = int(self.fisheye_frustum_image_scale)
