from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

from py123d.visualization.color.color import ELLIS_5
from py123d.visualization.viser.camera_gui_controller import CameraGuiConfig
from py123d.visualization.viser.elements.box_detections_se3_element import DetectionConfig
from py123d.visualization.viser.elements.camera_frustum_element import CameraFrustumConfig
from py123d.visualization.viser.elements.ego_state_se3_element import EgoConfig
from py123d.visualization.viser.elements.lidar_element import LidarConfig
from py123d.visualization.viser.elements.map_element import MapConfig
from py123d.visualization.viser.playback_controller import PlaybackConfig
from py123d.visualization.viser.render_controller import RenderConfig

CONTRAST_COLOR = (255, 255, 255)


@dataclass
class ServerConfig:
    host: str = "localhost"
    port: int = 8080
    label: str = "123D Viser Server"
    verbose: bool = True


@dataclass
class ThemeConfig:
    control_layout: Literal["floating", "collapsible", "fixed"] = "floating"
    control_width: Literal["small", "medium", "large"] = "large"
    dark_mode: bool = False
    show_logo: bool = True
    show_share_button: bool = True
    brand_color: Optional[Tuple[int, int, int]] = ELLIS_5[0].rgb


_SUB_CONFIG_FIELDS = {
    "server": ServerConfig,
    "theme": ThemeConfig,
    "playback": PlaybackConfig,
    "map": MapConfig,
    "ego": EgoConfig,
    "detection": DetectionConfig,
    "camera_frustum": CameraFrustumConfig,
    "camera_gui": CameraGuiConfig,
    "lidar": LidarConfig,
    "render": RenderConfig,
}


@dataclass
class ViserConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    theme: ThemeConfig = field(default_factory=ThemeConfig)
    playback: PlaybackConfig = field(default_factory=PlaybackConfig)
    map: MapConfig = field(default_factory=MapConfig)
    ego: EgoConfig = field(default_factory=EgoConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    camera_frustum: CameraFrustumConfig = field(default_factory=CameraFrustumConfig)
    camera_gui: CameraGuiConfig = field(default_factory=CameraGuiConfig)
    lidar: LidarConfig = field(default_factory=LidarConfig)
    render: RenderConfig = field(default_factory=RenderConfig)

    def __post_init__(self):
        # Hydra instantiate with _convert_='all' produces plain dicts for nested configs.
        # Convert them to the proper dataclass types.
        for field_name, config_cls in _SUB_CONFIG_FIELDS.items():
            value = getattr(self, field_name)
            if isinstance(value, dict):
                setattr(self, field_name, config_cls(**value))
