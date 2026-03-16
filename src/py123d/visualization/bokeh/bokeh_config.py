from dataclasses import dataclass


@dataclass
class BokehConfig:
    """Configuration for the Bokeh scene viewer."""

    # Server
    server_port: int = 5006

    # BEV
    bev_radius: float = 80.0

    # Layers
    show_map: bool = True
    show_lidar: bool = True
    show_detections: bool = True
