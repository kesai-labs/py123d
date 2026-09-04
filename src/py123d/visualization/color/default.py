from typing import Dict

from py123d.datatypes.detections.box_detection_label import DefaultBoxDetectionLabel
from py123d.datatypes.detections.traffic_light_detections import TrafficLightStatus
from py123d.datatypes.map_objects.map_layer_types import MapLayer
from py123d.datatypes.sensors.camera_segmentation_label import (
    DEFAULT_CAMERA_SEGMENTATION_RGB,
    DefaultCameraSegmentationLabel,
)
from py123d.datatypes.sensors.lidar_segmentation_label import DefaultLidarSegmentationLabel
from py123d.visualization.color.color import (
    BLACK,
    DARKER_GREY,
    ELLIS_5,
    LIGHT_GREY,
    TAB_10,
    WHITE,
    Color,
)
from py123d.visualization.color.config import PlotConfig

HEADING_MARKER_STYLE: str = "^"  # "^": triangle, "-": line
linewidth = 1.0

MAP_SURFACE_CONFIG: Dict[MapLayer, PlotConfig] = {
    MapLayer.LANE: PlotConfig(
        fill_color=LIGHT_GREY,
        fill_color_alpha=1.0,
        line_color=LIGHT_GREY,
        line_color_alpha=0.0,
        line_width=linewidth,
        line_style="-",
        zorder=1,
    ),
    MapLayer.LANE_GROUP: PlotConfig(
        fill_color=LIGHT_GREY,
        fill_color_alpha=1.0,
        line_color=LIGHT_GREY,
        line_color_alpha=0.0,
        line_width=linewidth,
        line_style="-",
        zorder=1,
    ),
    MapLayer.INTERSECTION: PlotConfig(
        fill_color=LIGHT_GREY,
        fill_color_alpha=1.0,
        line_color=LIGHT_GREY,
        line_color_alpha=0.0,
        line_width=linewidth,
        line_style="-",
        zorder=1,
    ),
    MapLayer.CROSSWALK: PlotConfig(
        fill_color=Color("#d0b9ca"),
        fill_color_alpha=1.0,
        line_color=Color("#d0b9ca"),
        line_color_alpha=0.0,
        line_width=linewidth,
        line_style="-",
        zorder=2,
    ),
    MapLayer.WALKWAY: PlotConfig(
        fill_color=Color("#d4d19e"),
        fill_color_alpha=1.0,
        line_color=Color("#d4d19e"),
        line_color_alpha=0.0,
        line_width=linewidth,
        line_style="-",
        zorder=1,
    ),
    MapLayer.CARPARK: PlotConfig(
        fill_color=Color("#b9d3b4"),
        fill_color_alpha=1.0,
        line_color=Color("#b9d3b4"),
        line_color_alpha=0.0,
        line_width=0.0,
        line_style="-",
        zorder=1,
    ),
    MapLayer.GENERIC_DRIVABLE: PlotConfig(
        fill_color=LIGHT_GREY,
        fill_color_alpha=1.0,
        line_color=LIGHT_GREY,
        line_color_alpha=0.0,
        line_width=linewidth,
        line_style="-",
        zorder=1,
    ),
    MapLayer.SHOULDER: PlotConfig(
        fill_color=Color("#c0c0c0"),
        fill_color_alpha=1.0,
        line_color=Color("#c0c0c0"),
        line_color_alpha=0.0,
        line_width=linewidth,
        line_style="-",
        zorder=1,
    ),
    MapLayer.NONE_LANE: PlotConfig(
        fill_color=Color("#cccbc4"),
        fill_color_alpha=1.0,
        line_color=Color("#cccbc4"),
        line_color_alpha=0.0,
        line_width=linewidth,
        line_style="-",
        zorder=1,
    ),
    MapLayer.STOP_ZONE: PlotConfig(
        fill_color=Color("#cd9595"),
        fill_color_alpha=1.0,
        line_color=Color("#cd9595"),
        line_color_alpha=0.0,
        line_width=linewidth,
        line_style="-",
        zorder=1,
    ),
    MapLayer.SPEED_BUMP: PlotConfig(
        fill_color=Color("#FFC400"),
        fill_color_alpha=1.0,
        line_color=Color("#FFC400"),
        line_color_alpha=0.0,
        line_width=linewidth,
        line_style="-",
        zorder=2,
    ),
}


BOX_DETECTION_CONFIG: Dict[DefaultBoxDetectionLabel, PlotConfig] = {
    # Vehicles
    DefaultBoxDetectionLabel.EGO: PlotConfig(
        fill_color=Color("#DE7061"),
        fill_color_alpha=1.0,
        line_color=BLACK,
        line_color_alpha=1.0,
        line_width=linewidth,
        line_style="-",
        marker_style=HEADING_MARKER_STYLE,
        zorder=4,
    ),
    DefaultBoxDetectionLabel.VEHICLE: PlotConfig(
        fill_color=Color("#699CDB"),
        fill_color_alpha=1.0,
        line_color=BLACK,
        line_color_alpha=1.0,
        line_width=linewidth,
        line_style="-",
        marker_style=HEADING_MARKER_STYLE,
        marker_size=1.0,
        zorder=3,
    ),
    DefaultBoxDetectionLabel.TRAIN: PlotConfig(
        fill_color=Color("#4e79a7"),
        fill_color_alpha=1.0,
        line_color=BLACK,
        line_color_alpha=1.0,
        line_width=linewidth,
        line_style="-",
        marker_style=HEADING_MARKER_STYLE,
        marker_size=1.0,
        zorder=3,
    ),
    # VRUs
    DefaultBoxDetectionLabel.TWO_WHEELER: PlotConfig(
        fill_color=Color("#76b7b2"),
        fill_color_alpha=1.0,
        line_color=BLACK,
        line_color_alpha=1.0,
        line_width=linewidth,
        line_style="-",
        marker_style=HEADING_MARKER_STYLE,
        marker_size=1.0,
        zorder=2,
    ),
    DefaultBoxDetectionLabel.PERSON: PlotConfig(
        fill_color=Color("#b07aa1"),
        fill_color_alpha=1.0,
        line_color=BLACK,
        line_color_alpha=1.0,
        line_width=linewidth,
        line_style="-",
        marker_style=None,
        marker_size=1.0,
        zorder=2,
    ),
    DefaultBoxDetectionLabel.ANIMAL: PlotConfig(
        fill_color=Color("#9467bd"),
        fill_color_alpha=1.0,
        line_color=BLACK,
        line_color_alpha=1.0,
        line_width=linewidth,
        line_style="-",
        marker_style=None,
        marker_size=1.0,
        zorder=2,
    ),
    # Traffic Control
    DefaultBoxDetectionLabel.TRAFFIC_SIGN: PlotConfig(
        fill_color=Color("#E38C47"),
        fill_color_alpha=1.0,
        line_color=BLACK,
        line_color_alpha=1.0,
        line_width=linewidth,
        line_style="-",
        marker_style=None,
        zorder=2,
    ),
    DefaultBoxDetectionLabel.TRAFFIC_CONE: PlotConfig(
        fill_color=Color("#E38C47"),
        fill_color_alpha=1.0,
        line_color=BLACK,
        line_color_alpha=1.0,
        line_width=linewidth,
        line_style="-",
        marker_style=None,
        zorder=2,
    ),
    DefaultBoxDetectionLabel.TRAFFIC_LIGHT: PlotConfig(
        fill_color=Color("#E38C47"),
        fill_color_alpha=1.0,
        line_color=BLACK,
        line_color_alpha=1.0,
        line_width=linewidth,
        line_style="-",
        marker_style=None,
        zorder=2,
    ),
    # Other Obstacles
    DefaultBoxDetectionLabel.BARRIER: PlotConfig(
        fill_color=Color("#E38C47"),
        fill_color_alpha=1.0,
        line_color=BLACK,
        line_color_alpha=1.0,
        line_width=linewidth,
        line_style="-",
        marker_style=None,
        zorder=2,
    ),
    DefaultBoxDetectionLabel.GENERIC_OBJECT: PlotConfig(
        fill_color=Color("#E38C47"),
        fill_color_alpha=1.0,
        line_color=BLACK,
        line_color_alpha=1.0,
        line_width=linewidth,
        line_style="-",
        marker_style=None,
        zorder=2,
    ),
    # Miscellaneous
    DefaultBoxDetectionLabel.OTHER: PlotConfig(
        fill_color=Color("#9c8aa5"),
        fill_color_alpha=1.0,
        line_color=BLACK,
        line_color_alpha=1.0,
        line_width=linewidth,
        line_style="-",
        marker_style=None,
        zorder=2,
    ),
}

EGO_VEHICLE_CONFIG: PlotConfig = PlotConfig(
    fill_color=ELLIS_5[0],
    fill_color_alpha=1.0,
    line_color=BLACK,
    line_color_alpha=1.0,
    line_width=linewidth,
    line_style="-",
    marker_style=HEADING_MARKER_STYLE,
    zorder=4,
)

CENTERLINE_CONFIG: PlotConfig = PlotConfig(
    fill_color=WHITE,
    fill_color_alpha=1.0,
    line_color=DARKER_GREY,
    line_color_alpha=1.0,
    line_width=linewidth,
    line_style="--",
    zorder=3,
)

ROUTE_CONFIG: PlotConfig = PlotConfig(
    fill_color=Color("#f2c6c0ff"),
    fill_color_alpha=1.0,
    line_color=Color("#f2c6c0ff"),
    line_color_alpha=0.0,
    line_width=linewidth,
    line_style="-",
    zorder=2,
)

MARK_CONFIG: PlotConfig = PlotConfig(
    fill_color=TAB_10[6],
    fill_color_alpha=1.0,
    line_color=TAB_10[6],
    line_color_alpha=1.0,
    line_width=linewidth,
    line_style="-",
    marker_style=HEADING_MARKER_STYLE,
    marker_size=1.0,
    zorder=3,
)


# ----------------------------------------------------------------------------------------------------------------------
# Semantic segmentation colors
# ----------------------------------------------------------------------------------------------------------------------
# The unified taxonomies are nuScenes-lidarseg style (LiDAR) and Cityscapes style (camera), but both are
# colored with the canonical Cityscapes palette [1]_ so segmentation overlays use familiar, recognizable
# colors (road = purple, vegetation = olive green, person = crimson, vehicle = dark blue, ...).
# [1] https://github.com/mcordts/cityscapesScripts (labels.py)

# Cityscapes-palette colors, keyed by the *unified* LiDAR label (raw dataset ids reach these via to_default()).
DEFAULT_LIDAR_SEGMENTATION_COLORS: Dict[DefaultLidarSegmentationLabel, Color] = {
    DefaultLidarSegmentationLabel.IGNORE: BLACK,
    DefaultLidarSegmentationLabel.VEHICLE: Color.from_rgb((0, 0, 142)),  # Cityscapes car
    DefaultLidarSegmentationLabel.TWO_WHEELER: Color.from_rgb((119, 11, 32)),  # Cityscapes bicycle
    DefaultLidarSegmentationLabel.PERSON: Color.from_rgb((220, 20, 60)),  # Cityscapes person
    DefaultLidarSegmentationLabel.RIDER: Color.from_rgb((255, 0, 0)),  # Cityscapes rider
    DefaultLidarSegmentationLabel.DRIVABLE_SURFACE: Color.from_rgb((128, 64, 128)),  # Cityscapes road
    DefaultLidarSegmentationLabel.SIDEWALK: Color.from_rgb((244, 35, 232)),  # Cityscapes sidewalk
    DefaultLidarSegmentationLabel.TERRAIN: Color.from_rgb((152, 251, 152)),  # Cityscapes terrain
    DefaultLidarSegmentationLabel.VEGETATION: Color.from_rgb((107, 142, 35)),  # Cityscapes vegetation
    DefaultLidarSegmentationLabel.MANMADE: Color.from_rgb((70, 70, 70)),  # Cityscapes building
    DefaultLidarSegmentationLabel.TRAFFIC_CONE: Color.from_rgb((250, 170, 30)),  # Cityscapes traffic light
    DefaultLidarSegmentationLabel.BARRIER: Color.from_rgb((102, 102, 156)),  # Cityscapes wall
    DefaultLidarSegmentationLabel.OTHER: DARKER_GREY,
}

# Cityscapes-palette colors, keyed by the unified camera label. Derived from the canonical RGB tuples that live in
# the data layer (DEFAULT_CAMERA_SEGMENTATION_RGB), which are the single source of truth — see that constant's
# docstring for why the values live there rather than here.
DEFAULT_CAMERA_SEGMENTATION_COLORS: Dict[DefaultCameraSegmentationLabel, Color] = {
    label: Color.from_rgb(rgb) for label, rgb in DEFAULT_CAMERA_SEGMENTATION_RGB.items()
}


TRAFFIC_LIGHT_CONFIG: Dict[TrafficLightStatus, PlotConfig] = {
    TrafficLightStatus.RED: PlotConfig(
        line_color=TAB_10[3],
        line_color_alpha=1.0,
        line_width=1.5,
        line_style="--",
        zorder=3,
    ),
    TrafficLightStatus.YELLOW: PlotConfig(
        line_color=TAB_10[1],
        line_color_alpha=1.0,
        line_width=1.5,
        line_style="--",
        zorder=3,
    ),
    TrafficLightStatus.GREEN: PlotConfig(
        line_color=TAB_10[2],
        line_color_alpha=1.0,
        line_width=1.5,
        line_style="--",
        zorder=3,
    ),
    TrafficLightStatus.OFF: PlotConfig(
        line_color=TAB_10[5],
        line_color_alpha=1.0,
        line_width=linewidth,
        line_style="--",
        zorder=3,
    ),
    TrafficLightStatus.UNKNOWN: PlotConfig(
        line_color=TAB_10[5],
        line_color_alpha=1.0,
        line_width=1.5,
        line_style="--",
        zorder=3,
    ),
}
