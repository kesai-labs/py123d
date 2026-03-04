import traceback
from collections import defaultdict
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as geom

from py123d.api import MapAPI, SceneAPI
from py123d.datatypes.detections.box_detection_label import DefaultBoxDetectionLabel
from py123d.datatypes.detections.box_detections import BoxDetectionsSE3
from py123d.datatypes.detections.traffic_light_detections import TrafficLightDetections
from py123d.datatypes.map_objects.map_layer_types import MapLayer
from py123d.datatypes.map_objects.map_objects import Lane
from py123d.datatypes.vehicle_state.ego_state import EgoStateSE2, EgoStateSE3
from py123d.geometry import BoundingBoxSE2, BoundingBoxSE3, Point2D, PoseSE2Index, Vector2D
from py123d.geometry.transform.transform_se2 import translate_se2_along_body_frame
from py123d.visualization.color.config import PlotConfig
from py123d.visualization.color.default import (
    BOX_DETECTION_CONFIG,
    CENTERLINE_CONFIG,
    EGO_VEHICLE_CONFIG,
    MAP_SURFACE_CONFIG,
    TRAFFIC_LIGHT_CONFIG,
)
from py123d.visualization.matplotlib.utils import (
    add_shapely_linestring_to_ax,
    add_shapely_linestrings_to_ax,
    add_shapely_polygons_to_ax,
    get_pose_triangle,
    shapely_geometry_local_coords,
)

# from py123d.visualization.matplotlib.zebra import visualize_crosswalk_stripes


def add_scene_on_ax(ax: plt.Axes, scene: SceneAPI, iteration: int = 0, radius: float = 80) -> plt.Axes:
    ego_vehicle_state = scene.get_ego_state_se3_at_iteration(iteration)
    box_detections = scene.get_box_detections_at_iteration(iteration)
    traffic_light_detections = scene.get_traffic_light_detections_at_iteration(iteration)
    map_api = scene.get_map_api()

    assert ego_vehicle_state is not None, "Ego vehicle state is required to plot the scene."
    point_2d = ego_vehicle_state.bounding_box_se2.center_se2.pose_se2.point_2d
    if map_api is not None:
        add_default_map_on_ax(ax, map_api, point_2d, radius=radius)
        if traffic_light_detections is not None:
            add_traffic_lights_to_ax(ax, traffic_light_detections, map_api)

    add_box_detections_to_ax(ax, box_detections)
    add_ego_vehicle_to_ax(ax, ego_vehicle_state)

    ax.set_xlim(point_2d.x - radius, point_2d.x + radius)
    ax.set_ylim(point_2d.y - radius, point_2d.y + radius)
    ax.set_aspect("equal", adjustable="box")
    return ax


def add_default_map_on_ax(
    ax: plt.Axes,
    map_api: MapAPI,
    point_2d: Point2D,
    radius: float,
    route_lane_group_ids: Optional[List[int]] = None,
) -> None:
    layers: List[MapLayer] = [
        MapLayer.LANE,
        MapLayer.LANE_GROUP,
        MapLayer.GENERIC_DRIVABLE,
        MapLayer.CARPARK,
        MapLayer.CROSSWALK,
        MapLayer.INTERSECTION,
        MapLayer.WALKWAY,
    ]
    x_min, x_max = point_2d.x - radius, point_2d.x + radius
    y_min, y_max = point_2d.y - radius, point_2d.y + radius
    patch = geom.box(x_min, y_min, x_max, y_max)
    map_objects_dict = map_api.query(geometry=patch, layers=layers)  # , predicate="intersects")

    for layer, map_objects in map_objects_dict.items():
        try:
            # if layer == MapLayer.CROSSWALK:
            #     for map_object in map_objects:
            #         visualize_crosswalk_stripes(map_object.shapely_polygon, ax)

            if layer in [
                MapLayer.LANE_GROUP,
                MapLayer.GENERIC_DRIVABLE,
                MapLayer.CARPARK,
                MapLayer.CROSSWALK,
                MapLayer.INTERSECTION,
                MapLayer.WALKWAY,
            ]:
                polygons = []
                for map_object in map_objects:
                    polygons.append(map_object.shapely_polygon)
                if len(polygons) > 0:
                    add_shapely_polygons_to_ax(
                        ax,
                        polygons,
                        MAP_SURFACE_CONFIG[layer],
                        label=layer.serialize(),
                    )

            if layer in [MapLayer.LANE]:
                lines = []
                for map_object in map_objects:
                    map_object: Lane
                    lines.append(map_object.centerline.linestring)
                if len(lines) > 0:
                    add_shapely_linestrings_to_ax(
                        ax,
                        lines,
                        CENTERLINE_CONFIG,
                        label=layer.serialize(),
                    )

        except Exception:
            print(f"Error adding map object of type {layer.name}")
            traceback.print_exc()

    ax.set_title(f"Map: {map_api.location}")


def add_box_detections_to_ax(ax: plt.Axes, box_detections: BoxDetectionsSE3) -> None:
    boxes_per_type: Dict[DefaultBoxDetectionLabel, List[BoundingBoxSE2]] = defaultdict(list)
    for box_detection in box_detections:
        boxes_per_type[box_detection.metadata.default_label].append(box_detection.bounding_box_se2)

    for box_detection_type, bounding_boxes_se2 in boxes_per_type.items():
        plot_config = BOX_DETECTION_CONFIG[box_detection_type]
        add_bounding_boxes_to_ax(ax, bounding_boxes_se2, plot_config)  # type: ignore


def add_ego_vehicle_to_ax(ax: plt.Axes, ego_vehicle_state: Union[EgoStateSE3, EgoStateSE2]) -> None:
    add_bounding_boxes_to_ax(ax, [ego_vehicle_state.bounding_box_se2], EGO_VEHICLE_CONFIG)


def add_traffic_lights_to_ax(ax: plt.Axes, traffic_light_detections: TrafficLightDetections, map_api: MapAPI) -> None:
    for traffic_light_detection in traffic_light_detections:
        lane = map_api.get_map_object(traffic_light_detection.lane_id, MapLayer.LANE)
        assert isinstance(lane, Lane), f"Lane with id {traffic_light_detection.lane_id} not found."
        if lane is not None:
            add_shapely_linestring_to_ax(
                ax,
                lane.centerline.linestring,
                TRAFFIC_LIGHT_CONFIG[traffic_light_detection.status],
            )
        else:
            raise ValueError(f"Lane with id {traffic_light_detection.lane_id} not found in map {map_api.location}.")


def add_bounding_boxes_to_ax(
    ax: plt.Axes,
    bounding_boxes: List[Union[BoundingBoxSE2, BoundingBoxSE3]],
    plot_config: PlotConfig,
) -> None:
    add_shapely_polygons_to_ax(
        ax,
        [box.shapely_polygon for box in bounding_boxes],
        plot_config,
    )
    markers: List[geom.Polygon] = []
    for bounding_box in bounding_boxes:
        if plot_config.marker_style is not None:
            assert plot_config.marker_style in ["-", "^"], f"Unknown marker style: {plot_config.marker_style}"
            if plot_config.marker_style == "-":
                center_se2 = bounding_box.center_se2
                arrow = np.zeros((2, 2), dtype=np.float64)
                arrow[0] = center_se2.point_2d.array
                arrow[1] = translate_se2_along_body_frame(
                    center_se2,
                    Vector2D(bounding_box.length / 2.0 + 0.5, 0.0),
                ).array[PoseSE2Index.XY]
                ax.plot(
                    arrow[:, 0],
                    arrow[:, 1],
                    color=plot_config.line_color.hex,
                    alpha=plot_config.line_color_alpha,
                    linewidth=plot_config.line_width,
                    zorder=plot_config.zorder,
                    linestyle=plot_config.line_style,
                )
            elif plot_config.marker_style == "^":
                min_extent = min(bounding_box.length, bounding_box.width)
                marker_size = min(plot_config.marker_size, min_extent)
                marker_polygon = get_pose_triangle(marker_size)
                global_marker_polygon = shapely_geometry_local_coords(marker_polygon, bounding_box.center_se2)
                markers.append(global_marker_polygon)  # type: ignore
            else:
                raise ValueError(f"Unknown marker style: {plot_config.marker_style}")
    add_shapely_polygons_to_ax(ax, markers, plot_config, disable_smoothing=True)
