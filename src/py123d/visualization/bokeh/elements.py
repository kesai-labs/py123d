"""Helper functions to extract Bokeh-renderable data from py123d scene objects."""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import shapely.geometry as geom

from py123d.api import MapAPI, SceneAPI
from py123d.datatypes.detections.box_detection_label import DefaultBoxDetectionLabel
from py123d.datatypes.detections.box_detections import BoxDetectionsSE3
from py123d.datatypes.map_objects.map_layer_types import MapLayer, StopZoneType
from py123d.datatypes.map_objects.map_objects import Lane
from py123d.datatypes.vehicle_state.ego_state import EgoStateSE3
from py123d.geometry import BoundingBoxSE2
from py123d.visualization.color.default import BOX_DETECTION_CONFIG, EGO_VEHICLE_CONFIG, MAP_SURFACE_CONFIG


def _polygon_to_xy(polygon: geom.Polygon) -> Tuple[List[float], List[float]]:
    """Extract x, y coordinate lists from a shapely Polygon exterior."""
    coords = np.asarray(polygon.exterior.coords)[:, :2]
    return coords[:, 0].tolist(), coords[:, 1].tolist()


def _multipolygon_to_xy(
    polygons: List[geom.Polygon],
) -> Tuple[List[List[float]], List[List[float]]]:
    """Convert a list of shapely polygons to lists-of-lists suitable for Bokeh patches."""
    xs: List[List[float]] = []
    ys: List[List[float]] = []
    for poly in polygons:
        if isinstance(poly, geom.MultiPolygon):
            for p in poly.geoms:
                x, y = _polygon_to_xy(p)
                xs.append(x)
                ys.append(y)
        elif isinstance(poly, geom.Polygon):
            x, y = _polygon_to_xy(poly)
            xs.append(x)
            ys.append(y)
    return xs, ys


def get_map_data(
    map_api: MapAPI,
    center_x: float,
    center_y: float,
    radius: float,
) -> Dict[str, Dict[str, Any]]:
    """Query map layers and return patch data keyed by layer name.

    Returns a dict: layer_name -> {"xs": [...], "ys": [...], "color": hex, "alpha": float}
    Also returns centerlines separately as multi_line data.
    """
    patch = geom.box(center_x - radius, center_y - radius, center_x + radius, center_y + radius)
    polygon_layers = [
        MapLayer.LANE_GROUP,
        MapLayer.GENERIC_DRIVABLE,
        MapLayer.CARPARK,
        MapLayer.CROSSWALK,
        MapLayer.INTERSECTION,
        MapLayer.WALKWAY,
        MapLayer.STOP_ZONE,
    ]
    layers = polygon_layers + [MapLayer.LANE]

    result: Dict[str, Dict[str, Any]] = {}
    try:
        map_objects_dict = map_api.query(geometry=patch, layers=layers)
    except Exception:
        return result

    has_no_lane_groups = len(map_objects_dict.get(MapLayer.LANE_GROUP, [])) == 0

    for layer in polygon_layers:
        map_objects = map_objects_dict.get(layer, [])
        if not map_objects:
            continue

        if layer == MapLayer.STOP_ZONE:
            map_objects = [mo for mo in map_objects if mo.stop_zone_type != StopZoneType.TURN_STOP]

        polygons = [mo.shapely_polygon for mo in map_objects]
        if polygons:
            config = MAP_SURFACE_CONFIG[layer]
            xs, ys = _multipolygon_to_xy(polygons)
            result[f"map_{layer.serialize()}"] = {
                "xs": xs,
                "ys": ys,
                "color": config.fill_color.hex,
                "alpha": config.fill_color_alpha,
            }

    # Lanes: centerlines + optionally lane polygons if no lane groups
    lane_objects = map_objects_dict.get(MapLayer.LANE, [])
    if lane_objects:
        # Centerlines as multi_line
        centerline_xs: List[List[float]] = []
        centerline_ys: List[List[float]] = []
        for mo in lane_objects:
            mo: Lane
            coords = np.asarray(mo.centerline.linestring.coords)[:, :2]
            centerline_xs.append(coords[:, 0].tolist())
            centerline_ys.append(coords[:, 1].tolist())
        result["map_centerlines"] = {
            "xs": centerline_xs,
            "ys": centerline_ys,
            "type": "lines",
            "color": "#787878",
            "alpha": 1.0,
        }

        if has_no_lane_groups:
            polygons = [mo.shapely_polygon for mo in lane_objects]
            config = MAP_SURFACE_CONFIG[MapLayer.LANE]
            xs, ys = _multipolygon_to_xy(polygons)
            result["map_lane_surface"] = {
                "xs": xs,
                "ys": ys,
                "color": config.fill_color.hex,
                "alpha": config.fill_color_alpha,
            }

    return result


def get_ego_data(ego_state: EgoStateSE3) -> Dict[str, Any]:
    """Return patch data for the ego vehicle bounding box."""
    bb = ego_state.bounding_box_se2
    poly = bb.shapely_polygon
    x, y = _polygon_to_xy(poly)
    return {
        "xs": [x],
        "ys": [y],
        "color": EGO_VEHICLE_CONFIG.fill_color.hex,
        "line_color": EGO_VEHICLE_CONFIG.line_color.hex,
        "alpha": EGO_VEHICLE_CONFIG.fill_color_alpha,
    }


def get_detection_data(
    box_detections: Optional[BoxDetectionsSE3],
) -> Dict[str, Dict[str, Any]]:
    """Return patch data for box detections, grouped by label type."""
    result: Dict[str, Dict[str, Any]] = {}
    if box_detections is None:
        return result

    boxes_per_type: Dict[DefaultBoxDetectionLabel, List[BoundingBoxSE2]] = defaultdict(list)
    for det in box_detections:
        boxes_per_type[det.attributes.default_label].append(det.bounding_box_se2)

    for label, bboxes in boxes_per_type.items():
        if label not in BOX_DETECTION_CONFIG:
            continue
        config = BOX_DETECTION_CONFIG[label]
        polygons = [bb.shapely_polygon for bb in bboxes]
        xs, ys = _multipolygon_to_xy(polygons)
        result[label.serialize()] = {
            "xs": xs,
            "ys": ys,
            "color": config.fill_color.hex,
            "line_color": config.line_color.hex,
            "alpha": config.fill_color_alpha,
        }

    return result


def get_lidar_bev_data(
    scene: SceneAPI,
    iteration: int,
    ego_state: EgoStateSE3,
) -> Optional[Dict[str, Any]]:
    """Return x, y scatter data for lidar points in BEV (first available lidar)."""
    lidar_ids = scene.available_lidar_ids
    if not lidar_ids:
        return None

    lidar = scene.get_lidar_at_iteration(iteration, lidar_ids[0])
    if lidar is None:
        return None

    xy = lidar.xy
    if len(xy) == 0:
        return None

    # Transform lidar points from ego frame to global frame using ego pose
    ego_pose = ego_state.imu_se3
    cos_yaw = np.cos(ego_pose.yaw)
    sin_yaw = np.sin(ego_pose.yaw)
    x_global = xy[:, 0] * cos_yaw - xy[:, 1] * sin_yaw + ego_pose.x
    y_global = xy[:, 0] * sin_yaw + xy[:, 1] * cos_yaw + ego_pose.y

    # Subsample if too many points for performance
    max_points = 30000
    if len(x_global) > max_points:
        indices = np.random.choice(len(x_global), max_points, replace=False)
        x_global = x_global[indices]
        y_global = y_global[indices]

    return {"x": x_global.tolist(), "y": y_global.tolist()}


def get_camera_image_rgba(scene: SceneAPI, iteration: int, camera_id) -> Optional[np.ndarray]:
    """Return camera image as RGBA uint8 numpy array (flipped for Bokeh coordinate system)."""
    camera = scene.get_pinhole_camera_at_iteration(iteration, camera_id)
    if camera is None:
        return None

    img = camera.image
    if img.ndim == 2:
        # Grayscale -> RGBA
        rgba = np.zeros((*img.shape, 4), dtype=np.uint8)
        rgba[..., 0] = img
        rgba[..., 1] = img
        rgba[..., 2] = img
        rgba[..., 3] = 255
    elif img.shape[2] == 3:
        # RGB -> RGBA
        rgba = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
        rgba[..., :3] = img
        rgba[..., 3] = 255
    else:
        rgba = img

    # Bokeh image_rgba expects bottom-up row order
    rgba = np.flipud(rgba)
    return rgba
