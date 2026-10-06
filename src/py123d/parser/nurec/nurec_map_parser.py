from __future__ import annotations

import io
import json
import logging
import math
import re
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple, Union
from zipfile import ZipFile

import numpy as np
import pandas as pd
from typing_extensions import override

from py123d.datatypes import (
    BaseMapObject,
    Carpark,
    Crosswalk,
    GenericDrivable,
    Intersection,
    IntersectionType,
    Lane,
    LaneGroup,
    LaneType,
    MapMetadata,
    RoadEdge,
    RoadEdgeType,
    RoadLine,
    RoadLineType,
    SpeedBump,
    StopZone,
    StopZoneType,
    Walkway,
)
from py123d.geometry import Polyline2D, Polyline3D
from py123d.parser.base_dataset_parser import BaseMapParser
from py123d.parser.opendrive.opendrive_map_parser import iter_xodr_map_objects

logger = logging.getLogger(__name__)

# WGS84 ellipsoid constants, used to align the xodr map source (which lives in the
# OpenDRIVE file's own geo-projected frame) with the clip-local rig frame that ego
# poses and clip_gt map data already share.
_WGS84_A_M = 6378137.0
_WGS84_F = 1.0 / 298.257223563

# Wait lines mark where traffic enters an intersection or crossing, where it leaves,
# or neither. Only the entering ones oblige traffic to stop.
_STOPPING_WAIT_LINE_SUBTYPES = frozenset({"ENTRY", "CROSSWALK_ENTRY"})
_PASSING_WAIT_LINE_SUBTYPES = frozenset({"EXIT", "NOT_APPLICABLE", "BUFFER_ZONE"})

# Association `key.kind` values understood by `_read_associations`; anything else is
# counted in `_ClipgtRelations.unknown_kinds` and dropped.
_KNOWN_ASSOCIATION_KINDS = frozenset(
    {
        "NEXT_LANE",
        "PREVIOUS_LANE",
        "LEFT_LANE",
        "RIGHT_LANE",
        "ROAD_SEGMENT_SIBLING_LANE",
        "WAIT_LINE_TO_LANE",
        "INTERSECTION_AREA_TO_LANE",
        "LIGHT_TO_LANE",
        "SIGN_TO_LANE",
    }
)


@dataclass
class _ClipgtRelations:
    """Relations between map objects, keyed by clipgt `map_id`.

    Lane relations are directed; the rest are stored both ways, since the
    relation names do not say which endpoint comes first.
    """

    successors: Dict[str, set] = field(default_factory=dict)
    left: Dict[str, set] = field(default_factory=dict)
    right: Dict[str, set] = field(default_factory=dict)
    siblings: Dict[str, set] = field(default_factory=dict)
    wait_line_lanes: Dict[str, set] = field(default_factory=dict)
    intersection_lanes: Dict[str, set] = field(default_factory=dict)
    light_lanes: Dict[str, set] = field(default_factory=dict)
    sign_lanes: Dict[str, set] = field(default_factory=dict)
    # Counts of association `key.kind` values outside the vocabulary below, so
    # schema drift across a large batch shows up in logs instead of vanishing silently.
    unknown_kinds: Dict[str, int] = field(default_factory=dict)


def _clipgt_member(layer: str) -> str:
    """Archive member holding one clipgt map layer."""
    return f"clipgt/{layer}.parquet"


def _has_clipgt_layers(member_names: Set[str]) -> bool:
    """True when a USDZ carries the clipgt layers needed to build a map."""
    return all(_clipgt_member(layer) in member_names for layer in ("lane", "road_boundary"))


# The order in which each `map_source` tries the two sources a scene can carry.
_MAP_SOURCE_PREFERENCES: Dict[str, Tuple[str, ...]] = {
    "clip_gt": ("clip_gt",),
    "xodr": ("xodr",),
    "clip_gt_or_xodr": ("clip_gt", "xodr"),
    "xodr_or_clip_gt": ("xodr", "clip_gt"),
}
NUREC_MAP_SOURCES: Tuple[str, ...] = tuple(_MAP_SOURCE_PREFERENCES)


def resolve_map_source(member_names: Set[str], map_source: str, xodr_member: str = "map.xodr") -> Optional[str]:
    """Picks the source a scene's map is read from, given the members of its USDZ.

    :param member_names: Member names of the USDZ archive.
    :param map_source: Requested map source, one of :data:`NUREC_MAP_SOURCES`.
    :param xodr_member: Name of the OpenDRIVE member.
    :return: ``"clip_gt"`` or ``"xodr"``, or None if the scene carries none of the requested sources.
    """
    available = {"clip_gt": _has_clipgt_layers(member_names), "xodr": xodr_member in member_names}
    for source in _MAP_SOURCE_PREFERENCES[map_source]:
        if available[source]:
            return source
    return None


def _mads_points_xyz(entry: Dict, key: str) -> Optional[np.ndarray]:
    """(N,3) float array from a MADS parquet point list, or None if unusable."""
    try:
        pts = entry[key]
    except (KeyError, IndexError, TypeError):
        return None
    if pts is None or len(pts) < 2:
        return None
    try:
        array = np.asarray(
            [[float(p["x"]), float(p["y"]), float(p["z"])] for p in pts],
            dtype=np.float64,
        )
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    if array.ndim != 2 or array.shape[0] < 2:
        return None
    # Drop consecutive duplicates: Polyline3D/shapely reject zero-length segments.
    keep = np.ones(len(array), dtype=bool)
    keep[1:] = np.any(np.abs(np.diff(array, axis=0)) > 1e-9, axis=1)
    array = array[keep]
    return array if len(array) >= 2 else None


@dataclass(frozen=True)
class NuRecMapAlignment:
    """Rigid transform from the xodr map source's frame into the clip-local rig frame
    ego poses and clip_gt map data already share: translate by ``-origin_m``, then
    rotate by ``row_rotation``.
    """

    origin_m: np.ndarray
    row_rotation: np.ndarray

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        return (points - self.origin_m) @ self.row_rotation


def _read_xodr_geo_origin(archive: ZipFile, xodr_member: str) -> Dict[str, float]:
    """Latitude/longitude of an OpenDRIVE file's ``+lat_0``/``+lon_0`` geoReference origin."""
    root = ET.fromstring(archive.read(xodr_member))
    geo_reference = root.find("./header/geoReference")
    if geo_reference is None or geo_reference.text is None:
        raise ValueError(f"No OpenDRIVE geoReference found in {xodr_member}.")
    text = geo_reference.text.strip()
    lat_match = re.search(r"(?<![A-Za-z0-9_])\+lat_0=([^\s]+)", text)
    lon_match = re.search(r"(?<![A-Za-z0-9_])\+lon_0=([^\s]+)", text)
    if lat_match is None or lon_match is None:
        raise ValueError(f"Could not parse +lat_0/+lon_0 from geoReference: {text!r}.")
    return {"latitude": float(lat_match.group(1)), "longitude": float(lon_match.group(1))}


def _ecef_enu_basis(latitude: float, longitude: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """East/north/up unit vectors (in ECEF) of the local tangent frame at (latitude, longitude)."""
    lat_rad, lon_rad = math.radians(latitude), math.radians(longitude)
    sin_lat, cos_lat = math.sin(lat_rad), math.cos(lat_rad)
    sin_lon, cos_lon = math.sin(lon_rad), math.cos(lon_rad)
    east = np.array([-sin_lon, cos_lon, 0.0], dtype=np.float64)
    north = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat], dtype=np.float64)
    up = np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat], dtype=np.float64)
    return east, north, up


def _ecef_from_geodetic(latitude: float, longitude: float, altitude: float) -> np.ndarray:
    """WGS84 geodetic (lat, lon, alt) to ECEF (x, y, z), in meters."""
    lat_rad, lon_rad = math.radians(latitude), math.radians(longitude)
    e2 = _WGS84_F * (2.0 - _WGS84_F)
    radius = _WGS84_A_M / math.sqrt(1.0 - e2 * math.sin(lat_rad) ** 2)
    return np.array(
        [
            (radius + altitude) * math.cos(lat_rad) * math.cos(lon_rad),
            (radius + altitude) * math.cos(lat_rad) * math.sin(lon_rad),
            (radius * (1.0 - e2) + altitude) * math.sin(lat_rad),
        ],
        dtype=np.float64,
    )


def _ecef_to_geodetic(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """WGS84 ECEF (x, y, z) to geodetic (latitude, longitude, altitude), via Bowring iteration."""
    e2 = _WGS84_F * (2.0 - _WGS84_F)
    longitude = math.atan2(y, x)
    xy_norm = math.hypot(x, y)
    latitude = math.atan2(z, xy_norm * (1.0 - e2))
    altitude = 0.0
    for _ in range(8):
        radius = _WGS84_A_M / math.sqrt(1.0 - e2 * math.sin(latitude) ** 2)
        altitude = xy_norm / math.cos(latitude) - radius
        latitude = math.atan2(z, xy_norm * (1.0 - e2 * radius / (radius + altitude)))
    radius = _WGS84_A_M / math.sqrt(1.0 - e2 * math.sin(latitude) ** 2)
    altitude = xy_norm / math.cos(latitude) - radius
    return math.degrees(latitude), math.degrees(longitude), altitude


def _local_enu_from_lat_lng(
    latitude: float, longitude: float, origin_latitude: float, origin_longitude: float
) -> Tuple[float, float]:
    """East/north offset in meters of (latitude, longitude) from an ENU tangent-plane origin."""
    east, north, _ = _ecef_enu_basis(origin_latitude, origin_longitude)
    delta = _ecef_from_geodetic(latitude, longitude, 0.0) - _ecef_from_geodetic(origin_latitude, origin_longitude, 0.0)
    return float(np.dot(delta, east)), float(np.dot(delta, north))


def _rotation_matrix_from_axis_angle(axis_angle: Dict[str, float]) -> np.ndarray:
    """Rotation matrix from a NuRec {qx, qy, qz, angle} axis (unnormalized) + angle-in-degrees encoding."""
    axis = np.array([axis_angle["qx"], axis_angle["qy"], axis_angle["qz"]], dtype=np.float64)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm == 0.0:
        raise ValueError(f"NuRec axis-angle rotation has zero-length axis: {axis_angle}.")
    x, y, z = axis / axis_norm
    angle_rad = math.radians(float(axis_angle["angle"]))
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    one_minus_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
            [y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s],
            [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c],
        ],
        dtype=np.float64,
    )


def _row_rotation_from_world_base(T_world_base: np.ndarray, geo_origin: Dict[str, float]) -> np.ndarray:
    """Row-vector rotation (world frame -> local ENU at geo_origin) from a world<-base ECEF pose."""
    east, north, up = _ecef_enu_basis(geo_origin["latitude"], geo_origin["longitude"])
    enu_basis = np.stack((east, north, up), axis=0)
    return enu_basis @ T_world_base[:3, :3]


def _resolve_alignment_z_origin(
    z_origin: str,
    z_origin_m: Optional[float],
    fallback_altitude: float,
    pose_record: Optional[Dict] = None,
    alignment_pose: Optional[Dict] = None,
) -> float:
    """Altitude (m) that maps to z=0 in the aligned frame, per the requested z_origin policy."""
    if z_origin_m is not None:
        return float(z_origin_m)
    if z_origin in {"selected_alignment_altitude", "world_base_altitude"}:
        return float(fallback_altitude)
    if z_origin in {"first_alignment_altitude", "first_alignment_pose_altitude"} and pose_record is not None:
        return float(pose_record["record"][0]["alignment_world_pose"]["lat_lng_alt"]["altitude"])
    if z_origin == "current_alignment_altitude" and alignment_pose is not None:
        return float(alignment_pose["lat_lng_alt"]["altitude"])
    if z_origin == "alignment_origin_altitude" and pose_record is not None:
        return float(pose_record["alignment_origin"]["altitude"])
    if z_origin in {"zero", "none"}:
        return 0.0
    raise ValueError(f"Unsupported NuRec map alignment z_origin: {z_origin!r}.")


def _alignment_origin_from_lat_lng_alt(
    geo_origin: Dict[str, float],
    latitude: float,
    longitude: float,
    altitude: float,
    xy_origin_m: Optional[Tuple[float, float]],
    z_origin: str,
    z_origin_m: Optional[float],
    pose_record: Optional[Dict] = None,
    alignment_pose: Optional[Dict] = None,
) -> np.ndarray:
    if xy_origin_m is None:
        x_origin_m, y_origin_m = _local_enu_from_lat_lng(
            latitude=latitude,
            longitude=longitude,
            origin_latitude=geo_origin["latitude"],
            origin_longitude=geo_origin["longitude"],
        )
    else:
        x_origin_m, y_origin_m = float(xy_origin_m[0]), float(xy_origin_m[1])
    z_origin_value = _resolve_alignment_z_origin(
        z_origin=z_origin,
        z_origin_m=z_origin_m,
        fallback_altitude=altitude,
        pose_record=pose_record,
        alignment_pose=alignment_pose,
    )
    return np.array([x_origin_m, y_origin_m, z_origin_value], dtype=np.float64)


def _read_json_member(archive: ZipFile, member: str, location: str) -> Dict:
    """JSON payload of an archive member, or a clear scene-identified error if it's absent."""
    try:
        payload = archive.read(member)
    except KeyError:
        raise ValueError(f"NuRec map {location}: archive has no {member!r} member") from None
    return json.loads(payload)


def _load_world_base_map_alignment(
    archive: ZipFile,
    geo_origin: Dict[str, float],
    rig_member: str,
    xy_origin_m: Optional[Tuple[float, float]],
    z_origin: str,
    z_origin_m: Optional[float],
    location: str,
) -> NuRecMapAlignment:
    """Alignment anchored on ``rig_trajectories.json``'s world<-base ECEF pose (present in every scene)."""
    rig_json = _read_json_member(archive, rig_member, location)
    T_world_base = np.asarray(rig_json["T_world_base"], dtype=np.float64)
    base_latitude, base_longitude, base_altitude = _ecef_to_geodetic(*T_world_base[:3, 3])
    origin_m = _alignment_origin_from_lat_lng_alt(
        geo_origin=geo_origin,
        latitude=base_latitude,
        longitude=base_longitude,
        altitude=base_altitude,
        xy_origin_m=xy_origin_m,
        z_origin=z_origin,
        z_origin_m=z_origin_m,
    )
    row_rotation = _row_rotation_from_world_base(T_world_base, geo_origin)
    return NuRecMapAlignment(origin_m=origin_m, row_rotation=row_rotation)


def _load_pose_record_map_alignment(
    archive: ZipFile,
    geo_origin: Dict[str, float],
    pose_record_member: str,
    record_index: int,
    invert_rotation: bool,
    xy_origin_m: Optional[Tuple[float, float]],
    z_origin: str,
    z_origin_m: Optional[float],
    location: str,
) -> NuRecMapAlignment:
    """Alignment anchored on one ``pose_record.json`` sample; only present in some scenes."""
    pose_record = _read_json_member(archive, pose_record_member, location)
    records = pose_record.get("record", [])
    if not records or not 0 <= record_index < len(records):
        raise ValueError(
            f"NuRec map {location}: {pose_record_member} has no record at index {record_index} "
            f"(found {len(records)} record(s))"
        )
    alignment_pose = records[record_index].get("alignment_world_pose", {})
    axis_angle = alignment_pose.get("axis_angle")
    if axis_angle is None:
        raise ValueError(f"NuRec map {location}: no alignment_world_pose.axis_angle found in {pose_record_member}.")
    rotation = _rotation_matrix_from_axis_angle(axis_angle)
    row_rotation = rotation if invert_rotation else rotation.T
    lat_lng_alt = alignment_pose["lat_lng_alt"]
    origin_m = _alignment_origin_from_lat_lng_alt(
        geo_origin=geo_origin,
        latitude=float(lat_lng_alt["latitude"]),
        longitude=float(lat_lng_alt["longitude"]),
        altitude=float(lat_lng_alt["altitude"]),
        xy_origin_m=xy_origin_m,
        z_origin=z_origin,
        z_origin_m=z_origin_m,
        pose_record=pose_record,
        alignment_pose=alignment_pose,
    )
    return NuRecMapAlignment(origin_m=origin_m, row_rotation=row_rotation)


def _transform_polyline(polyline: Union[Polyline2D, Polyline3D], alignment: NuRecMapAlignment) -> Polyline3D:
    points = np.asarray(polyline.array, dtype=np.float64)
    if points.shape[1] == 2:
        points = np.hstack((points, np.zeros((points.shape[0], 1), dtype=np.float64)))
    return Polyline3D.from_array(alignment.transform_points(points))


def _transform_map_object(map_object: BaseMapObject, alignment: NuRecMapAlignment) -> BaseMapObject:
    """A copy of ``map_object`` with every polyline/outline moved into the aligned frame."""
    if isinstance(map_object, Lane):
        return Lane(
            object_id=map_object.object_id,
            lane_type=map_object.lane_type,
            left_boundary=_transform_polyline(map_object.left_boundary, alignment),
            right_boundary=_transform_polyline(map_object.right_boundary, alignment),
            centerline=_transform_polyline(map_object.centerline, alignment),
            lane_group_id=map_object.lane_group_id,
            left_lane_id=map_object.left_lane_id,
            right_lane_id=map_object.right_lane_id,
            predecessor_ids=list(map_object.predecessor_ids),
            successor_ids=list(map_object.successor_ids),
            speed_limit_mps=map_object.speed_limit_mps,
        )
    if isinstance(map_object, LaneGroup):
        return LaneGroup(
            object_id=map_object.object_id,
            lane_ids=list(map_object.lane_ids),
            left_boundary=_transform_polyline(map_object.left_boundary, alignment),
            right_boundary=_transform_polyline(map_object.right_boundary, alignment),
            intersection_id=map_object.intersection_id,
            predecessor_ids=list(map_object.predecessor_ids),
            successor_ids=list(map_object.successor_ids),
        )
    if isinstance(map_object, Intersection):
        return Intersection(
            object_id=map_object.object_id,
            intersection_type=map_object.intersection_type,
            lane_group_ids=list(map_object.lane_group_ids),
            outline=_transform_polyline(map_object.outline, alignment),
        )
    if isinstance(map_object, Crosswalk):
        return Crosswalk(object_id=map_object.object_id, outline=_transform_polyline(map_object.outline, alignment))
    if isinstance(map_object, Carpark):
        return Carpark(object_id=map_object.object_id, outline=_transform_polyline(map_object.outline, alignment))
    if isinstance(map_object, Walkway):
        return Walkway(object_id=map_object.object_id, outline=_transform_polyline(map_object.outline, alignment))
    if isinstance(map_object, GenericDrivable):
        return GenericDrivable(
            object_id=map_object.object_id, outline=_transform_polyline(map_object.outline, alignment)
        )
    if isinstance(map_object, StopZone):
        return StopZone(
            object_id=map_object.object_id,
            stop_zone_type=map_object.stop_zone_type,
            outline=_transform_polyline(map_object.outline, alignment),
            lane_ids=list(map_object.lane_ids),
        )
    if isinstance(map_object, SpeedBump):
        return SpeedBump(
            object_id=map_object.object_id,
            outline=_transform_polyline(map_object.outline, alignment),
            speed_bump_type=map_object.speed_bump_type,
        )
    if isinstance(map_object, RoadEdge):
        return RoadEdge(
            object_id=map_object.object_id,
            road_edge_type=map_object.road_edge_type,
            polyline=_transform_polyline(map_object.polyline, alignment),
        )
    if isinstance(map_object, RoadLine):
        return RoadLine(
            object_id=map_object.object_id,
            road_line_type=map_object.road_line_type,
            polyline=_transform_polyline(map_object.polyline, alignment),
        )
    logger.warning("NuRec map alignment: leaving unsupported map object type unaligned: %s", type(map_object).__name__)
    return map_object


class NuRecMapParser(BaseMapParser):
    """Map parser for one NuRec USDZ scene.

    A scene can carry its map in two forms, and ``map_source`` selects which is read:

    - ``"clip_gt"``: the MADS clipgt layers. Coordinates are already in
      the clip-local rig frame, so no geo-projection is needed, and polylines are
      emitted at source resolution. The raw lane rails are not re-emitted as road
      lines: they are already the Lane boundaries, and adjacent lanes share a rail,
      so every interior boundary would appear twice.
    - ``"xodr"``: the embedded ``map.xodr`` OpenDRIVE file, converted through
      py123d's existing OpenDRIVE parser. Since xodr coordinates live in their own
      geo-projected frame, they are aligned into the clip-local rig frame using
      ``rig_trajectories.json``'s world<-base ECEF pose (or, optionally, a
      ``pose_record.json`` sample).
    - ``"clip_gt_or_xodr"`` (default): prefers ``"clip_gt"``, falling back to
      ``"xodr"`` when the archive has no clipgt lane/road_boundary layers. Some
      NuRec scenes (e.g. ~20% of nurec-2601_train) ship ``map.xodr`` only.
    - ``"xodr_or_clip_gt"``: prefers ``"xodr"``, falling back to ``"clip_gt"``.

    :func:`resolve_map_source` tells which of the two a scene will use. On a scene
    that carries none of the requested sources, :meth:`iter_map_objects` raises;
    :class:`~py123d.parser.nurec.nurec_parser.NuRecParser` creates no map parser
    for such a scene and converts its log without a map.
    """

    def __init__(
        self,
        usdz_path: Union[str, Path],
        location: str = "",
        split: Optional[str] = None,
        log_name: Optional[str] = None,
        map_source: str = "clip_gt_or_xodr",
        xodr_member: str = "map.xodr",
        xodr_interpolation_step_size: float = 1.0,
        xodr_connection_distance_threshold: float = 0.1,
        xodr_internal_only: bool = True,
        alignment_enabled: bool = True,
        alignment_source: str = "world_base",
        alignment_rig_member: str = "rig_trajectories.json",
        alignment_pose_record_member: str = "pose_record.json",
        alignment_record_index: int = 0,
        alignment_invert_rotation: bool = True,
        alignment_xy_origin_m: Optional[Tuple[float, float]] = None,
        alignment_z_origin: str = "world_base_altitude",
        alignment_z_origin_m: Optional[float] = None,
    ) -> None:
        if map_source not in NUREC_MAP_SOURCES:
            raise ValueError(f"Unsupported NuRec map_source: {map_source!r}. Use one of {NUREC_MAP_SOURCES}.")
        if alignment_source not in ("world_base", "pose_record"):
            raise ValueError(
                f"Unsupported NuRec map alignment source: {alignment_source!r}. Use 'world_base' or 'pose_record'."
            )
        self._usdz_path = Path(usdz_path)
        self._location = location
        self._split = split
        self._log_name = log_name
        self._map_source = map_source
        self._xodr_member = xodr_member
        self._xodr_interpolation_step_size = xodr_interpolation_step_size
        self._xodr_connection_distance_threshold = xodr_connection_distance_threshold
        self._xodr_internal_only = xodr_internal_only
        self._alignment_enabled = alignment_enabled
        self._alignment_source = alignment_source
        self._alignment_rig_member = alignment_rig_member
        self._alignment_pose_record_member = alignment_pose_record_member
        self._alignment_record_index = alignment_record_index
        self._alignment_invert_rotation = alignment_invert_rotation
        self._alignment_xy_origin_m = alignment_xy_origin_m
        self._alignment_z_origin = alignment_z_origin
        self._alignment_z_origin_m = alignment_z_origin_m

    @override
    def get_map_metadata(self) -> MapMetadata:
        """Inherited, see superclass."""
        return MapMetadata(
            dataset="nurec",
            location=self._location,
            map_has_z=True,
            map_is_per_log=True,
            split=self._split,
            log_name=self._log_name,
        )

    def _read_layer(self, archive: ZipFile, layer: str) -> List[Dict]:
        """Row payloads of one clipgt layer; empty (with a warning) if absent."""
        return [payload for _, payload in self._read_layer_rows(archive, layer)]

    def _read_layer_rows(self, archive: ZipFile, layer: str) -> List[Tuple[Optional[str], Dict]]:
        """Row payloads of one clipgt layer paired with their `key.map_id`."""
        frame = self._read_layer_frame(archive, layer)
        if frame is None or layer not in frame.columns:
            return []
        if "key" in frame.columns:
            map_ids = [key.get("map_id") if isinstance(key, dict) else None for key in frame["key"]]
        else:
            logger.warning("NuRec map %s: %s layer lacks column 'key'", self._location, layer)
            map_ids = [None] * len(frame)
        return list(zip(map_ids, frame[layer]))

    def _read_layer_frame(self, archive: ZipFile, layer: str) -> Optional["pd.DataFrame"]:
        """Full dataframe of one clipgt layer, or None (with a warning) if absent."""
        member = _clipgt_member(layer)
        try:
            payload = archive.read(member)
        except KeyError:
            logger.warning("NuRec map %s has no %s; skipping layer", self._location, member)
            return None
        return pd.read_parquet(io.BytesIO(payload))

    def _read_associations(self, archive: ZipFile) -> _ClipgtRelations:
        """Map relations from the clipgt association layer.

        NEXT_LANE and its inverse PREVIOUS_LANE are merged, since neither alone
        covers every link. Relation names do not consistently say which endpoint
        comes first (`WAIT_LINE_TO_LANE` lists the lane, `INTERSECTION_AREA_TO_LANE`
        the intersection), so those are stored both ways and resolved by the caller.
        """
        relations = _ClipgtRelations()
        frame = self._read_layer_frame(archive, "association")
        if frame is None:
            return relations
        if "key" not in frame.columns or "association" not in frame.columns:
            logger.warning("NuRec map %s: association layer lacks key/association columns", self._location)
            return relations

        for key, association in zip(frame["key"], frame["association"]):
            kind = key.get("kind") if isinstance(key, dict) else None
            if not isinstance(association, dict):
                continue
            subjects = association.get("subjects")
            objects = association.get("objects")
            if subjects is None or objects is None:
                continue
            if kind not in _KNOWN_ASSOCIATION_KINDS:
                relations.unknown_kinds[str(kind)] = relations.unknown_kinds.get(str(kind), 0) + 1
                continue
            for subject in subjects:
                for obj in objects:
                    if subject == obj:
                        continue
                    if kind == "NEXT_LANE":
                        relations.successors.setdefault(subject, set()).add(obj)
                    elif kind == "PREVIOUS_LANE":
                        relations.successors.setdefault(obj, set()).add(subject)
                    elif kind == "LEFT_LANE":
                        relations.left.setdefault(subject, set()).add(obj)
                    elif kind == "RIGHT_LANE":
                        relations.right.setdefault(subject, set()).add(obj)
                    elif kind == "ROAD_SEGMENT_SIBLING_LANE":
                        relations.siblings.setdefault(subject, set()).add(obj)
                        relations.siblings.setdefault(obj, set()).add(subject)
                    elif kind == "WAIT_LINE_TO_LANE":
                        relations.wait_line_lanes.setdefault(subject, set()).add(obj)
                        relations.wait_line_lanes.setdefault(obj, set()).add(subject)
                    elif kind == "INTERSECTION_AREA_TO_LANE":
                        relations.intersection_lanes.setdefault(subject, set()).add(obj)
                        relations.intersection_lanes.setdefault(obj, set()).add(subject)
                    elif kind == "LIGHT_TO_LANE":
                        relations.light_lanes.setdefault(subject, set()).add(obj)
                        relations.light_lanes.setdefault(obj, set()).add(subject)
                    elif kind == "SIGN_TO_LANE":
                        relations.sign_lanes.setdefault(subject, set()).add(obj)
                        relations.sign_lanes.setdefault(obj, set()).add(subject)
        if relations.unknown_kinds:
            logger.warning(
                "NuRec map %s: dropped %d association rows with an unrecognised kind %s",
                self._location,
                sum(relations.unknown_kinds.values()),
                sorted(relations.unknown_kinds),
            )
        return relations

    @override
    def iter_map_objects(self) -> Iterator[BaseMapObject]:
        """Inherited, see superclass."""
        # The archive is opened once and reused for whichever source is taken.
        with ZipFile(self._usdz_path) as archive:
            member_names = set(archive.namelist())
            source = resolve_map_source(member_names, self._map_source, self._xodr_member)
            if source is None:
                raise self._missing_map_source_error(member_names)
            preferred_source = _MAP_SOURCE_PREFERENCES[self._map_source][0]
            if source != preferred_source:
                # E.g. ~20% of nurec-2601_train ships map.xodr only.
                logger.info(
                    "NuRec map %s: map_source=%r but archive has no %s source; falling back to %s",
                    self._location,
                    self._map_source,
                    preferred_source,
                    source,
                )
            if source == "clip_gt":
                yield from self._iter_map_objects_clip_gt(archive)
            else:
                yield from self._iter_map_objects_xodr(archive)

    def _missing_map_source_error(self, member_names: Set[str]) -> ValueError:
        """Error for an archive that carries none of the sources ``map_source`` asks for."""
        requested_sources = _MAP_SOURCE_PREFERENCES[self._map_source]
        missing: List[str] = []
        if "clip_gt" in requested_sources:
            clipgt_members = sorted(
                name.removeprefix("clipgt/").removesuffix(".parquet")
                for name in member_names
                if name.startswith("clipgt/")
            )
            missing.append(f"the required 'lane'/'road_boundary' clipgt layers (available: {clipgt_members})")
        if "xodr" in requested_sources:
            missing.append(f"a {self._xodr_member!r} member")
        return ValueError(
            f"NuRec map {self._location}: map_source={self._map_source!r} but the archive is missing "
            + " and ".join(missing)
        )

    def _iter_map_objects_xodr(self, archive: ZipFile) -> Iterator[BaseMapObject]:
        """Map objects from the embedded map.xodr, aligned into the clip-local rig frame."""
        xodr_bytes = archive.read(self._xodr_member)
        alignment = self._load_alignment(archive) if self._alignment_enabled else None

        with tempfile.NamedTemporaryFile(suffix=".xodr", delete=False) as tmp_file:
            tmp_file.write(xodr_bytes)
            tmp_path = Path(tmp_file.name)
        try:
            for map_object in iter_xodr_map_objects(
                xodr_file=tmp_path,
                interpolation_step_size=self._xodr_interpolation_step_size,
                connection_distance_threshold=self._xodr_connection_distance_threshold,
                internal_only=self._xodr_internal_only,
            ):
                yield _transform_map_object(map_object, alignment) if alignment is not None else map_object
        finally:
            tmp_path.unlink(missing_ok=True)

    def _load_alignment(self, archive: ZipFile) -> NuRecMapAlignment:
        geo_origin = _read_xodr_geo_origin(archive, self._xodr_member)
        if self._alignment_source == "world_base":
            return _load_world_base_map_alignment(
                archive=archive,
                geo_origin=geo_origin,
                rig_member=self._alignment_rig_member,
                xy_origin_m=self._alignment_xy_origin_m,
                z_origin=self._alignment_z_origin,
                z_origin_m=self._alignment_z_origin_m,
                location=self._location,
            )
        if self._alignment_source == "pose_record":
            return _load_pose_record_map_alignment(
                archive=archive,
                geo_origin=geo_origin,
                pose_record_member=self._alignment_pose_record_member,
                record_index=self._alignment_record_index,
                invert_rotation=self._alignment_invert_rotation,
                xy_origin_m=self._alignment_xy_origin_m,
                z_origin=self._alignment_z_origin,
                z_origin_m=self._alignment_z_origin_m,
                location=self._location,
            )
        raise ValueError(
            f"Unsupported NuRec map alignment source: {self._alignment_source!r}. Use 'world_base' or 'pose_record'."
        )

    def _iter_map_objects_clip_gt(self, archive: ZipFile) -> Iterator[BaseMapObject]:
        """Map objects from the clipgt MADS layers of an archive that carries them."""
        lane_rows = self._read_layer_rows(archive, "lane")
        boundaries = self._read_layer(archive, "road_boundary")
        relations = self._read_associations(archive)
        crosswalks = self._read_layer(archive, "crosswalk")
        gore_areas = self._read_layer(archive, "gore_area")
        road_islands = self._read_layer(archive, "road_island")
        wait_line_rows = self._read_layer_rows(archive, "wait_line")
        lane_lines = self._read_layer(archive, "lane_line")
        intersection_rows = self._read_layer_rows(archive, "intersection_area")
        sign_categories = {
            map_id: sign.get("category")
            for map_id, sign in self._read_layer_rows(archive, "traffic_sign")
            if map_id is not None
        }

        next_id = 0

        # A lane's successors and neighbours can appear later in the file, so ids
        # are assigned first and the Lane objects built once they are all known.
        parsed_lanes: List[Tuple[int, Optional[str], np.ndarray, np.ndarray, np.ndarray, Optional[float]]] = []
        lane_id_by_map_id: Dict[str, int] = {}
        for map_id, lane in lane_rows:
            left_d = _mads_points_xyz(lane, "left_rail")
            right_d = _mads_points_xyz(lane, "right_rail")
            if left_d is None or right_d is None:
                continue
            center_d = _centerline_from_rails(left_d, right_d)
            if center_d is None:
                continue
            speed_limit_mps = _mads_speed_limit_mps(lane)
            parsed_lanes.append((next_id, map_id, left_d, right_d, center_d, speed_limit_mps))
            if map_id is not None:
                if map_id in lane_id_by_map_id:
                    logger.warning(
                        "NuRec map %s: duplicate lane map_id %s; connectivity keeps the first",
                        self._location,
                        map_id,
                    )
                else:
                    lane_id_by_map_id[map_id] = next_id
            next_id += 1

        succ_ids: Dict[int, List[int]] = {}
        pred_ids: Dict[int, List[int]] = {}
        for map_id, lane_id in lane_id_by_map_id.items():
            for succ_map_id in sorted(relations.successors.get(map_id, ())):
                succ_lane_id = lane_id_by_map_id.get(succ_map_id)
                if succ_lane_id is None:
                    continue
                succ_ids.setdefault(lane_id, []).append(succ_lane_id)
                pred_ids.setdefault(succ_lane_id, []).append(lane_id)
        n_edges = sum(len(ids) for ids in succ_ids.values())
        logger.info(
            "NuRec map %s: lane connectivity %d edges over %d lanes (%d lanes without successors)",
            self._location,
            n_edges,
            len(parsed_lanes),
            len(parsed_lanes) - len(succ_ids),
        )

        left_by_lane_id = {
            lane_id: _neighbour_lane_id(relations.left.get(map_id), lane_id_by_map_id)
            for map_id, lane_id in lane_id_by_map_id.items()
        }
        right_by_lane_id = {
            lane_id: _neighbour_lane_id(relations.right.get(map_id), lane_id_by_map_id)
            for map_id, lane_id in lane_id_by_map_id.items()
        }

        # Group boundaries are the outer rails of the outermost lanes, so the
        # members are ordered across the road first.
        lane_geometry = {object_id: (left, right, center) for object_id, _, left, right, center, _ in parsed_lanes}
        group_members = _lane_groups(
            [map_id for _, map_id, *_ in parsed_lanes if map_id is not None], relations.siblings
        )
        group_id_by_lane_id: Dict[int, int] = {}
        groups: List[Tuple[int, List[int]]] = []
        for member_map_ids in group_members:
            lane_ids = [lane_id_by_map_id[map_id] for map_id in member_map_ids if map_id in lane_id_by_map_id]
            if not lane_ids:
                continue
            ordered = _order_left_to_right([lane_geometry[lane_id][2] for lane_id in lane_ids])
            lane_ids = [lane_ids[index] for index in ordered]
            group_id = next_id
            next_id += 1
            groups.append((group_id, lane_ids))
            for lane_id in lane_ids:
                group_id_by_lane_id[lane_id] = group_id

        for object_id, map_id, left_d, right_d, center_d, speed_limit_mps in parsed_lanes:
            yield Lane(
                object_id=object_id,
                lane_type=LaneType.SURFACE_STREET,
                left_boundary=Polyline3D.from_array(left_d),
                right_boundary=Polyline3D.from_array(right_d),
                centerline=Polyline3D.from_array(center_d),
                lane_group_id=group_id_by_lane_id.get(object_id),
                left_lane_id=left_by_lane_id.get(object_id),
                right_lane_id=right_by_lane_id.get(object_id),
                predecessor_ids=sorted(pred_ids.get(object_id, [])),
                successor_ids=sorted(succ_ids.get(object_id, [])),
                speed_limit_mps=speed_limit_mps,
            )

        for group_id, lane_ids in groups:
            group_successors = sorted(
                {
                    group_id_by_lane_id[successor]
                    for lane_id in lane_ids
                    for successor in succ_ids.get(lane_id, ())
                    if group_id_by_lane_id.get(successor, group_id) != group_id
                }
            )
            group_predecessors = sorted(
                {
                    group_id_by_lane_id[predecessor]
                    for lane_id in lane_ids
                    for predecessor in pred_ids.get(lane_id, ())
                    if group_id_by_lane_id.get(predecessor, group_id) != group_id
                }
            )
            yield LaneGroup(
                object_id=group_id,
                lane_ids=lane_ids,
                left_boundary=Polyline3D.from_array(lane_geometry[lane_ids[0]][0]),
                right_boundary=Polyline3D.from_array(lane_geometry[lane_ids[-1]][1]),
                predecessor_ids=group_predecessors,
                successor_ids=group_successors,
            )

        for boundary in boundaries:
            pts = _mads_points_xyz(boundary, "location")
            if pts is None:
                continue
            yield RoadEdge(
                object_id=next_id,
                road_edge_type=RoadEdgeType.ROAD_EDGE_BOUNDARY,
                polyline=Polyline3D.from_array(pts),
            )
            next_id += 1

        for crosswalk in crosswalks:
            pts = _mads_points_xyz(crosswalk, "location")
            if pts is None or len(pts) < 3:
                continue
            yield Crosswalk(object_id=next_id, outline=Polyline3D.from_array(pts))
            next_id += 1

        for gore_area in gore_areas:
            pts = _mads_points_xyz(gore_area, "location")
            if pts is None or len(pts) < 3:
                continue
            yield GenericDrivable(object_id=next_id, outline=Polyline3D.from_array(pts))
            next_id += 1

        for road_island in road_islands:
            pts = _mads_points_xyz(road_island, "location")
            if pts is None or len(pts) < 3:
                continue
            yield Walkway(object_id=next_id, outline=Polyline3D.from_array(pts))
            next_id += 1

        unknown_subtypes: Dict[str, int] = {}
        for map_id, wait_line in wait_line_rows:
            subtype = wait_line.get("intersection_subtype")
            if subtype not in _STOPPING_WAIT_LINE_SUBTYPES:
                if subtype not in _PASSING_WAIT_LINE_SUBTYPES:
                    unknown_subtypes[str(subtype)] = unknown_subtypes.get(str(subtype), 0) + 1
                continue
            pts = _mads_points_xyz(wait_line, "location")
            if pts is None or float(np.linalg.norm(pts[-1][:2] - pts[0][:2])) < 1e-6:
                continue
            related = _wait_line_lanes(map_id, relations)
            yield StopZone(
                object_id=next_id,
                stop_zone_type=_stop_zone_type(related, wait_line.get("category"), subtype, relations, sign_categories),
                outline=Polyline3D.from_array(_segment_to_outline(pts)),
                lane_ids=sorted(
                    lane_id_by_map_id[lane_map_id] for lane_map_id in related if lane_map_id in lane_id_by_map_id
                ),
            )
            next_id += 1

        if unknown_subtypes:
            logger.warning(
                "NuRec map %s: dropped %d wait lines with an unrecognised subtype %s",
                self._location,
                sum(unknown_subtypes.values()),
                sorted(unknown_subtypes),
            )

        unknown_line_styles: Dict[str, int] = {}
        for lane_line in lane_lines:
            pts = _mads_points_xyz(lane_line, "line_rail")
            if pts is None or len(pts) < 2:
                continue
            road_line_type = _mads_road_line_type(lane_line)
            if road_line_type == RoadLineType.UNKNOWN:
                styles_raw = lane_line.get("styles")
                if styles_raw is not None and len(styles_raw) > 0:
                    majority_style = max(set(styles_raw), key=list(styles_raw).count)
                    unknown_line_styles[str(majority_style)] = unknown_line_styles.get(str(majority_style), 0) + 1
            yield RoadLine(
                object_id=next_id,
                road_line_type=road_line_type,
                polyline=Polyline3D.from_array(pts),
            )
            next_id += 1

        if unknown_line_styles:
            logger.warning(
                "NuRec map %s: %d lane lines had an unrecognised style %s",
                self._location,
                sum(unknown_line_styles.values()),
                sorted(unknown_line_styles),
            )

        for map_id, area in intersection_rows:
            pts = _mads_points_xyz(area, "location")
            if pts is None or len(pts) < 3:
                continue
            related = relations.intersection_lanes.get(map_id, ()) if map_id is not None else ()
            lane_group_ids = sorted(
                {
                    group_id_by_lane_id[lane_id_by_map_id[lane_map_id]]
                    for lane_map_id in related
                    if lane_map_id in lane_id_by_map_id and lane_id_by_map_id[lane_map_id] in group_id_by_lane_id
                }
            )
            yield Intersection(
                object_id=next_id,
                intersection_type=_intersection_type(related, area.get("category"), relations, sign_categories),
                lane_group_ids=lane_group_ids,
                outline=Polyline3D.from_array(pts),
            )
            next_id += 1


def _neighbour_lane_id(
    neighbour_map_ids: Optional[set],
    lane_id_by_map_id: Dict[str, int],
) -> Optional[int]:
    """The lane beside this one, or None where it lies outside the scene.

    Lane holds one id per side, so the first neighbour present wins.
    """
    for map_id in sorted(neighbour_map_ids or ()):
        lane_id = lane_id_by_map_id.get(map_id)
        if lane_id is not None:
            return lane_id
    return None


def _wait_line_lanes(map_id: Optional[str], relations: _ClipgtRelations) -> set:
    """Lanes a wait line applies to.

    Read from the wait line's own id, which reads `<wait line>-<lane>`; the
    association layer covers only a fraction of them.
    """
    if map_id is None:
        return set()
    lanes = set(relations.wait_line_lanes.get(map_id, ()))
    _, separator, lane_map_id = map_id.partition("-")
    if separator:
        lanes.add(lane_map_id)
    return lanes


def _stop_zone_type(
    lane_map_ids: Iterable[str],
    category: Optional[str],
    subtype: Optional[str],
    relations: _ClipgtRelations,
    sign_categories: Dict[str, Optional[str]],
) -> StopZoneType:
    """What obliges traffic to stop at a wait line.

    A light or sign controlling the lane wins, then the crossing the line guards,
    and only then the line's own category. That category marks a painted stop bar
    rather than a stop sign — scenes carry far more STOP wait lines than stop
    signs — so it types just the lines nothing else accounts for.
    """
    if any(lane_map_id in relations.light_lanes for lane_map_id in lane_map_ids):
        return StopZoneType.TRAFFIC_LIGHT
    sign_ids = {sign for lane_map_id in lane_map_ids for sign in relations.sign_lanes.get(lane_map_id, ())}
    sign_names = {sign_categories.get(sign_id) or "" for sign_id in sign_ids}
    if any("YIELD" in name for name in sign_names):
        return StopZoneType.YIELD_SIGN
    if any("STOP" in name for name in sign_names):
        return StopZoneType.STOP_SIGN
    if subtype == "CROSSWALK_ENTRY":
        return StopZoneType.PEDESTRIAN_CROSSING
    if category == "STOP":
        return StopZoneType.STOP_SIGN
    return StopZoneType.UNKNOWN


def _intersection_type(
    lane_map_ids: Iterable[str],
    category: Optional[str],
    relations: _ClipgtRelations,
    sign_categories: Dict[str, Optional[str]],
) -> IntersectionType:
    """How an intersection is controlled, from the lights and signs on its lanes.

    The clipgt category describes shape (`FOUR_WAY`, ...) rather than control,
    so it is not used.
    """
    if any(lane_map_id in relations.light_lanes for lane_map_id in lane_map_ids):
        return IntersectionType.TRAFFIC_LIGHT
    sign_ids = {sign for lane_map_id in lane_map_ids for sign in relations.sign_lanes.get(lane_map_id, ())}
    sign_names = {sign_categories.get(sign_id) or "" for sign_id in sign_ids}
    if "STOP" in (category or "") or any("STOP" in name for name in sign_names):
        return IntersectionType.STOP_SIGN
    return IntersectionType.DEFAULT


def _lane_groups(lane_map_ids: List[str], siblings: Dict[str, set]) -> List[List[str]]:
    """Lanes of one road segment: connected components of the sibling relation.

    Traversal follows the source order of the lanes to keep groups stable
    across runs.
    """
    known = set(lane_map_ids)
    seen: set = set()
    groups: List[List[str]] = []
    for map_id in lane_map_ids:
        if map_id in seen:
            continue
        seen.add(map_id)
        stack, group = [map_id], [map_id]
        while stack:
            for sibling in sorted(siblings.get(stack.pop(), ())):
                if sibling in known and sibling not in seen:
                    seen.add(sibling)
                    group.append(sibling)
                    stack.append(sibling)
        groups.append(group)
    return groups


def _order_left_to_right(centerlines: List[np.ndarray]) -> List[int]:
    """Indices of lanes ordered from the leftmost to the rightmost of a group.

    Ordered geometrically, by offset along the normal of the shared heading. The
    left/right relations are not used: they are incomplete for roads whose
    neighbouring lanes leave the clip.
    """
    if len(centerlines) < 2:
        return list(range(len(centerlines)))
    heading = np.sum([line[-1, :2] - line[0, :2] for line in centerlines], axis=0)
    norm = float(np.linalg.norm(heading))
    if norm < 1e-9:
        return list(range(len(centerlines)))
    left_normal = np.array([-heading[1], heading[0]]) / norm
    offsets = [float(line[len(line) // 2, :2] @ left_normal) for line in centerlines]
    return sorted(range(len(centerlines)), key=lambda index: -offsets[index])


def _mads_speed_limit_mps(lane: Dict) -> Optional[float]:
    """Lane speed limit in m/s, or None where clipgt leaves it at 0.

    clipgt stores mph: values are either round mph (25, 35, 45, ...) or exact
    mph equivalents of round metric limits (31.0685 = 50 km/h, 43.4959 = 70).
    """
    raw = lane.get("speed_limit")
    try:
        mph = float(raw) if raw is not None else 0.0
    except (TypeError, ValueError):
        return None
    if mph <= 0.0:
        return None
    return mph * 0.44704


def _segment_to_outline(points: np.ndarray, half_width_m: float = 0.5) -> np.ndarray:
    """Thin closed rectangle around a stop-line segment (StopZone wants a surface).

    The caller guarantees the segment has a length to take a normal of.
    """
    p0, p1 = points[0], points[-1]
    direction = p1[:2] - p0[:2]
    norm = float(np.linalg.norm(direction))
    normal = np.array([-direction[1], direction[0]]) / norm * half_width_m
    corners = [
        [p0[0] + normal[0], p0[1] + normal[1], p0[2]],
        [p1[0] + normal[0], p1[1] + normal[1], p1[2]],
        [p1[0] - normal[0], p1[1] - normal[1], p1[2]],
        [p0[0] - normal[0], p0[1] - normal[1], p0[2]],
    ]
    corners.append(corners[0])
    return np.asarray(corners, dtype=np.float64)


def _mads_road_line_type(lane_line: Dict) -> RoadLineType:
    """Majority style+color of a clipgt lane_line mapped to a RoadLineType."""
    styles_raw = lane_line.get("styles")
    colors_raw = lane_line.get("colors")
    styles = [s for s in (list(styles_raw) if styles_raw is not None else []) if s]
    colors = [c for c in (list(colors_raw) if colors_raw is not None else []) if c]
    style = max(set(styles), key=styles.count) if styles else ""
    yellow = (max(set(colors), key=colors.count) if colors else "WHITE") == "YELLOW"
    if style == "SOLID_SINGLE":
        return RoadLineType.SOLID_YELLOW if yellow else RoadLineType.SOLID_WHITE
    if style in ("LONG_DASHED_SINGLE", "SHORT_DASHED_SINGLE"):
        return RoadLineType.DASHED_YELLOW if yellow else RoadLineType.DASHED_WHITE
    if style == "SOLID_GROUP":
        return RoadLineType.DOUBLE_SOLID_YELLOW if yellow else RoadLineType.DOUBLE_SOLID_WHITE
    if style == "DASHED_SOLID":
        return RoadLineType.DASH_SOLID_YELLOW if yellow else RoadLineType.DASH_SOLID_WHITE
    return RoadLineType.UNKNOWN


def _normalized_arclength(polyline: np.ndarray) -> Optional[np.ndarray]:
    """Distance along a polyline, scaled to 0 at its start and 1 at its end."""
    segments = np.linalg.norm(np.diff(polyline[:, :2], axis=0), axis=1)
    total = segments.sum()
    if total <= 0:
        return None
    return np.concatenate([[0.0], np.cumsum(segments)]) / total


def _centerline_from_rails(left: np.ndarray, right: np.ndarray) -> Optional[np.ndarray]:
    """Centerline as the midpoint of the two rails, paired by normalized arc-length.

    The rails rarely share a point count, so index-pairing would skew the center.
    """
    left_arclength, right_arclength = _normalized_arclength(left), _normalized_arclength(right)
    if left_arclength is None or right_arclength is None:
        return None
    grid = np.linspace(0.0, 1.0, max(len(left), len(right)))
    left_points = np.stack([np.interp(grid, left_arclength, left[:, axis]) for axis in range(3)], axis=1)
    right_points = np.stack([np.interp(grid, right_arclength, right[:, axis]) for axis in range(3)], axis=1)
    return (left_points + right_points) / 2.0
