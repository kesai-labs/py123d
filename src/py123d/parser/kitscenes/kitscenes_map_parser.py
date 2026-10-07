"""KITScenes Multimodal map parser for the per-scene Lanelet2 HD maps (``maps/map.osm``)."""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Final, Iterator, List, Optional, Set, Tuple

import numpy as np
import numpy.typing as npt
import pyproj
import shapely
import shapely.geometry as geom
from scipy.spatial import cKDTree

from py123d.datatypes import (
    BaseMapObject,
    Crosswalk,
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
    StopZone,
    StopZoneType,
)
from py123d.geometry import Polyline3D
from py123d.parser.base_dataset_parser import BaseMapParser
from py123d.parser.kitscenes.kitscenes_constants import DATASET_NAME, MAP_FILE, MAP_ORIGIN_FILE

logger = logging.getLogger(__name__)

LANE_TYPE_MAPPING: Final[Dict[str, LaneType]] = {
    "road": LaneType.SURFACE_STREET,
    "highway": LaneType.FREEWAY,
    "bicycle_lane": LaneType.BIKE_LANE,
    "bus_lane": LaneType.BUS_LANE,
    "emergency_lane": LaneType.UNDEFINED,
}

# German road markings are white. Keys are the Lanelet2 ``(type, subtype)`` of a linestring.
ROAD_LINE_TYPE_MAPPING: Final[Dict[Tuple[str, str], RoadLineType]] = {
    **{(line, "solid"): RoadLineType.SOLID_WHITE for line in ("line_thin", "line_thick", "bike_marking")},
    **{(line, "dashed"): RoadLineType.DASHED_WHITE for line in ("line_thin", "line_thick", "bike_marking")},
    **{(line, "dashed_solid"): RoadLineType.DASH_SOLID_WHITE for line in ("line_thin", "line_thick")},
    **{(line, "solid_dashed"): RoadLineType.SOLID_DASH_WHITE for line in ("line_thin", "line_thick")},
    **{(line, "solid_solid"): RoadLineType.DOUBLE_SOLID_WHITE for line in ("line_thin", "line_thick")},
}
ROAD_EDGE_LINE_TYPES: Final[Set[str]] = {"curbstone", "road_border"}

# Traffic sign codes (German traffic code) that make a stop line a stop or yield line.
STOP_SIGN_CODES: Final[Set[str]] = {"de206"}
YIELD_SIGN_CODES: Final[Set[str]] = {"de205"}

# Lanes of opposite directions are emitted twice; the reversed copy gets this ID offset.
REVERSED_LANE_ID_OFFSET: Final[int] = 1_000_000_000
STOP_ZONE_DEPTH_M: Final[float] = 0.5
MIN_CONFLICT_AREA_M2: Final[float] = 0.5
MAX_TRAFFIC_LIGHT_LOOKAHEAD_M: Final[float] = 30.0
CENTERLINE_RESOLUTION_M: Final[float] = 0.1


class KITScenesMapParser(BaseMapParser):
    """Lightweight, picklable handle to the Lanelet2 map of one KITScenes scene."""

    def __init__(self, data_root: Path, split: str, scene_id: str, location: Optional[str]) -> None:
        self._data_root = Path(data_root)
        self._split = split
        self._scene_id = scene_id
        self._location = location

    def get_map_metadata(self) -> MapMetadata:
        """Inherited, see superclass."""
        return get_kitscenes_map_metadata(self._split, self._scene_id, self._location)

    def iter_map_objects(self) -> Iterator[BaseMapObject]:
        """Inherited, see superclass."""
        scene_dir = self._data_root / "data" / self._split / self._scene_id
        lanelet_map = _load_lanelet_map(scene_dir / MAP_FILE, scene_dir / MAP_ORIGIN_FILE)

        lanes = _extract_lanes(lanelet_map)
        lane_groups = _extract_lane_groups(lanes, lanelet_map)
        intersections = _extract_intersections(lanes, lane_groups, lanelet_map)

        yield from _iter_lanes(lanes)
        yield from _iter_lane_groups(lane_groups)
        yield from _iter_intersections(intersections)
        yield from _iter_crosswalks(lanelet_map)
        yield from _iter_road_edges(lanelet_map)
        yield from _iter_road_lines(lanelet_map)
        yield from _iter_stop_zones(lanelet_map, lanes)


def get_kitscenes_map_metadata(split: str, scene_id: str, location: Optional[str]) -> MapMetadata:
    """Map metadata of a KITScenes scene. Every scene ships its own map crop, so maps are stored per log.

    :param split: KITScenes split name without dataset prefix, e.g. ``"val"``.
    :param scene_id: Scene UUID, used as the log name.
    :param location: City of the scene.
    """
    return MapMetadata(
        dataset=DATASET_NAME,
        split=f"{DATASET_NAME}_{split}",
        log_name=scene_id,
        location=location,
        map_has_z=True,
        map_is_per_log=True,
    )


# Lanelet2 OSM loading
# ----------------------------------------------------------------------------------------------------------------------


@dataclass
class _LaneletMap:
    """Raw Lanelet2 primitives with node positions projected to the local metric frame of the scene."""

    points: Dict[int, npt.NDArray[np.float64]]
    way_node_ids: Dict[int, List[int]]
    way_tags: Dict[int, Dict[str, str]]
    relation_tags: Dict[int, Dict[str, str]]
    relation_members: Dict[int, List[Tuple[str, str, int]]]  # (role, member type, ref)

    def way_points(self, way_id: int) -> npt.NDArray[np.float64]:
        return np.array([self.points[node_id] for node_id in self.way_node_ids[way_id]], dtype=np.float64)

    def relations_of_type(self, relation_type: str) -> Iterator[int]:
        for relation_id, tags in self.relation_tags.items():
            if tags.get("type") == relation_type:
                yield relation_id

    def members(self, relation_id: int, role: str) -> List[int]:
        return [ref for member_role, _, ref in self.relation_members[relation_id] if member_role == role]


def _load_lanelet_map(map_path: Path, origin_path: Path) -> _LaneletMap:
    """Load a Lanelet2 ``.osm`` file without the lanelet2 library.

    Positions are projected like Lanelet2's ``UtmProjector``: UTM coordinates minus the UTM coordinates of the map
    origin. This is the frame of the KITScenes ego poses. Heights are the node ``ele`` tags, as in the poses.
    """
    with origin_path.open("r", encoding="utf-8") as file:
        origin = json.load(file)
    utm_zone = int((origin["longitude"] + 180.0) // 6.0) + 1
    hemisphere_offset = 32600 if origin["latitude"] >= 0 else 32700
    transformer = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{hemisphere_offset + utm_zone}", always_xy=True)
    origin_x, origin_y = transformer.transform(origin["longitude"], origin["latitude"])

    root = ET.parse(map_path).getroot()
    node_ids, longitudes, latitudes, elevations = [], [], [], []
    for node in root.iter("node"):
        node_ids.append(int(node.get("id")))
        longitudes.append(float(node.get("lon")))
        latitudes.append(float(node.get("lat")))
        elevation_tag = node.find("tag[@k='ele']")
        elevations.append(float(elevation_tag.get("v")) if elevation_tag is not None else 0.0)
    xs, ys = transformer.transform(np.array(longitudes), np.array(latitudes))
    xyz = np.stack([np.asarray(xs) - origin_x, np.asarray(ys) - origin_y, np.array(elevations)], axis=-1)

    way_node_ids: Dict[int, List[int]] = {}
    way_tags: Dict[int, Dict[str, str]] = {}
    for way in root.iter("way"):
        way_id = int(way.get("id"))
        way_node_ids[way_id] = [int(node_ref.get("ref")) for node_ref in way.iter("nd")]
        way_tags[way_id] = {tag.get("k"): tag.get("v") for tag in way.iter("tag")}

    relation_tags: Dict[int, Dict[str, str]] = {}
    relation_members: Dict[int, List[Tuple[str, str, int]]] = {}
    for relation in root.iter("relation"):
        relation_id = int(relation.get("id"))
        relation_tags[relation_id] = {tag.get("k"): tag.get("v") for tag in relation.iter("tag")}
        relation_members[relation_id] = [
            (member.get("role"), member.get("type"), int(member.get("ref"))) for member in relation.iter("member")
        ]

    return _LaneletMap(
        points=dict(zip(node_ids, xyz)),
        way_node_ids=way_node_ids,
        way_tags=way_tags,
        relation_tags=relation_tags,
        relation_members=relation_members,
    )


# Lanes
# ----------------------------------------------------------------------------------------------------------------------


@dataclass
class _LaneData:
    lane_id: int
    lanelet_id: int
    lane_type: LaneType
    left_way_id: int
    right_way_id: int
    left_node_ids: Tuple[int, ...]
    right_node_ids: Tuple[int, ...]
    left_boundary: Polyline3D
    right_boundary: Polyline3D
    speed_limit_mps: Optional[float]
    polygon: shapely.Polygon
    left_lane_id: Optional[int] = None
    right_lane_id: Optional[int] = None
    predecessor_ids: List[int] = field(default_factory=list)
    successor_ids: List[int] = field(default_factory=list)
    lane_group_id: Optional[int] = None


def _orient_boundaries(
    left_node_ids: List[int], right_node_ids: List[int], lanelet_map: _LaneletMap
) -> Tuple[List[int], List[int]]:
    """Orient both boundaries in driving direction.

    Linestrings in Lanelet2 files may be stored in either direction, since neighboring lanelets share them. The
    boundaries are first aligned with each other. The driving direction is then the one in which the ``left``
    boundary lies on the left side.
    """
    left = np.array([lanelet_map.points[node_id][:2] for node_id in left_node_ids])
    right = np.array([lanelet_map.points[node_id][:2] for node_id in right_node_ids])
    aligned_distance = np.linalg.norm(left[0] - right[0]) + np.linalg.norm(left[-1] - right[-1])
    crossed_distance = np.linalg.norm(left[0] - right[-1]) + np.linalg.norm(left[-1] - right[0])
    if crossed_distance < aligned_distance:
        right_node_ids, right = right_node_ids[::-1], right[::-1]

    direction = (left[-1] + right[-1]) - (left[0] + right[0])
    left_to_right = right.mean(axis=0) - left.mean(axis=0)
    if direction[0] * left_to_right[1] - direction[1] * left_to_right[0] > 0:  # left boundary is on the right side
        left_node_ids, right_node_ids = left_node_ids[::-1], right_node_ids[::-1]
    return list(left_node_ids), list(right_node_ids)


def _is_bidirectional(tags: Dict[str, str]) -> bool:
    return tags.get("one_way", "yes").lower() in ("no", "0", "false")


def _parse_speed_limit_mps(tags: Dict[str, str]) -> Optional[float]:
    speed_limit = tags.get("speed_limit")
    if speed_limit is None:
        return None
    try:
        return float(speed_limit.lower().replace("km/h", "").strip()) / 3.6
    except ValueError:
        return None


def _extract_lanes(lanelet_map: _LaneletMap) -> Dict[int, _LaneData]:
    """Build directed lanes from lanelets and derive their topology from shared nodes and linestrings.

    Lanelet2 files store no explicit topology. A lane succeeds another if its boundaries start at the end nodes of
    the other's boundaries. Lanes are neighbors if they share a boundary linestring in the same direction.
    """
    lanes: Dict[int, _LaneData] = {}
    for lanelet_id in lanelet_map.relations_of_type("lanelet"):
        tags = lanelet_map.relation_tags[lanelet_id]
        lane_type = LANE_TYPE_MAPPING.get(tags.get("subtype", "road"))
        if lane_type is None:
            continue  # e.g. crosswalks and walkways
        left_way_id = lanelet_map.members(lanelet_id, "left")[0]
        right_way_id = lanelet_map.members(lanelet_id, "right")[0]
        left_node_ids, right_node_ids = _orient_boundaries(
            lanelet_map.way_node_ids[left_way_id], lanelet_map.way_node_ids[right_way_id], lanelet_map
        )

        directions = [(lanelet_id, left_node_ids, right_node_ids, left_way_id, right_way_id)]
        if _is_bidirectional(tags):
            directions.append(
                (
                    lanelet_id + REVERSED_LANE_ID_OFFSET,
                    right_node_ids[::-1],
                    left_node_ids[::-1],
                    right_way_id,
                    left_way_id,
                )
            )

        for lane_id, left_ids, right_ids, left_way, right_way in directions:
            left_boundary = Polyline3D.from_array(np.array([lanelet_map.points[node_id] for node_id in left_ids]))
            right_boundary = Polyline3D.from_array(np.array([lanelet_map.points[node_id] for node_id in right_ids]))
            lanes[lane_id] = _LaneData(
                lane_id=lane_id,
                lanelet_id=lanelet_id,
                lane_type=lane_type,
                left_way_id=left_way,
                right_way_id=right_way,
                left_node_ids=tuple(left_ids),
                right_node_ids=tuple(right_ids),
                left_boundary=left_boundary,
                right_boundary=right_boundary,
                speed_limit_mps=_parse_speed_limit_mps(tags),
                polygon=_boundaries_to_polygon(left_boundary, right_boundary),
            )

    lanes_by_start: Dict[Tuple[int, int], List[int]] = {}
    lanes_by_right_boundary: Dict[Tuple[int, ...], int] = {}
    for lane in lanes.values():
        lanes_by_start.setdefault((lane.left_node_ids[0], lane.right_node_ids[0]), []).append(lane.lane_id)
        lanes_by_right_boundary[lane.right_node_ids] = lane.lane_id

    for lane in lanes.values():
        lane.successor_ids = lanes_by_start.get((lane.left_node_ids[-1], lane.right_node_ids[-1]), [])
        for successor_id in lane.successor_ids:
            lanes[successor_id].predecessor_ids.append(lane.lane_id)
        left_lane_id = lanes_by_right_boundary.get(lane.left_node_ids)
        if left_lane_id is not None and left_lane_id != lane.lane_id:
            lane.left_lane_id = left_lane_id
            lanes[left_lane_id].right_lane_id = lane.lane_id
    return lanes


def _boundaries_to_polygon(left_boundary: Polyline3D, right_boundary: Polyline3D) -> shapely.Polygon:
    outline = np.vstack([left_boundary.array[:, :2], right_boundary.array[::-1, :2]])
    return shapely.make_valid(geom.Polygon(outline)).buffer(0)


def _get_centerline_from_boundaries(left_boundary: Polyline3D, right_boundary: Polyline3D) -> Polyline3D:
    num_points = max(int(np.ceil(max(left_boundary.length, right_boundary.length) / CENTERLINE_RESOLUTION_M)), 2)
    left_array = left_boundary.interpolate(np.linspace(0, left_boundary.length, num_points, endpoint=True))
    right_array = right_boundary.interpolate(np.linspace(0, right_boundary.length, num_points, endpoint=True))
    return Polyline3D.from_array(np.mean([left_array, right_array], axis=0))


# Lane groups & intersections
# ----------------------------------------------------------------------------------------------------------------------


@dataclass
class _LaneGroupData:
    lane_group_id: int
    lane_ids: List[int]
    left_boundary: Polyline3D
    right_boundary: Polyline3D
    predecessor_ids: List[int] = field(default_factory=list)
    successor_ids: List[int] = field(default_factory=list)
    intersection_id: Optional[int] = None


def _extract_lane_groups(lanes: Dict[int, _LaneData], lanelet_map: _LaneletMap) -> Dict[int, _LaneGroupData]:
    """Group same-direction neighbors, ordered from left to right. Groups are not joined across curbs.

    Overlapping lanelets, e.g. where lanes merge or split, can share a boundary. The neighbor of a lane then does not
    point back at it, so lanes are only grouped if they are each other's neighbor. Every lane is in exactly one group.
    """
    right_neighbor: Dict[int, int] = {}
    for lane in lanes.values():
        right_id = lane.right_lane_id
        if right_id is None or lanes[right_id].left_lane_id != lane.lane_id:
            continue
        if lanelet_map.way_tags[lane.right_way_id].get("type") in ROAD_EDGE_LINE_TYPES:
            continue
        right_neighbor[lane.lane_id] = right_id
    left_neighbor = {right_id: left_id for left_id, right_id in right_neighbor.items()}

    lane_groups: Dict[int, _LaneGroupData] = {}
    for lane_id in sorted(lanes):
        if lanes[lane_id].lane_group_id is not None:
            continue
        leftmost_id = lane_id
        visited = {lane_id}
        while True:
            left_id = left_neighbor.get(leftmost_id)
            if left_id is None or left_id in visited:
                break
            visited.add(left_id)
            leftmost_id = left_id

        group_lane_ids = [leftmost_id]
        while True:
            right_id = right_neighbor.get(group_lane_ids[-1])
            if right_id is None or right_id in group_lane_ids:
                break
            group_lane_ids.append(right_id)

        lane_group_id = len(lane_groups)
        for group_lane_id in group_lane_ids:
            lanes[group_lane_id].lane_group_id = lane_group_id
        lane_groups[lane_group_id] = _LaneGroupData(
            lane_group_id=lane_group_id,
            lane_ids=group_lane_ids,
            left_boundary=lanes[group_lane_ids[0]].left_boundary,
            right_boundary=lanes[group_lane_ids[-1]].right_boundary,
        )

    for lane_group in lane_groups.values():
        predecessor_ids = {lanes[p].lane_group_id for i in lane_group.lane_ids for p in lanes[i].predecessor_ids}
        successor_ids = {lanes[s].lane_group_id for i in lane_group.lane_ids for s in lanes[i].successor_ids}
        lane_group.predecessor_ids = sorted(predecessor_ids - {lane_group.lane_group_id})
        lane_group.successor_ids = sorted(successor_ids - {lane_group.lane_group_id})
    return lane_groups


@dataclass
class _IntersectionData:
    intersection_id: int
    intersection_type: IntersectionType
    lane_group_ids: List[int]
    outline: Polyline3D


def _extract_intersections(
    lanes: Dict[int, _LaneData], lane_groups: Dict[int, _LaneGroupData], lanelet_map: _LaneletMap
) -> Dict[int, _IntersectionData]:
    """Derive intersections, since Lanelet2 has no intersection primitive.

    A lane is part of an intersection if it overlaps another lane that is neither in its lane group, nor directly
    connected to it, nor branching from or merging with it. Overlapping intersection lane groups are merged into one
    intersection. It is controlled by a traffic light if a lane referencing one leads into it.
    """
    lane_ids = list(lanes)
    tree = shapely.STRtree([lanes[lane_id].polygon for lane_id in lane_ids])
    intersection_lane_ids: Set[int] = set()
    for index, lane_id in enumerate(lane_ids):
        lane = lanes[lane_id]
        for other_index in tree.query(lane.polygon, predicate="intersects"):
            other = lanes[lane_ids[other_index]]
            if other_index <= index or other.lane_group_id == lane.lane_group_id:
                continue
            if other.lanelet_id == lane.lanelet_id:
                continue  # the two directions of a bidirectional lanelet
            if other.lane_id in lane.successor_ids or other.lane_id in lane.predecessor_ids:
                continue
            if set(other.predecessor_ids) & set(lane.predecessor_ids) or set(other.successor_ids) & set(
                lane.successor_ids
            ):
                continue
            if lane.polygon.intersection(other.polygon).area >= MIN_CONFLICT_AREA_M2:
                intersection_lane_ids.update((lane.lane_id, other.lane_id))

    intersection_group_ids = sorted({lanes[lane_id].lane_group_id for lane_id in intersection_lane_ids})
    group_polygons = {
        group_id: shapely.union_all([lanes[lane_id].polygon for lane_id in lane_groups[group_id].lane_ids])
        for group_id in intersection_group_ids
    }
    merged = shapely.union_all(list(group_polygons.values()))
    polygons = list(merged.geoms) if isinstance(merged, geom.MultiPolygon) else [merged] if not merged.is_empty else []

    traffic_light_lanelet_ids = _lanelets_with_regulatory_element(lanelet_map, "traffic_light")
    boundary_points = np.concatenate(
        [np.vstack([lane.left_boundary.array, lane.right_boundary.array]) for lane in lanes.values()]
    )
    boundary_tree = cKDTree(boundary_points[:, :2])

    intersections: Dict[int, _IntersectionData] = {}
    for intersection_id, polygon in enumerate(polygons):
        member_group_ids = [
            group_id for group_id, group_polygon in group_polygons.items() if polygon.intersects(group_polygon)
        ]
        for group_id in member_group_ids:
            lane_groups[group_id].intersection_id = intersection_id

        outline_xy = np.array(polygon.exterior.coords)[:, :2]
        _, nearest = boundary_tree.query(outline_xy)
        outline = Polyline3D.from_array(np.column_stack([outline_xy, boundary_points[nearest, 2]]))
        intersections[intersection_id] = _IntersectionData(
            intersection_id=intersection_id,
            intersection_type=IntersectionType.DEFAULT,
            lane_group_ids=member_group_ids,
            outline=outline,
        )

    # Stop lines of traffic lights are usually a few meters before the intersection, e.g. behind a crosswalk. Follow
    # the lanes of each traffic light forward until they enter an intersection.
    for lane in lanes.values():
        if lane.lanelet_id not in traffic_light_lanelet_ids:
            continue
        frontier: List[Tuple[int, float]] = [(lane.lane_id, 0.0)]  # (lane ID, distance driven before the lane)
        visited: Set[int] = set()
        while frontier:
            lane_id, distance = frontier.pop()
            if lane_id in visited or distance > MAX_TRAFFIC_LIGHT_LOOKAHEAD_M:
                continue
            visited.add(lane_id)
            intersection_id = lane_groups[lanes[lane_id].lane_group_id].intersection_id
            if intersection_id is not None:
                intersections[intersection_id].intersection_type = IntersectionType.TRAFFIC_LIGHT
                continue
            next_distance = distance + lanes[lane_id].left_boundary.length
            frontier.extend((successor_id, next_distance) for successor_id in lanes[lane_id].successor_ids)
    return intersections


def _lanelets_with_regulatory_element(lanelet_map: _LaneletMap, subtype: str) -> Set[int]:
    lanelet_ids: Set[int] = set()
    for lanelet_id in lanelet_map.relations_of_type("lanelet"):
        for regulatory_element_id in lanelet_map.members(lanelet_id, "regulatory_element"):
            if lanelet_map.relation_tags.get(regulatory_element_id, {}).get("subtype") == subtype:
                lanelet_ids.add(lanelet_id)
    return lanelet_ids


# Map object iterators
# ----------------------------------------------------------------------------------------------------------------------


def _iter_lanes(lanes: Dict[int, _LaneData]) -> Iterator[Lane]:
    for lane in lanes.values():
        yield Lane(
            object_id=lane.lane_id,
            lane_type=lane.lane_type,
            lane_group_id=lane.lane_group_id,
            left_boundary=lane.left_boundary,
            right_boundary=lane.right_boundary,
            centerline=_get_centerline_from_boundaries(lane.left_boundary, lane.right_boundary),
            left_lane_id=lane.left_lane_id,
            right_lane_id=lane.right_lane_id,
            predecessor_ids=lane.predecessor_ids,
            successor_ids=lane.successor_ids,
            speed_limit_mps=lane.speed_limit_mps,
        )


def _iter_lane_groups(lane_groups: Dict[int, _LaneGroupData]) -> Iterator[LaneGroup]:
    for lane_group in lane_groups.values():
        yield LaneGroup(
            object_id=lane_group.lane_group_id,
            lane_ids=lane_group.lane_ids,
            left_boundary=lane_group.left_boundary,
            right_boundary=lane_group.right_boundary,
            intersection_id=lane_group.intersection_id,
            predecessor_ids=lane_group.predecessor_ids,
            successor_ids=lane_group.successor_ids,
        )


def _iter_intersections(intersections: Dict[int, _IntersectionData]) -> Iterator[Intersection]:
    for intersection in intersections.values():
        yield Intersection(
            object_id=intersection.intersection_id,
            intersection_type=intersection.intersection_type,
            lane_group_ids=intersection.lane_group_ids,
            outline=intersection.outline,
        )


def _iter_crosswalks(lanelet_map: _LaneletMap) -> Iterator[Crosswalk]:
    for lanelet_id in lanelet_map.relations_of_type("lanelet"):
        if lanelet_map.relation_tags[lanelet_id].get("subtype") != "crosswalk":
            continue
        left_node_ids, right_node_ids = _orient_boundaries(
            lanelet_map.way_node_ids[lanelet_map.members(lanelet_id, "left")[0]],
            lanelet_map.way_node_ids[lanelet_map.members(lanelet_id, "right")[0]],
            lanelet_map,
        )
        node_ids = left_node_ids + right_node_ids[::-1] + left_node_ids[:1]
        yield Crosswalk(
            object_id=lanelet_id,
            outline=Polyline3D.from_array(np.array([lanelet_map.points[node_id] for node_id in node_ids])),
        )


def _iter_road_edges(lanelet_map: _LaneletMap) -> Iterator[RoadEdge]:
    for way_id, tags in lanelet_map.way_tags.items():
        if tags.get("type") in ROAD_EDGE_LINE_TYPES and len(lanelet_map.way_node_ids[way_id]) >= 2:
            yield RoadEdge(
                object_id=way_id,
                road_edge_type=RoadEdgeType.ROAD_EDGE_BOUNDARY,
                polyline=Polyline3D.from_array(lanelet_map.way_points(way_id)),
            )


def _iter_road_lines(lanelet_map: _LaneletMap) -> Iterator[RoadLine]:
    for way_id, tags in lanelet_map.way_tags.items():
        road_line_type = ROAD_LINE_TYPE_MAPPING.get((tags.get("type", ""), tags.get("subtype", "")))
        if road_line_type is not None and len(lanelet_map.way_node_ids[way_id]) >= 2:
            yield RoadLine(
                object_id=way_id,
                road_line_type=road_line_type,
                polyline=Polyline3D.from_array(lanelet_map.way_points(way_id)),
            )


def _iter_stop_zones(lanelet_map: _LaneletMap, lanes: Dict[int, _LaneData]) -> Iterator[StopZone]:
    """Convert stop lines referenced by traffic lights, stop/yield signs or right-of-way rules into stop zones."""
    lane_ids_by_lanelet: Dict[int, List[int]] = {}
    for lane in lanes.values():
        lane_ids_by_lanelet.setdefault(lane.lanelet_id, []).append(lane.lane_id)

    regulated_lanelets: Dict[int, List[int]] = {}
    for lanelet_id in lanelet_map.relations_of_type("lanelet"):
        for regulatory_element_id in lanelet_map.members(lanelet_id, "regulatory_element"):
            regulated_lanelets.setdefault(regulatory_element_id, []).append(lanelet_id)

    stop_zones: Dict[int, Tuple[StopZoneType, Set[int]]] = {}  # stop line way ID -> (type, lane IDs)
    for regulatory_element_id in lanelet_map.relations_of_type("regulatory_element"):
        subtype = lanelet_map.relation_tags[regulatory_element_id].get("subtype")
        lanelet_ids = regulated_lanelets.get(regulatory_element_id, [])
        if subtype == "traffic_light":
            stop_zone_type = StopZoneType.TRAFFIC_LIGHT
        elif subtype == "traffic_sign":
            sign_codes = {
                lanelet_map.way_tags.get(way_id, {}).get("subtype")
                for way_id in lanelet_map.members(regulatory_element_id, "refers")
            }
            if sign_codes & STOP_SIGN_CODES:
                stop_zone_type = StopZoneType.STOP_SIGN
            elif sign_codes & YIELD_SIGN_CODES:
                stop_zone_type = StopZoneType.YIELD_SIGN
            else:
                continue
        elif subtype == "right_of_way":
            stop_zone_type = StopZoneType.YIELD_SIGN
            lanelet_ids = lanelet_map.members(regulatory_element_id, "yield")
        else:
            continue

        lane_ids = {lane_id for lanelet_id in lanelet_ids for lane_id in lane_ids_by_lanelet.get(lanelet_id, [])}
        for stop_line_id in lanelet_map.members(regulatory_element_id, "ref_line"):
            if stop_line_id not in lanelet_map.way_node_ids:
                continue
            previous_type, previous_lane_ids = stop_zones.get(stop_line_id, (stop_zone_type, set()))
            # A stop line shared by several rules keeps the strongest control, a traffic light first.
            if previous_type == StopZoneType.TRAFFIC_LIGHT:
                stop_zone_type = previous_type
            stop_zones[stop_line_id] = (stop_zone_type, previous_lane_ids | lane_ids)

    for stop_line_id, (stop_zone_type, lane_ids) in stop_zones.items():
        stop_line = lanelet_map.way_points(stop_line_id)
        polygon = geom.LineString(stop_line[:, :2]).buffer(STOP_ZONE_DEPTH_M / 2, cap_style="flat")
        if polygon.is_empty or not isinstance(polygon, geom.Polygon):
            continue
        outline_xy = np.array(polygon.exterior.coords)[:, :2]
        outline = np.column_stack([outline_xy, np.full(len(outline_xy), stop_line[:, 2].mean())])
        yield StopZone(
            object_id=stop_line_id,
            stop_zone_type=stop_zone_type,
            outline=Polyline3D.from_array(outline),
            lane_ids=sorted(lane_ids),
        )
