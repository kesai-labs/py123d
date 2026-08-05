from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple, Union
from zipfile import ZipFile

import numpy as np
import pandas as pd
from typing_extensions import override

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

logger = logging.getLogger(__name__)

# Wait lines mark where traffic enters an intersection or crossing, where it leaves,
# or neither. Only the entering ones oblige traffic to stop.
_STOPPING_WAIT_LINE_SUBTYPES = frozenset({"ENTRY", "CROSSWALK_ENTRY"})
_PASSING_WAIT_LINE_SUBTYPES = frozenset({"EXIT", "NOT_APPLICABLE", "BUFFER_ZONE"})


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


def _clipgt_member(layer: str) -> str:
    """Archive member holding one clipgt map layer."""
    return f"clipgt/{layer}.parquet"


def _has_clipgt_layers(member_names: Set[str]) -> bool:
    """True when a USDZ carries the clipgt layers needed to build a map."""
    return all(_clipgt_member(layer) in member_names for layer in ("lane", "road_boundary"))


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


class NuRecMapParser(BaseMapParser):
    """Map parser for one NuRec USDZ scene, built from the MADS clipgt layers.

    Coordinates are already in the clip-local rig frame, so no geo-projection is
    needed, and polylines are emitted at source resolution.

    The raw lane rails are not re-emitted as road lines: they are already the Lane
    boundaries, and adjacent lanes share a rail, so every interior boundary would
    appear twice.
    """

    def __init__(self, usdz_path: Union[str, Path], location: str) -> None:
        self._usdz_path = Path(usdz_path)
        self._location = location

    @override
    def get_map_metadata(self) -> MapMetadata:
        """Inherited, see superclass."""
        return MapMetadata(
            dataset="nurec",
            location=self._location,
            map_has_z=True,
            map_is_per_log=False,
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
        return relations

    @override
    def iter_map_objects(self) -> Iterator[BaseMapObject]:
        """Inherited, see superclass."""
        with ZipFile(self._usdz_path) as archive:
            if not _has_clipgt_layers(set(archive.namelist())):
                raise ValueError(
                    f"NuRec map {self._location}: no clipgt map layers, which are the only map source supported"
                )
            lane_rows = self._read_layer_rows(archive, "lane")
            boundaries = self._read_layer(archive, "road_boundary")
            relations = self._read_associations(archive)
            crosswalks = self._read_layer(archive, "crosswalk")
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

        for lane_line in lane_lines:
            pts = _mads_points_xyz(lane_line, "line_rail")
            if pts is None or len(pts) < 2:
                continue
            yield RoadLine(
                object_id=next_id,
                road_line_type=_mads_road_line_type(lane_line),
                polyline=Polyline3D.from_array(pts),
            )
            next_id += 1

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
