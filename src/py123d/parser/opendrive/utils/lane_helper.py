from dataclasses import dataclass
from functools import cached_property
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import shapely

from py123d.geometry import PoseSE2Index
from py123d.geometry.polyline import Polyline3D, PolylineSE2
from py123d.geometry.utils.units import kmph_to_mps, mph_to_mps
from py123d.parser.opendrive.utils.id_system import (
    derive_lane_group_id,
    derive_lane_id,
    lane_group_id_from_lane_id,
)
from py123d.parser.opendrive.xodr_parser.lane import XODRLane, XODRLaneSection
from py123d.parser.opendrive.xodr_parser.reference import XODRReferenceLine
from py123d.parser.opendrive.xodr_parser.road import XODRRoadType


@dataclass
class OpenDriveLaneHelper:
    lane_id: str
    open_drive_lane: XODRLane
    s_inner_offset: float
    s_range: Tuple[float, float]
    inner_boundary: XODRReferenceLine
    outer_boundary: XODRReferenceLine
    speed_limit_mps: Optional[float]
    interpolation_step_size: float

    # lazy loaded
    predecessor_lane_ids: Optional[List[str]] = None
    successor_lane_ids: Optional[List[str]] = None

    def __post_init__(self):
        self.predecessor_lane_ids: List[str] = []
        self.successor_lane_ids: List[str] = []

    @property
    def id(self) -> int:
        return self.open_drive_lane.id

    @property
    def type(self) -> str:
        return self.open_drive_lane.type

    @cached_property
    def _s_positions(self) -> npt.NDArray[np.float64]:
        length = self.s_range[1] - self.s_range[0]
        _s_positions = np.linspace(
            start=self.s_range[0],
            stop=self.s_range[1],
            num=int(np.ceil(length / self.interpolation_step_size)) + 1,
            endpoint=True,
            dtype=np.float64,
        )
        _s_positions[..., -1] = np.clip(_s_positions[..., -1], 0.0, self.s_range[-1])
        return _s_positions

    @cached_property
    def _lane_section_end_mask(self) -> npt.NDArray[np.float64]:
        lane_section_end_mask = np.zeros(len(self._s_positions), dtype=bool)
        lane_section_end_mask[-1] = True
        return lane_section_end_mask

    @cached_property
    def inner_polyline_se2(self) -> PolylineSE2:
        s_arr = self.s_inner_offset + self._s_positions - self.s_range[0]
        t_arr = np.zeros_like(s_arr)
        inner_polyline = self.inner_boundary.interpolate_se2_batch(s_arr, t_arr, self._lane_section_end_mask)
        polyline_array = np.flip(inner_polyline, axis=0) if self.id > 0 else inner_polyline
        return PolylineSE2.from_array(polyline_array)

    @cached_property
    def inner_polyline_3d(self) -> Polyline3D:
        s_arr = self.s_inner_offset + self._s_positions - self.s_range[0]
        t_arr = np.zeros_like(s_arr)
        inner_polyline = self.inner_boundary.interpolate_3d_batch(s_arr, t_arr, self._lane_section_end_mask)
        polyline_array = np.flip(inner_polyline, axis=0) if self.id > 0 else inner_polyline
        return Polyline3D.from_array(polyline_array)

    @cached_property
    def outer_polyline_se2(self) -> PolylineSE2:
        s_arr = self._s_positions - self.s_range[0]
        t_arr = np.zeros_like(s_arr)
        outer_polyline = self.outer_boundary.interpolate_se2_batch(s_arr, t_arr, self._lane_section_end_mask)
        polyline_array = np.flip(outer_polyline, axis=0) if self.id > 0 else outer_polyline
        return PolylineSE2.from_array(polyline_array)

    @cached_property
    def outer_polyline_3d(self) -> Polyline3D:
        s_arr = self._s_positions - self.s_range[0]
        t_arr = np.zeros_like(s_arr)
        outer_polyline = self.outer_boundary.interpolate_3d_batch(s_arr, t_arr, self._lane_section_end_mask)
        polyline_array = np.flip(outer_polyline, axis=0) if self.id > 0 else outer_polyline
        return Polyline3D.from_array(polyline_array)

    @cached_property
    def center_polyline_se2(self) -> PolylineSE2:
        return PolylineSE2.from_array(
            np.concatenate(
                [
                    self.inner_polyline_se2.array[None, ...],
                    self.outer_polyline_se2.array[None, ...],
                ],
                axis=0,
            ).mean(axis=0)
        )

    @cached_property
    def center_polyline_3d(self) -> Polyline3D:
        return Polyline3D.from_array(
            np.concatenate(
                [
                    self.outer_polyline_3d.array[None, ...],
                    self.inner_polyline_3d.array[None, ...],
                ],
                axis=0,
            ).mean(axis=0)
        )

    @property
    def outline_polyline_3d(self) -> Polyline3D:
        inner_polyline = self.inner_polyline_3d.array
        outer_polyline = self.outer_polyline_3d.array[::-1]
        return Polyline3D.from_array(
            np.concatenate(
                [
                    inner_polyline,
                    outer_polyline,
                    inner_polyline[None, 0],
                ],
                axis=0,
                dtype=np.float64,
            )
        )

    @property
    def shapely_polygon(self) -> shapely.Polygon:
        inner_polyline = self.inner_polyline_se2[..., PoseSE2Index.XY]
        outer_polyline = self.outer_polyline_se2[..., PoseSE2Index.XY][::-1]
        polygon_exterior = np.concatenate(
            [
                inner_polyline,
                outer_polyline,
                inner_polyline[None, 0],
            ],
            axis=0,
            dtype=np.float64,
        )
        return shapely.Polygon(polygon_exterior)


@dataclass
class OpenDriveLaneGroupHelper:
    lane_group_id: str
    lane_helpers: List[OpenDriveLaneHelper]

    # loaded during __post_init__
    predecessor_lane_group_ids: Optional[List[str]] = None
    successor_lane_group_ids: Optional[List[str]] = None
    junction_id: Optional[int] = None

    def __post_init__(self):
        predecessor_lane_group_ids = []
        successor_lane_group__ids = []
        for lane_helper in self.lane_helpers:
            for predecessor_lane_id in lane_helper.predecessor_lane_ids:
                predecessor_lane_group_ids.append(lane_group_id_from_lane_id(predecessor_lane_id))
            for successor_lane_id in lane_helper.successor_lane_ids:
                successor_lane_group__ids.append(lane_group_id_from_lane_id(successor_lane_id))
        self.predecessor_lane_group_ids: List[str] = list(set(predecessor_lane_group_ids))
        self.successor_lane_group_ids: List[str] = list(set(successor_lane_group__ids))

        assert len(set([lane_group_id_from_lane_id(lane_helper.lane_id) for lane_helper in self.lane_helpers])) == 1

    def _get_inner_lane_helper(self) -> OpenDriveLaneHelper:
        lane_helper_ids = [lane_helper.open_drive_lane.id for lane_helper in self.lane_helpers]
        inner_lane_helper_idx = np.argmin(lane_helper_ids) if lane_helper_ids[0] > 0 else np.argmax(lane_helper_ids)
        return self.lane_helpers[inner_lane_helper_idx]

    def _get_outer_lane_helper(self) -> OpenDriveLaneHelper:
        lane_helper_ids = [lane_helper.open_drive_lane.id for lane_helper in self.lane_helpers]
        outer_lane_helper_idx = np.argmax(lane_helper_ids) if lane_helper_ids[0] > 0 else np.argmin(lane_helper_ids)
        return self.lane_helpers[outer_lane_helper_idx]

    @cached_property
    def inner_polyline_se2(self) -> PolylineSE2:
        return self._get_inner_lane_helper().inner_polyline_se2

    @cached_property
    def outer_polyline_se2(self) -> PolylineSE2:
        return self._get_outer_lane_helper().outer_polyline_se2

    @cached_property
    def inner_polyline_3d(self) -> Polyline3D:
        return self._get_inner_lane_helper().inner_polyline_3d

    @cached_property
    def outer_polyline_3d(self) -> Polyline3D:
        return self._get_outer_lane_helper().outer_polyline_3d

    @property
    def outline_polyline_3d(self) -> Polyline3D:
        inner_polyline = self.inner_polyline_3d.array
        outer_polyline = self.outer_polyline_3d.array[::-1]
        return Polyline3D.from_array(
            np.concatenate(
                [
                    inner_polyline,
                    outer_polyline,
                    inner_polyline[None, 0],
                ],
                axis=0,
                dtype=np.float64,
            )
        )

    @property
    def shapely_polygon(self) -> shapely.Polygon:
        inner_polyline = self.inner_polyline_se2[..., PoseSE2Index.XY]
        outer_polyline = self.outer_polyline_se2[..., PoseSE2Index.XY][::-1]
        polygon_exterior = np.concatenate(
            [
                inner_polyline,
                outer_polyline,
                inner_polyline[None, 0],
            ],
            axis=0,
            dtype=np.float64,
        )
        return shapely.Polygon(polygon_exterior)


def lane_section_to_lane_helpers(
    lane_section_id: str,
    lane_section: XODRLaneSection,
    reference_line: XODRReferenceLine,
    s_min: float,
    s_max: float,
    road_types: List[XODRRoadType],
    interpolation_step_size: float,
) -> Dict[str, OpenDriveLaneHelper]:
    lane_helpers: Dict[str, OpenDriveLaneHelper] = {}

    for lanes, t_sign, side in zip([lane_section.left_lanes, lane_section.right_lanes], [1.0, -1.0], ["left", "right"]):
        lane_group_id = derive_lane_group_id(lane_section_id, side)
        lane_boundaries = [reference_line]
        for lane in lanes:
            lane_id = derive_lane_id(lane_group_id, lane.id)
            s_inner_offset = lane_section.s if len(lane_boundaries) == 1 else 0.0
            lane_boundaries.append(
                XODRReferenceLine.from_reference_line(
                    reference_line=lane_boundaries[-1],
                    widths=lane.widths,
                    s_offset=s_inner_offset,
                    t_sign=t_sign,
                )
            )
            lane_helper = OpenDriveLaneHelper(
                lane_id=lane_id,
                open_drive_lane=lane,
                s_inner_offset=s_inner_offset,
                s_range=(s_min, s_max),
                inner_boundary=lane_boundaries[-2],
                outer_boundary=lane_boundaries[-1],
                speed_limit_mps=_get_speed_limit_mps(s_min, road_types),
                interpolation_step_size=interpolation_step_size,
            )
            lane_helpers[lane_id] = lane_helper

    return lane_helpers


def _get_speed_limit_mps(s: float, road_types: List[XODRRoadType]) -> Optional[float]:
    # NOTE: Likely not correct way to extract speed limit from CARLA maps, but serves as a placeholder
    speed_limit_mps: Optional[float] = None
    s_road_types = [road_type.s for road_type in road_types] + [float("inf")]

    if len(road_types) > 0:
        for road_type_idx, road_type in enumerate(road_types):
            if s >= road_type.s and s < s_road_types[road_type_idx + 1]:
                if road_type.speed is not None:
                    if road_type.speed.unit == "mps":
                        speed_limit_mps = road_type.speed.max
                    elif road_type.speed.unit == "km/h":
                        speed_limit_mps = kmph_to_mps(road_type.speed.max)
                    elif road_type.speed.unit == "mph":
                        speed_limit_mps = mph_to_mps(road_type.speed.max)
                break
    return speed_limit_mps
