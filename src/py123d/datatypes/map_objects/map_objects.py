from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Sequence, Union

import numpy as np
import shapely.geometry as geom
from trimesh import Trimesh

from py123d.datatypes.map_objects.base_map_objects import BaseMapLineObject, BaseMapSurfaceObject, MapObjectIDType
from py123d.datatypes.map_objects.map_layer_types import MapLayer, RoadEdgeType, RoadLineType, StopZoneType
from py123d.datatypes.map_objects.utils import get_trimesh_from_boundaries
from py123d.geometry import Polyline2D, Polyline3D

if TYPE_CHECKING:
    from py123d.api import MapAPI


class Lane(BaseMapSurfaceObject):
    """Class representing a lane in a map."""

    __slots__ = (
        "_lane_group_id",
        "_left_boundary",
        "_right_boundary",
        "_centerline",
        "_left_lane_id",
        "_right_lane_id",
        "_predecessor_ids",
        "_successor_ids",
        "_speed_limit_mps",
        "_map_api",
    )

    def __init__(
        self,
        object_id: MapObjectIDType,
        lane_group_id: MapObjectIDType,
        left_boundary: Polyline3D,
        right_boundary: Polyline3D,
        centerline: Polyline3D,
        left_lane_id: Optional[MapObjectIDType] = None,
        right_lane_id: Optional[MapObjectIDType] = None,
        predecessor_ids: List[MapObjectIDType] = [],
        successor_ids: List[MapObjectIDType] = [],
        speed_limit_mps: Optional[float] = None,
        outline: Optional[Polyline3D] = None,
        shapely_polygon: Optional[geom.Polygon] = None,
        map_api: Optional["MapAPI"] = None,
    ) -> None:
        """Initialize a :class:`Lane` instance.

        Notes
        -----
        If the map_api is provided, neighboring lanes and lane group can be accessed through the properties.
        If the outline is not provided, it will be constructed from the left and right boundaries.
        If the shapely_polygon is not provided, it will be constructed from the outline.

        :param object_id: The unique identifier for the lane.
        :param lane_group_id: The unique identifier for the lane group this lane belongs to.
        :param left_boundary: Polyline of left boundary of the lane.
        :param right_boundary: Polyline of right boundary of the lane.
        :param centerline: Polyline of centerline of the lane.
        :param left_lane_id: The unique identifier for the left neighboring lane, defaults to None.
        :param right_lane_id: The unique identifier for the right neighboring lane, defaults to None.
        :param predecessor_ids: The unique identifiers for the predecessor lanes, defaults to [].
        :param successor_ids: The unique identifiers for the successor lanes, defaults to [].
        :param speed_limit_mps: The speed limit for the lane in meters per second, defaults to None.
        :param outline: The outline of the lane, defaults to None.
        :param shapely_polygon: The Shapely polygon representation of the lane, defaults to None.
        :param map_api: The MapAPI instance for accessing map objects, defaults to None.
        """
        if outline is None:
            outline_array = np.vstack(
                (
                    left_boundary.array,
                    right_boundary.array[::-1],
                    left_boundary.array[0],
                )
            )
            outline = Polyline3D.from_array(outline_array)
        super().__init__(object_id, outline, shapely_polygon)

        self._lane_group_id = lane_group_id
        self._left_boundary = left_boundary
        self._right_boundary = right_boundary
        self._centerline = centerline
        self._left_lane_id = left_lane_id
        self._right_lane_id = right_lane_id
        self._predecessor_ids = predecessor_ids
        self._successor_ids = successor_ids
        self._speed_limit_mps = speed_limit_mps
        self._map_api = map_api

    @property
    def layer(self) -> MapLayer:
        """The :class:`~py123d.datatypes.map_objects.map_layer_types.MapLayer` of the map object."""
        return MapLayer.LANE

    @property
    def lane_group_id(self) -> MapObjectIDType:
        """ID of the lane group this lane belongs to."""
        return self._lane_group_id

    @property
    def lane_group(self) -> Optional[LaneGroup]:
        """The :class:`LaneGroup` this lane belongs to."""
        lane_group: Optional[LaneGroup] = None
        if self._map_api is not None:
            lane_group_ = self._map_api.get_map_object(self.lane_group_id, MapLayer.LANE_GROUP)
            if isinstance(lane_group_, LaneGroup):
                lane_group = lane_group_
        return lane_group

    @property
    def left_boundary(self) -> Polyline3D:
        """The left boundary of the lane as a :class:`~py123d.geometry.Polyline3D`."""
        return self._left_boundary

    @property
    def right_boundary(self) -> Polyline3D:
        """The right boundary of the lane as a :class:`~py123d.geometry.Polyline3D`."""
        return self._right_boundary

    @property
    def centerline(self) -> Polyline3D:
        """The centerline of the lane as a :class:`~py123d.geometry.Polyline3D`."""
        return self._centerline

    @property
    def left_lane_id(self) -> Optional[MapObjectIDType]:
        """ID of the left neighboring lane."""
        return self._left_lane_id

    @property
    def left_lane(self) -> Optional[Lane]:
        """The left neighboring :class:`Lane`, if available."""
        left_lane: Optional[Lane] = None
        if self._map_api is not None and self.left_lane_id is not None:
            left_lane_ = self._map_api.get_map_object(self.left_lane_id, self.layer)
            if isinstance(left_lane_, Lane):
                left_lane = left_lane_
        return left_lane

    @property
    def right_lane_id(self) -> Optional[MapObjectIDType]:
        """ID of the right neighboring lane."""
        return self._right_lane_id

    @property
    def right_lane(self) -> Optional[Lane]:
        """The right neighboring :class:`Lane`, if available."""
        right_lane: Optional[Lane] = None
        if self._map_api is not None and self.right_lane_id is not None:
            right_lane_ = self._map_api.get_map_object(self.right_lane_id, self.layer)
            if isinstance(right_lane_, Lane):
                right_lane = right_lane_
        return right_lane

    @property
    def predecessor_ids(self) -> List[MapObjectIDType]:
        """List of IDs of the predecessor lanes."""
        return self._predecessor_ids

    @property
    def predecessors(self) -> List[Lane]:
        """List of predecessor :class:`Lane` instances."""
        predecessors: List[Lane] = []
        if self._map_api is not None:
            for lane_id in self.predecessor_ids:
                predecessor_ = self._map_api.get_map_object(lane_id, self.layer)
                if predecessor_ is not None and isinstance(predecessor_, Lane):
                    predecessors.append(predecessor_)
        return predecessors

    @property
    def successor_ids(self) -> List[MapObjectIDType]:
        """List of IDs of the successor lanes."""
        return self._successor_ids

    @property
    def successors(self) -> List[Lane]:
        """List of successor :class:`Lane` instances."""
        successors: List[Lane] = []
        if self._map_api is not None:
            for lane_id in self.successor_ids:
                successor_ = self._map_api.get_map_object(lane_id, self.layer)
                if successor_ is not None and isinstance(successor_, Lane):
                    successors.append(successor_)
        return successors

    @property
    def speed_limit_mps(self) -> Optional[float]:
        """The speed limit of the lane in meters per second."""
        return self._speed_limit_mps

    @property
    def trimesh_mesh(self) -> Trimesh:
        """The trimesh mesh representation of the lane."""
        return get_trimesh_from_boundaries(self.left_boundary, self.right_boundary)


class LaneGroup(BaseMapSurfaceObject):
    """Class representing a group of lanes going in the same direction."""

    __slots__ = (
        "_lane_ids",
        "_left_boundary",
        "_right_boundary",
        "_intersection_id",
        "_predecessor_ids",
        "_successor_ids",
        "_map_api",
    )

    def __init__(
        self,
        object_id: MapObjectIDType,
        lane_ids: List[MapObjectIDType],
        left_boundary: Polyline3D,
        right_boundary: Polyline3D,
        intersection_id: Optional[MapObjectIDType] = None,
        predecessor_ids: List[MapObjectIDType] = [],
        successor_ids: List[MapObjectIDType] = [],
        outline: Optional[Polyline3D] = None,
        shapely_polygon: Optional[geom.Polygon] = None,
        map_api: Optional["MapAPI"] = None,
    ):
        """Initialize a :class:`LaneGroup` instance.

        Notes
        -----
        If the map_api is provided, neighboring lane groups and intersection can be accessed through the properties.
        If the outline is not provided, it will be constructed from the left and right boundaries.
        If the shapely_polygon is not provided, it will be constructed from the outline.

        :param object_id: The ID of the lane group.
        :param lane_ids: The IDs of the lanes in the group.
        :param left_boundary: The left boundary of the lane group.
        :param right_boundary: The right boundary of the lane group.
        :param intersection_id: The ID of the intersection, defaults to None
        :param predecessor_ids: The IDs of the predecessor lanes, defaults to []
        :param successor_ids: The IDs of the successor lanes, defaults to []
        :param outline: The outline of the lane group, defaults to None
        :param shapely_polygon: The shapely polygon representation of the lane group, defaults to None
        :param map_api: The map API instance, defaults to None
        """
        if outline is None:
            outline_array = np.vstack(
                (
                    left_boundary.array,
                    right_boundary.array[::-1],
                    left_boundary.array[0],
                )
            )
            outline = Polyline3D.from_array(outline_array)
        super().__init__(object_id, outline, shapely_polygon)

        self._lane_ids = lane_ids
        self._left_boundary = left_boundary
        self._right_boundary = right_boundary
        self._intersection_id = intersection_id
        self._predecessor_ids = predecessor_ids
        self._successor_ids = successor_ids
        self._map_api = map_api

    @property
    def layer(self) -> MapLayer:
        """The :class:`~py123d.datatypes.map_objects.map_layer_types.MapLayer` of the map object."""
        return MapLayer.LANE_GROUP

    @property
    def lane_ids(self) -> List[MapObjectIDType]:
        """List of IDs of the lanes in the group."""
        return self._lane_ids

    @property
    def lanes(self) -> List[Lane]:
        """List of :class:`Lane` instances in the group."""
        lanes: List[Lane] = []
        if self._map_api is not None:
            for lane_id in self.lane_ids:
                lane = self._map_api.get_map_object(lane_id, MapLayer.LANE)
                if lane is not None and isinstance(lane, Lane):
                    lanes.append(lane)
        return lanes

    @property
    def left_boundary(self) -> Polyline3D:
        """The left boundary of the lane group."""
        return self._left_boundary

    @property
    def right_boundary(self) -> Polyline3D:
        """The right boundary of the lane group."""
        return self._right_boundary

    @property
    def intersection_id(self) -> Optional[MapObjectIDType]:
        """ID of the intersection the lane group belongs to, if available."""
        return self._intersection_id

    @property
    def intersection(self) -> Optional[Intersection]:
        """The :class:`Intersection` the lane group belongs to, if available."""
        intersection: Optional[Intersection] = None
        if self._map_api is not None and self.intersection_id is not None:
            intersection = self._map_api.get_map_object(self.intersection_id, MapLayer.INTERSECTION)  # type: ignore
        return intersection

    @property
    def predecessor_ids(self) -> List[MapObjectIDType]:
        """List of IDs of the predecessor lane groups."""
        return self._predecessor_ids

    @property
    def predecessors(self) -> List[LaneGroup]:
        """List of predecessor :class:`LaneGroup` instances."""
        predecessors: List[LaneGroup] = []
        if self._map_api is not None:
            for lane_group_id in self.predecessor_ids:
                predecessor = self._map_api.get_map_object(lane_group_id, self.layer)
                if predecessor is not None and isinstance(predecessor, LaneGroup):
                    predecessors.append(predecessor)
        return predecessors

    @property
    def successor_ids(self) -> List[MapObjectIDType]:
        """List of IDs of the successor lane groups."""
        return self._successor_ids

    @property
    def successors(self) -> List[LaneGroup]:
        """List of successor :class:`LaneGroup` instances."""
        successors: List[LaneGroup] = []
        if self._map_api is not None:
            for lane_group_id in self.successor_ids:
                successor = self._map_api.get_map_object(lane_group_id, self.layer)
                if successor is not None and isinstance(successor, LaneGroup):
                    successors.append(successor)
        return successors

    @property
    def trimesh_mesh(self) -> Trimesh:
        """The trimesh mesh representation of the lane group."""
        return get_trimesh_from_boundaries(self.left_boundary, self.right_boundary)


class Intersection(BaseMapSurfaceObject):
    """Class representing an intersection in a map, which consists of multiple lane groups."""

    __slots__ = ("_lane_group_ids", "_map_api")

    def __init__(
        self,
        object_id: MapObjectIDType,
        lane_group_ids: List[MapObjectIDType],
        outline: Optional[Union[Polyline2D, Polyline3D]] = None,
        shapely_polygon: Optional[geom.Polygon] = None,
        map_api: Optional["MapAPI"] = None,
    ):
        """Initialize an :class:`Intersection` instance.

        Notes
        -----
        If the map_api is provided, lane groups can be accessed through the properties.
        Either outline or shapely_polygon must be provided.

        :param object_id: The ID of the intersection.
        :param lane_group_ids: The IDs of the lane groups that belong to the intersection.
        :param outline: The outline of the intersection, defaults to None.
        :param shapely_polygon: The Shapely polygon representation of the intersection, defaults to None.
        :param map_api: The MapAPI instance, defaults to None.
        """
        super().__init__(object_id, outline, shapely_polygon)
        self._lane_group_ids = lane_group_ids
        self._map_api = map_api

    @property
    def layer(self) -> MapLayer:
        """The :class:`~py123d.datatypes.map_objects.map_layer_types.MapLayer` of the map object."""
        return MapLayer.INTERSECTION

    @property
    def lane_group_ids(self) -> List[MapObjectIDType]:
        """List of IDs of the lane groups that belong to the intersection."""
        return self._lane_group_ids

    @property
    def lane_groups(self) -> List[LaneGroup]:
        """List of :class:`LaneGroup` instances that belong to the intersection."""
        lane_groups: List[LaneGroup] = []
        if self._map_api is not None:
            for lane_group_id in self.lane_group_ids:
                lane_group = self._map_api.get_map_object(lane_group_id, MapLayer.LANE_GROUP)
                if lane_group is not None and isinstance(lane_group, LaneGroup):
                    lane_groups.append(lane_group)
        return lane_groups


class Crosswalk(BaseMapSurfaceObject):
    """Class representing a crosswalk in a map."""

    __slots__ = ()

    def __init__(
        self,
        object_id: MapObjectIDType,
        outline: Optional[Union[Polyline2D, Polyline3D]] = None,
        shapely_polygon: Optional[geom.Polygon] = None,
    ):
        """Initialize a Crosswalk instance.

        Notes
        -----
        Either outline or shapely_polygon must be provided.

        :param object_id: The ID of the crosswalk.
        :param outline: The outline of the crosswalk, defaults to None.
        :param shapely_polygon: The Shapely polygon representation of the crosswalk, defaults to None.
        """
        super().__init__(object_id, outline, shapely_polygon)

    @property
    def layer(self) -> MapLayer:
        """The :class:`~py123d.datatypes.map_objects.map_layer_types.MapLayer` of the map object."""
        return MapLayer.CROSSWALK


class Carpark(BaseMapSurfaceObject):
    """Class representing a carpark or driveway in a map."""

    __slots__ = ()

    def __init__(
        self,
        object_id: MapObjectIDType,
        outline: Optional[Union[Polyline2D, Polyline3D]] = None,
        shapely_polygon: Optional[geom.Polygon] = None,
    ):
        """Initialize a Carpark instance.

        Notes
        -----
        Either outline or shapely_polygon must be provided.

        :param object_id: The ID of the carpark.
        :param outline: The outline of the carpark, defaults to None.
        :param shapely_polygon: The Shapely polygon representation of the carpark, defaults to None.
        """
        super().__init__(object_id, outline, shapely_polygon)

    @property
    def layer(self) -> MapLayer:
        """The :class:`~py123d.datatypes.map_objects.map_layer_types.MapLayer` of the map object."""
        return MapLayer.CARPARK


class Walkway(BaseMapSurfaceObject):
    """Class representing a walkway in a map."""

    __slots__ = ()

    def __init__(
        self,
        object_id: MapObjectIDType,
        outline: Optional[Union[Polyline2D, Polyline3D]] = None,
        shapely_polygon: Optional[geom.Polygon] = None,
    ):
        """Initialize a Walkway instance.

        Notes
        -----
        Either outline or shapely_polygon must be provided.

        :param object_id: The ID of the walkway.
        :param outline: The outline of the walkway, defaults to None.
        :param shapely_polygon: The Shapely polygon representation of the walkway, defaults to None.
        """
        super().__init__(object_id, outline, shapely_polygon)

    @property
    def layer(self) -> MapLayer:
        return MapLayer.WALKWAY


class GenericDrivable(BaseMapSurfaceObject):
    """Class representing a generic drivable area in a map.
    Can overlap with other drivable areas, depending on the dataset.
    """

    __slots__ = ()

    def __init__(
        self,
        object_id: MapObjectIDType,
        outline: Optional[Union[Polyline2D, Polyline3D]] = None,
        shapely_polygon: Optional[geom.Polygon] = None,
    ):
        """Initialize a GenericDrivable instance.

        Notes
        -----
        Either outline or shapely_polygon must be provided.

        :param object_id: The ID of the walkway.
        :param outline: The outline of the walkway, defaults to None.
        :param shapely_polygon: The Shapely polygon representation of the walkway, defaults to None.
        """
        super().__init__(object_id, outline, shapely_polygon)

    @property
    def layer(self) -> MapLayer:
        """The :class:`~py123d.datatypes.map_objects.map_layer_types.MapLayer` of the map object."""
        return MapLayer.GENERIC_DRIVABLE


class StopZone(BaseMapSurfaceObject):
    """Class representing a stop zone in a map."""

    __slots__ = ("_stop_zone_type", "_lane_ids")

    def __init__(
        self,
        object_id: MapObjectIDType,
        stop_zone_type: StopZoneType,
        outline: Optional[Union[Polyline2D, Polyline3D]] = None,
        shapely_polygon: Optional[geom.Polygon] = None,
        lane_ids: Optional[Sequence[MapObjectIDType]] = None,
    ):
        """Initialize a StopZone instance.

        Notes
        -----
        Either outline or shapely_polygon must be provided.

        TODOs
        -----
        - Add optionally ids/references to other map objects (e.g. pedestrian crossings) that are associated with the stop zone.
        - Add weak reference to the map API to access other map objects.


        :param object_id: The ID of the stop zone.
        :param stop_zone_type: The type of the stop zone (traffic light, stop sign, etc.).
        :param outline: The outline of the stop zone, defaults to None.
        :param shapely_polygon: The Shapely polygon representation of the stop zone, defaults to None.
        :param lane_ids: List of lane IDs this stop zone controls, defaults to None.
        """
        super().__init__(object_id, outline, shapely_polygon)
        self._stop_zone_type = stop_zone_type
        self._lane_ids = list(lane_ids) if lane_ids is not None else []

    @property
    def stop_zone_type(self) -> StopZoneType:
        """The type of the stop zone."""
        return self._stop_zone_type

    @property
    def lane_ids(self) -> List[MapObjectIDType]:
        """List of lane IDs this stop zone controls."""
        return self._lane_ids

    @property
    def layer(self) -> MapLayer:
        """The :class:`~py123d.datatypes.map_objects.map_layer_types.MapLayer` of the map object."""
        return MapLayer.STOP_ZONE


class RoadEdge(BaseMapLineObject):
    """Class representing a road edge in a map."""

    __slots__ = ("_road_edge_type",)

    def __init__(
        self,
        object_id: MapObjectIDType,
        road_edge_type: RoadEdgeType,
        polyline: Union[Polyline2D, Polyline3D],
    ):
        """Initialize a RoadEdge instance.

        :param object_id: The ID of the road edge.
        :param road_edge_type: The type of the road edge.
        :param polyline: The polyline representation of the road edge.
        """
        super().__init__(object_id, polyline)
        self._road_edge_type = road_edge_type

    @property
    def layer(self) -> MapLayer:
        return MapLayer.ROAD_EDGE

    @property
    def road_edge_type(self) -> RoadEdgeType:
        """The type of road edge, according to :class:`~py123d.datatypes.map_objects.map_layer_types.RoadEdgeType`."""
        return self._road_edge_type


class RoadLine(BaseMapLineObject):
    """Class representing a road line in a map."""

    __slots__ = ("_road_line_type",)

    def __init__(
        self,
        object_id: MapObjectIDType,
        road_line_type: RoadLineType,
        polyline: Union[Polyline2D, Polyline3D],
    ):
        """Initialize a RoadLine instance.

        :param object_id: The ID of the road line.
        :param road_line_type: The type of the road line.
        :param polyline: The polyline representation of the road line.
        """
        super().__init__(object_id, polyline)
        self._road_line_type = road_line_type

    @property
    def layer(self) -> MapLayer:
        """The :class:`~py123d.datatypes.map_objects.map_layer_types.MapLayer` of the map object."""
        return MapLayer.ROAD_LINE

    @property
    def road_line_type(self) -> RoadLineType:
        """The type of road edge, according to :class:`~py123d.datatypes.map_objects.map_layer_types.RoadLineType`."""
        return self._road_line_type
