from typing import List, Tuple

import numpy as np
import pytest
import shapely
import trimesh

from py123d.datatypes.map_objects import Intersection, Lane, LaneGroup, MapLayer
from py123d.datatypes.map_objects.map_layer_types import (
    IntersectionType,
    LaneType,
    RoadEdgeType,
    RoadLineType,
    StopZoneType,
)
from py123d.datatypes.map_objects.map_objects import (
    Carpark,
    Crosswalk,
    GenericDrivable,
    RoadEdge,
    RoadLine,
    StopZone,
    Walkway,
)
from py123d.geometry.polyline import Polyline2D, Polyline3D

from .mock_map_api import MockMapAPI


def _get_linked_map_object_setup() -> Tuple[List[Lane], List[LaneGroup], List[Intersection]]:
    """Helper function to create linked map objects for testing."""

    Z = 0.0

    # Lanes:
    lanes: List[Lane] = []

    # Middle Lane 0, group 0
    middle_left_boundary = np.array([[0.0, 1.0, Z], [50.0, 1.0, Z]])
    middle_right_boundary = np.array([[0.0, -1.0, Z], [50.0, -1.0, Z]])
    middle_centerline = np.mean(np.array([middle_right_boundary, middle_left_boundary]), axis=0)
    lanes.append(
        Lane(
            object_id=0,
            lane_type=LaneType.SURFACE_STREET,
            lane_group_id=0,
            left_boundary=Polyline3D.from_array(middle_left_boundary),
            right_boundary=Polyline3D.from_array(middle_right_boundary),
            centerline=Polyline3D.from_array(middle_centerline),
            left_lane_id=1,
            right_lane_id=2,
            predecessor_ids=[3],
            successor_ids=[4],
            speed_limit_mps=0.0,
        )
    )

    # Left Lane 1, group 0
    left_left_boundary = np.array([[0.0, 2.0, Z], [50.0, 2.0, Z]])
    left_right_boundary = middle_left_boundary.copy()
    left_centerline = np.mean(np.array([left_right_boundary, left_left_boundary]), axis=0)
    lanes.append(
        Lane(
            object_id=1,
            lane_type=LaneType.SURFACE_STREET,
            lane_group_id=0,
            left_boundary=Polyline3D.from_array(left_left_boundary),
            right_boundary=Polyline3D.from_array(left_right_boundary),
            centerline=Polyline3D.from_array(left_centerline),
            left_lane_id=None,
            right_lane_id=0,
            predecessor_ids=[],
            successor_ids=[],
            speed_limit_mps=0.0,
        )
    )

    # Right Lane 2, group 0
    right_right_boundary = np.array([[0.0, -2.0, Z], [50.0, -2.0, Z]])
    right_left_boundary = middle_right_boundary.copy()
    right_centerline = np.mean(np.array([right_right_boundary, right_left_boundary]), axis=0)
    lanes.append(
        Lane(
            object_id=2,
            lane_type=LaneType.SURFACE_STREET,
            lane_group_id=0,
            left_boundary=Polyline3D.from_array(right_left_boundary),
            right_boundary=Polyline3D.from_array(right_right_boundary),
            centerline=Polyline3D.from_array(right_centerline),
            left_lane_id=0,
            right_lane_id=None,
            predecessor_ids=[],
            successor_ids=[],
            speed_limit_mps=0.0,
        )
    )

    # Predecessor lane 3, group 1
    predecessor_left_boundary = np.array([[-50.0, 1.0, Z], [0.0, 1.0, Z]])
    predecessor_right_boundary = np.array([[-50.0, -1.0, Z], [0.0, -1.0, Z]])
    predecessor_centerline = np.mean(np.array([predecessor_right_boundary, predecessor_left_boundary]), axis=0)
    lanes.append(
        Lane(
            object_id=3,
            lane_type=LaneType.SURFACE_STREET,
            lane_group_id=1,
            left_boundary=Polyline3D.from_array(predecessor_left_boundary),
            right_boundary=Polyline3D.from_array(predecessor_right_boundary),
            centerline=Polyline3D.from_array(predecessor_centerline),
            left_lane_id=None,
            right_lane_id=None,
            predecessor_ids=[],
            successor_ids=[0],
            speed_limit_mps=0.0,
        )
    )

    # Successor lane 4, group 2
    successor_left_boundary = np.array([[50.0, 1.0, Z], [100.0, 1.0, Z]])
    successor_right_boundary = np.array([[50.0, -1.0, Z], [100.0, -1.0, Z]])
    successor_centerline = np.mean(np.array([successor_right_boundary, successor_left_boundary]), axis=0)
    lanes.append(
        Lane(
            object_id=4,
            lane_type=LaneType.SURFACE_STREET,
            lane_group_id=2,
            left_boundary=Polyline3D.from_array(successor_left_boundary),
            right_boundary=Polyline3D.from_array(successor_right_boundary),
            centerline=Polyline3D.from_array(successor_centerline),
            left_lane_id=None,
            right_lane_id=None,
            predecessor_ids=[0],
            successor_ids=[],
            speed_limit_mps=0.0,
        )
    )

    # Lane Groups:
    lane_groups = []

    # Middle lane group 0, lanes 0,1,2
    middle_lane_group = LaneGroup(
        object_id=0,
        lane_ids=[0, 1, 2],
        left_boundary=Polyline3D.from_array(left_left_boundary),
        right_boundary=Polyline3D.from_array(left_right_boundary),
        intersection_id=None,
        predecessor_ids=[1],
        successor_ids=[2],
    )
    lane_groups.append(middle_lane_group)

    # Predecessor lane group 1, lane 3, intersection 0
    predecessor_lane_group = LaneGroup(
        object_id=1,
        lane_ids=[3],
        left_boundary=Polyline3D.from_array(predecessor_left_boundary),
        right_boundary=Polyline3D.from_array(predecessor_right_boundary),
        intersection_id=0,
        predecessor_ids=[],
        successor_ids=[0],
    )
    lane_groups.append(predecessor_lane_group)

    # Successor lane group 2, lane 4, intersection 1
    successor_lane_group = LaneGroup(
        object_id=2,
        lane_ids=[4],
        left_boundary=Polyline3D.from_array(successor_left_boundary),
        right_boundary=Polyline3D.from_array(successor_right_boundary),
        intersection_id=1,
        predecessor_ids=[0],
        successor_ids=[],
    )
    lane_groups.append(successor_lane_group)

    # Intersections:
    intersections = []

    # Intersection 0, includes lane groups 1
    intersection_predecessor = Intersection(
        object_id=0,
        intersection_type=IntersectionType.TRAFFIC_LIGHT,
        lane_group_ids=[1],
        outline=predecessor_lane_group.outline,
    )
    intersections.append(intersection_predecessor)

    intersection_successor = Intersection(
        object_id=1,
        intersection_type=IntersectionType.STOP_SIGN,
        lane_group_ids=[2],
        outline=successor_lane_group.outline,
    )
    intersections.append(intersection_successor)

    return lanes, lane_groups, intersections


class TestLane:
    def setup_method(self) -> None:
        lanes, lane_groups, intersections = _get_linked_map_object_setup()
        self.lanes = lanes
        self.lane_groups = lane_groups
        self.intersections = intersections

    def test_set_up(self):
        """Test that the setup function creates the correct number of map objects."""
        assert len(self.lanes) == 5
        assert len(self.lane_groups) == 3
        assert len(self.intersections) == 2

    def test_properties(self):
        """Test that the properties of the Lane objects are correct."""
        lane0 = self.lanes[0]
        assert lane0.layer == MapLayer.LANE
        assert lane0.lane_group_id == 0
        assert isinstance(lane0.left_boundary, Polyline3D)
        assert isinstance(lane0.right_boundary, Polyline3D)
        assert isinstance(lane0.centerline, Polyline3D)

        assert lane0.left_lane_id == 1
        assert lane0.right_lane_id == 2
        assert lane0.predecessor_ids == [3]
        assert lane0.successor_ids == [4]
        assert lane0.speed_limit_mps == 0.0
        assert isinstance(lane0.trimesh_mesh, trimesh.base.Trimesh)

    def test_base_properties(self):
        """Test that the base_surface property of the Lane objects is correct."""
        lane0 = self.lanes[0]
        assert lane0.object_id == 0
        assert isinstance(lane0.outline, Polyline3D)
        assert isinstance(lane0.outline_2d, Polyline2D)
        assert isinstance(lane0.outline_3d, Polyline3D)
        assert isinstance(lane0.shapely_polygon, shapely.Polygon)

    def test_left_links(self):
        """Test that the left neighboring lanes are correctly linked."""
        map_api = MockMapAPI(
            lanes=self.lanes,
            lane_groups=self.lane_groups,
            intersections=self.intersections,
            add_map_api_links=True,
        )

        def _no_left_neighbor(lane: Lane):
            assert lane is not None
            assert lane.left_lane is None
            assert lane.left_lane_id is None

        # Middle Lane 0
        lane0: Lane = map_api.get_map_object_in_layer(0, MapLayer.LANE)
        assert lane0 is not None
        assert lane0.left_lane is not None
        assert isinstance(lane0.left_lane, Lane)
        assert lane0.left_lane.object_id == 1
        assert lane0.left_lane.object_id == lane0.left_lane_id

        # Left Lane 1
        lane1: Lane = map_api.get_map_object_in_layer(1, MapLayer.LANE)
        _no_left_neighbor(lane1)

        # Right Lane 2
        lane2: Lane = map_api.get_map_object_in_layer(2, MapLayer.LANE)
        assert lane2 is not None
        assert lane2.left_lane is not None
        assert isinstance(lane2.left_lane, Lane)
        assert lane2.left_lane.object_id == 0
        assert lane2.left_lane.object_id == lane2.left_lane_id

        # Predecessor Lane 3
        lane3: Lane = map_api.get_map_object_in_layer(3, MapLayer.LANE)
        _no_left_neighbor(lane3)

        # Successor Lane 4
        lane4: Lane = map_api.get_map_object_in_layer(4, MapLayer.LANE)
        _no_left_neighbor(lane4)

    def test_right_links(self):
        """Test that the right neighboring lanes are correctly linked."""
        map_api = MockMapAPI(
            lanes=self.lanes,
            lane_groups=self.lane_groups,
            intersections=self.intersections,
            add_map_api_links=True,
        )

        def _no_right_neighbor(lane: Lane):
            assert lane is not None
            assert lane.right_lane is None
            assert lane.right_lane_id is None

        # Middle Lane 0
        lane0: Lane = map_api.get_map_object_in_layer(0, MapLayer.LANE)
        assert lane0 is not None
        assert lane0.right_lane is not None
        assert isinstance(lane0.right_lane, Lane)
        assert lane0.right_lane.object_id == 2
        assert lane0.right_lane.object_id == lane0.right_lane_id

        # Left Lane 1
        lane1: Lane = map_api.get_map_object_in_layer(1, MapLayer.LANE)
        assert lane1 is not None
        assert lane1.right_lane is not None
        assert isinstance(lane1.right_lane, Lane)
        assert lane1.right_lane.object_id == 0
        assert lane1.right_lane.object_id == lane1.right_lane_id

        # Right Lane 2
        lane2: Lane = map_api.get_map_object_in_layer(2, MapLayer.LANE)
        _no_right_neighbor(lane2)

        # Predecessor Lane 3
        lane3: Lane = map_api.get_map_object_in_layer(3, MapLayer.LANE)
        _no_right_neighbor(lane3)

        # Successor Lane 4
        lane4: Lane = map_api.get_map_object_in_layer(4, MapLayer.LANE)
        _no_right_neighbor(lane4)

    def test_predecessor_links(self):
        """Test that the predecessor lanes are correctly linked."""
        map_api = MockMapAPI(
            lanes=self.lanes,
            lane_groups=self.lane_groups,
            intersections=self.intersections,
            add_map_api_links=True,
        )

        def _no_predecessors(lane: Lane):
            assert lane is not None
            assert lane.predecessors == []
            assert lane.predecessor_ids == []

        # Middle Lane 0
        lane0: Lane = map_api.get_map_object_in_layer(0, MapLayer.LANE)
        assert lane0 is not None
        assert lane0.predecessors is not None
        assert len(lane0.predecessors) == 1
        assert isinstance(lane0.predecessors[0], Lane)
        assert lane0.predecessors[0].object_id == 3
        assert lane0.predecessor_ids == [3]

        # Left Lane 1
        lane1: Lane = map_api.get_map_object_in_layer(1, MapLayer.LANE)
        _no_predecessors(lane1)

        # Right Lane 2
        lane2: Lane = map_api.get_map_object_in_layer(2, MapLayer.LANE)
        _no_predecessors(lane2)

        # Predecessor Lane 3
        lane3: Lane = map_api.get_map_object_in_layer(3, MapLayer.LANE)
        _no_predecessors(lane3)

        # Successor Lane 4
        lane4: Lane = map_api.get_map_object_in_layer(4, MapLayer.LANE)
        assert lane4 is not None
        assert lane4.predecessors is not None
        assert len(lane4.predecessors) == 1
        assert isinstance(lane4.predecessors[0], Lane)
        assert lane4.predecessors[0].object_id == 0
        assert lane4.predecessor_ids == [0]

    def test_successor_links(self):
        """Test that the successor lanes are correctly linked."""
        map_api = MockMapAPI(
            lanes=self.lanes,
            lane_groups=self.lane_groups,
            intersections=self.intersections,
            add_map_api_links=True,
        )

        def _no_successors(lane: Lane):
            assert lane is not None
            assert lane.successors == []
            assert lane.successor_ids == []

        # Middle Lane 0
        lane0: Lane = map_api.get_map_object_in_layer(0, MapLayer.LANE)
        assert lane0 is not None
        assert lane0.successors is not None
        assert len(lane0.successors) == 1
        assert isinstance(lane0.successors[0], Lane)
        assert lane0.successors[0].object_id == 4
        assert lane0.successor_ids == [4]

        # Left Lane 1
        lane1: Lane = map_api.get_map_object_in_layer(1, MapLayer.LANE)
        _no_successors(lane1)

        # Right Lane 2
        lane2: Lane = map_api.get_map_object_in_layer(2, MapLayer.LANE)
        _no_successors(lane2)

        # Predecessor Lane 3
        lane3: Lane = map_api.get_map_object_in_layer(3, MapLayer.LANE)
        assert lane3 is not None
        assert lane3.successors is not None
        assert len(lane3.successors) == 1
        assert isinstance(lane3.successors[0], Lane)
        assert lane3.successors[0].object_id == 0
        assert lane3.successor_ids == [0]

        # Successor Lane 4
        lane4: Lane = map_api.get_map_object_in_layer(4, MapLayer.LANE)
        _no_successors(lane4)

    def test_no_links(self):
        map_api = MockMapAPI(
            lanes=self.lanes,
            lane_groups=self.lane_groups,
            intersections=self.intersections,
            add_map_api_links=False,
        )
        for lane in self.lanes:
            lane_from_api: Lane = map_api.get_map_object_in_layer(lane.object_id, MapLayer.LANE)
            assert lane_from_api is not None
            assert lane_from_api.left_lane is None
            assert lane_from_api.right_lane is None
            assert lane_from_api.predecessors == []
            assert lane_from_api.successors == []

    def test_lane_group_links(self):
        """Test that the lane group links are correct."""
        map_api = MockMapAPI(
            lanes=self.lanes,
            lane_groups=self.lane_groups,
            intersections=self.intersections,
            add_map_api_links=True,
        )

        for lane in self.lanes:
            lane_from_api: Lane = map_api.get_map_object_in_layer(lane.object_id, MapLayer.LANE)
            assert lane_from_api is not None
            assert lane_from_api.lane_group is not None
            assert isinstance(lane_from_api.lane_group, LaneGroup)
            assert lane_from_api.lane_group.object_id == lane_from_api.lane_group_id


class TestLaneGroup:
    def setup_method(self):
        lanes, lane_groups, intersections = _get_linked_map_object_setup()
        self.lanes = lanes
        self.lane_groups = lane_groups
        self.intersections = intersections

    def test_properties(self):
        """Test that the properties of the LaneGroup objects are correct."""
        lane_group0 = self.lane_groups[0]
        assert lane_group0.layer == MapLayer.LANE_GROUP
        assert lane_group0.lane_ids == [0, 1, 2]
        assert isinstance(lane_group0.left_boundary, Polyline3D)
        assert isinstance(lane_group0.right_boundary, Polyline3D)
        assert lane_group0.intersection_id is None
        assert lane_group0.predecessor_ids == [1]
        assert lane_group0.successor_ids == [2]
        assert isinstance(lane_group0.trimesh_mesh, trimesh.base.Trimesh)

    def test_base_properties(self):
        """Test that the base surface properties of the LaneGroup objects are correct."""
        lane_group0 = self.lane_groups[0]
        assert lane_group0.object_id == 0
        assert isinstance(lane_group0.outline, Polyline3D)
        assert isinstance(lane_group0.outline_2d, Polyline2D)
        assert isinstance(lane_group0.outline_3d, Polyline3D)
        assert isinstance(lane_group0.shapely_polygon, shapely.Polygon)

    def test_lane_links(self):
        """Test that the lanes are correctly linked to the lane group."""
        map_api = MockMapAPI(
            lanes=self.lanes,
            lane_groups=self.lane_groups,
            intersections=self.intersections,
            add_map_api_links=True,
        )

        # Lane group 0 contains lanes 0, 1, 2
        lane_group0: LaneGroup = map_api.get_map_object_in_layer(0, MapLayer.LANE_GROUP)
        assert lane_group0 is not None
        assert lane_group0.lanes is not None
        assert len(lane_group0.lanes) == 3
        for i, lane in enumerate(lane_group0.lanes):
            assert isinstance(lane, Lane)
            assert lane.object_id == i

        # Lane group 1 contains lane 3
        lane_group1: LaneGroup = map_api.get_map_object_in_layer(1, MapLayer.LANE_GROUP)
        assert lane_group1 is not None
        assert lane_group1.lanes is not None
        assert len(lane_group1.lanes) == 1
        assert isinstance(lane_group1.lanes[0], Lane)
        assert lane_group1.lanes[0].object_id == 3

        # Lane group 2 contains lane 4
        lane_group2: LaneGroup = map_api.get_map_object_in_layer(2, MapLayer.LANE_GROUP)
        assert lane_group2 is not None
        assert lane_group2.lanes is not None
        assert len(lane_group2.lanes) == 1
        assert isinstance(lane_group2.lanes[0], Lane)
        assert lane_group2.lanes[0].object_id == 4

    def test_predecessor_links(self):
        """Test that the predecessor lane groups are correctly linked."""
        map_api = MockMapAPI(
            lanes=self.lanes,
            lane_groups=self.lane_groups,
            intersections=self.intersections,
            add_map_api_links=True,
        )

        def _no_predecessors(lane_group: LaneGroup):
            assert lane_group is not None
            assert lane_group.predecessors == []
            assert lane_group.predecessor_ids == []

        # Lane group 0 has predecessor lane group 1
        lane_group0: LaneGroup = map_api.get_map_object_in_layer(0, MapLayer.LANE_GROUP)
        assert lane_group0 is not None
        assert lane_group0.predecessors is not None
        assert len(lane_group0.predecessors) == 1
        assert isinstance(lane_group0.predecessors[0], LaneGroup)
        assert lane_group0.predecessors[0].object_id == 1
        assert lane_group0.predecessor_ids == [1]

        # Lane group 1 has no predecessors
        lane_group1: LaneGroup = map_api.get_map_object_in_layer(1, MapLayer.LANE_GROUP)
        _no_predecessors(lane_group1)

        # Lane group 2 has predecessor lane group 0
        lane_group2: LaneGroup = map_api.get_map_object_in_layer(2, MapLayer.LANE_GROUP)
        assert lane_group2 is not None
        assert lane_group2.predecessors is not None
        assert len(lane_group2.predecessors) == 1
        assert isinstance(lane_group2.predecessors[0], LaneGroup)
        assert lane_group2.predecessors[0].object_id == 0
        assert lane_group2.predecessor_ids == [0]

    def test_successor_links(self):
        """Test that the successor lane groups are correctly linked."""
        map_api = MockMapAPI(
            lanes=self.lanes,
            lane_groups=self.lane_groups,
            intersections=self.intersections,
            add_map_api_links=True,
        )

        def _no_successors(lane_group: LaneGroup):
            assert lane_group is not None
            assert lane_group.successors == []
            assert lane_group.successor_ids == []

        # Lane group 0 has successor lane group 2
        lane_group0: LaneGroup = map_api.get_map_object_in_layer(0, MapLayer.LANE_GROUP)
        assert lane_group0 is not None
        assert lane_group0.successors is not None
        assert len(lane_group0.successors) == 1
        assert isinstance(lane_group0.successors[0], LaneGroup)
        assert lane_group0.successors[0].object_id == 2
        assert lane_group0.successor_ids == [2]

        # Lane group 1 has successor lane group 0
        lane_group1: LaneGroup = map_api.get_map_object_in_layer(1, MapLayer.LANE_GROUP)
        assert lane_group1 is not None
        assert lane_group1.successors is not None
        assert len(lane_group1.successors) == 1
        assert isinstance(lane_group1.successors[0], LaneGroup)
        assert lane_group1.successors[0].object_id == 0
        assert lane_group1.successor_ids == [0]

        # Lane group 2 has no successors
        lane_group2: LaneGroup = map_api.get_map_object_in_layer(2, MapLayer.LANE_GROUP)
        _no_successors(lane_group2)

    def test_intersection_links(self):
        """Test that the intersection links are correct."""
        map_api = MockMapAPI(
            lanes=self.lanes,
            lane_groups=self.lane_groups,
            intersections=self.intersections,
            add_map_api_links=True,
        )

        # Lane group 0 has no intersection
        lane_group0: LaneGroup = map_api.get_map_object_in_layer(0, MapLayer.LANE_GROUP)
        assert lane_group0 is not None
        assert lane_group0.intersection_id is None
        assert lane_group0.intersection is None

        # Lane group 1 has intersection 0
        lane_group1: LaneGroup = map_api.get_map_object_in_layer(1, MapLayer.LANE_GROUP)
        assert lane_group1 is not None
        assert lane_group1.intersection_id == 0
        assert lane_group1.intersection is not None
        assert isinstance(lane_group1.intersection, Intersection)
        assert lane_group1.intersection.object_id == 0

        # Lane group 2 has intersection 1
        lane_group2: LaneGroup = map_api.get_map_object_in_layer(2, MapLayer.LANE_GROUP)
        assert lane_group2 is not None
        assert lane_group2.intersection_id == 1
        assert lane_group2.intersection is not None
        assert isinstance(lane_group2.intersection, Intersection)
        assert lane_group2.intersection.object_id == 1

    def test_no_links(self):
        """Test that when map_api is not provided, no links are available."""
        map_api = MockMapAPI(
            lanes=self.lanes,
            lane_groups=self.lane_groups,
            intersections=self.intersections,
            add_map_api_links=False,
        )
        for lane_group in self.lane_groups:
            lg_from_api: LaneGroup = map_api.get_map_object_in_layer(lane_group.object_id, MapLayer.LANE_GROUP)
            assert lg_from_api is not None
            assert lg_from_api.lanes == []
            assert lg_from_api.predecessors == []
            assert lg_from_api.successors == []
            assert lg_from_api.intersection is None


class TestIntersection:
    def setup_method(self):
        lanes, lane_groups, intersections = _get_linked_map_object_setup()
        self.lanes = lanes
        self.lane_groups = lane_groups
        self.intersections = intersections

    def test_properties(self):
        """Test that the properties of the Intersection objects are correct."""
        intersection0 = self.intersections[0]
        assert intersection0.layer == MapLayer.INTERSECTION
        assert intersection0.lane_group_ids == [1]
        assert isinstance(intersection0.outline, Polyline3D)

    def test_base_properties(self):
        """Test that the base surface properties of the Intersection objects are correct."""
        intersection0 = self.intersections[0]
        assert intersection0.object_id == 0
        assert isinstance(intersection0.outline, Polyline3D)
        assert isinstance(intersection0.outline_2d, Polyline2D)
        assert isinstance(intersection0.outline_3d, Polyline3D)
        assert isinstance(intersection0.shapely_polygon, shapely.Polygon)

    def test_lane_group_links(self):
        """Test that the lane groups are correctly linked to the intersection."""
        map_api = MockMapAPI(
            lanes=self.lanes,
            lane_groups=self.lane_groups,
            intersections=self.intersections,
            add_map_api_links=True,
        )

        # Intersection 0 contains lane group 1
        intersection0: Intersection = map_api.get_map_object_in_layer(0, MapLayer.INTERSECTION)
        assert intersection0 is not None
        assert intersection0.lane_groups is not None
        assert len(intersection0.lane_groups) == 1
        assert isinstance(intersection0.lane_groups[0], LaneGroup)
        assert intersection0.lane_groups[0].object_id == 1

        # Intersection 1 contains lane group 2
        intersection1: Intersection = map_api.get_map_object_in_layer(1, MapLayer.INTERSECTION)
        assert intersection1 is not None
        assert intersection1.lane_groups is not None
        assert len(intersection1.lane_groups) == 1
        assert isinstance(intersection1.lane_groups[0], LaneGroup)
        assert intersection1.lane_groups[0].object_id == 2

    def test_no_links(self):
        """Test that when map_api is not provided, no links are available."""
        map_api = MockMapAPI(
            lanes=self.lanes,
            lane_groups=self.lane_groups,
            intersections=self.intersections,
            add_map_api_links=False,
        )
        for intersection in self.intersections:
            int_from_api: Intersection = map_api.get_map_object_in_layer(intersection.object_id, MapLayer.INTERSECTION)
            assert int_from_api is not None
            assert int_from_api.lane_groups == []


class TestCrosswalk:
    def test_properties(self):
        """Test that the properties of the Crosswalk object are correct."""
        outline = Polyline3D.from_array(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        )
        crosswalk = Crosswalk(object_id=0, outline=outline)
        assert crosswalk.layer == MapLayer.CROSSWALK
        assert crosswalk.object_id == 0
        assert isinstance(crosswalk.outline, Polyline3D)
        assert isinstance(crosswalk.shapely_polygon, shapely.Polygon)

    def test_init_with_shapely_polygon(self):
        """Test initialization with shapely polygon."""
        shapely_polygon = shapely.Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
        crosswalk = Crosswalk(object_id=0, shapely_polygon=shapely_polygon)
        assert crosswalk.object_id == 0
        assert isinstance(crosswalk.shapely_polygon, shapely.Polygon)
        assert isinstance(crosswalk.outline_2d, Polyline2D)

    def test_init_with_polyline2d(self):
        """Test initialization with Polyline2D outline."""
        outline = Polyline2D.from_array(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]))
        crosswalk = Crosswalk(object_id=0, outline=outline)
        assert isinstance(crosswalk.outline_2d, Polyline2D)
        assert isinstance(crosswalk.shapely_polygon, shapely.Polygon)

    def test_base_surface_properties(self):
        """Test base surface object properties."""
        outline = Polyline3D.from_array(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        )
        crosswalk = Crosswalk(object_id=0, outline=outline)
        assert isinstance(crosswalk.outline_3d, Polyline3D)
        assert crosswalk.shapely_polygon.is_valid


class TestCarpark:
    def test_properties(self):
        """Test that the properties of the Carpark object are correct."""
        outline = Polyline3D.from_array(
            np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]])
        )
        carpark = Carpark(object_id=1, outline=outline)
        assert carpark.layer == MapLayer.CARPARK
        assert carpark.object_id == 1
        assert isinstance(carpark.outline, Polyline3D)
        assert isinstance(carpark.shapely_polygon, shapely.Polygon)

    def test_init_with_shapely_polygon(self):
        """Test initialization with shapely polygon."""
        shapely_polygon = shapely.Polygon([(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)])
        carpark = Carpark(object_id=1, shapely_polygon=shapely_polygon)
        assert carpark.object_id == 1
        assert isinstance(carpark.shapely_polygon, shapely.Polygon)
        assert isinstance(carpark.outline_2d, Polyline2D)

    def test_init_with_polyline2d(self):
        """Test initialization with Polyline2D outline."""
        outline = Polyline2D.from_array(np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [0.0, 0.0]]))
        carpark = Carpark(object_id=1, outline=outline)
        assert isinstance(carpark.outline_2d, Polyline2D)
        assert isinstance(carpark.shapely_polygon, shapely.Polygon)

    def test_polygon_area(self):
        """Test that the polygon area is calculated correctly."""
        outline = Polyline3D.from_array(
            np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]])
        )
        carpark = Carpark(object_id=1, outline=outline)
        assert carpark.shapely_polygon.area == pytest.approx(4.0)


class TestWalkway:
    def test_properties(self):
        """Test that the properties of the Walkway object are correct."""
        outline = Polyline2D.from_array(np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 1.0], [0.0, 1.0], [0.0, 0.0]]))
        walkway = Walkway(object_id=2, outline=outline)
        assert walkway.layer == MapLayer.WALKWAY
        assert walkway.object_id == 2
        assert isinstance(walkway.outline_2d, Polyline2D)
        assert isinstance(walkway.shapely_polygon, shapely.Polygon)

    def test_init_with_shapely_polygon(self):
        """Test initialization with shapely polygon."""
        shapely_polygon = shapely.Polygon([(0.0, 0.0), (3.0, 0.0), (3.0, 1.0), (0.0, 1.0)])
        walkway = Walkway(object_id=2, shapely_polygon=shapely_polygon)
        assert walkway.object_id == 2
        assert isinstance(walkway.shapely_polygon, shapely.Polygon)

    def test_init_with_polyline3d(self):
        """Test initialization with Polyline3D outline."""
        outline = Polyline3D.from_array(
            np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        )
        walkway = Walkway(object_id=2, outline=outline)
        assert isinstance(walkway.outline_3d, Polyline3D)
        assert isinstance(walkway.shapely_polygon, shapely.Polygon)

    def test_polygon_bounds(self):
        """Test that polygon bounds are correct."""
        outline = Polyline2D.from_array(np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 1.0], [0.0, 1.0], [0.0, 0.0]]))
        walkway = Walkway(object_id=2, outline=outline)
        bounds = walkway.shapely_polygon.bounds
        assert bounds == (0.0, 0.0, 3.0, 1.0)


class TestGenericDrivable:
    def test_properties(self):
        """Test that the properties of the GenericDrivable object are correct."""
        outline = Polyline3D.from_array(
            np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [5.0, 3.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 0.0]])
        )
        generic_drivable = GenericDrivable(object_id=3, outline=outline)
        assert generic_drivable.layer == MapLayer.GENERIC_DRIVABLE
        assert generic_drivable.object_id == 3
        assert isinstance(generic_drivable.outline, Polyline3D)
        assert isinstance(generic_drivable.shapely_polygon, shapely.Polygon)

    def test_init_with_shapely_polygon(self):
        """Test initialization with shapely polygon."""
        shapely_polygon = shapely.Polygon([(0.0, 0.0), (5.0, 0.0), (5.0, 3.0), (0.0, 3.0)])
        generic_drivable = GenericDrivable(object_id=3, shapely_polygon=shapely_polygon)
        assert generic_drivable.object_id == 3
        assert isinstance(generic_drivable.shapely_polygon, shapely.Polygon)

    def test_init_with_polyline2d(self):
        """Test initialization with Polyline2D outline."""
        outline = Polyline2D.from_array(np.array([[0.0, 0.0], [5.0, 0.0], [5.0, 3.0], [0.0, 3.0], [0.0, 0.0]]))
        generic_drivable = GenericDrivable(object_id=3, outline=outline)
        assert isinstance(generic_drivable.outline_2d, Polyline2D)
        assert isinstance(generic_drivable.shapely_polygon, shapely.Polygon)

    def test_polygon_area(self):
        """Test that the polygon area is calculated correctly."""
        outline = Polyline3D.from_array(
            np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [5.0, 3.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 0.0]])
        )
        generic_drivable = GenericDrivable(object_id=3, outline=outline)
        assert generic_drivable.shapely_polygon.area == pytest.approx(15.0)


class TestStopZone:
    def test_properties(self):
        """Test that the properties of the StopZone object are correct."""
        shapely_polygon = shapely.Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 0.5), (0.0, 0.5)])
        stop_zone = StopZone(object_id=4, stop_zone_type=StopZoneType.TRAFFIC_LIGHT, shapely_polygon=shapely_polygon)
        assert stop_zone.layer == MapLayer.STOP_ZONE
        assert stop_zone.object_id == 4
        assert stop_zone.stop_zone_type == StopZoneType.TRAFFIC_LIGHT
        assert isinstance(stop_zone.shapely_polygon, shapely.Polygon)
        assert isinstance(stop_zone.outline_2d, Polyline2D)

    def test_init_with_polyline3d(self):
        """Test initialization with Polyline3D outline."""
        outline = Polyline3D.from_array(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.5, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.0]])
        )
        stop_zone = StopZone(object_id=4, stop_zone_type=StopZoneType.STOP_SIGN, outline=outline)
        assert isinstance(stop_zone.outline, Polyline3D)
        assert isinstance(stop_zone.shapely_polygon, shapely.Polygon)
        assert stop_zone.stop_zone_type == StopZoneType.STOP_SIGN

    def test_init_with_polyline2d(self):
        """Test initialization with Polyline2D outline."""
        outline = Polyline2D.from_array(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.5], [0.0, 0.5], [0.0, 0.0]]))
        stop_zone = StopZone(object_id=4, stop_zone_type=StopZoneType.UNKNOWN, outline=outline)
        assert isinstance(stop_zone.outline_2d, Polyline2D)
        assert isinstance(stop_zone.shapely_polygon, shapely.Polygon)
        assert stop_zone.stop_zone_type == StopZoneType.UNKNOWN

    def test_polygon_area(self):
        """Test that the polygon area is calculated correctly."""
        shapely_polygon = shapely.Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 0.5), (0.0, 0.5)])
        stop_zone = StopZone(object_id=4, stop_zone_type=StopZoneType.TRAFFIC_LIGHT, shapely_polygon=shapely_polygon)
        assert stop_zone.shapely_polygon.area == pytest.approx(0.5)

    def test_lane_ids_property(self):
        """Test that lane_ids property works correctly."""
        outline = Polyline3D.from_array(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.5, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.0]])
        )
        lane_ids = ["1_0_right_-1", "1_0_right_-2"]
        stop_zone = StopZone(
            object_id=4,
            stop_zone_type=StopZoneType.TRAFFIC_LIGHT,
            outline=outline,
            lane_ids=lane_ids,
        )
        assert stop_zone.lane_ids == lane_ids

    def test_lane_ids_default_empty(self):
        """Test that lane_ids defaults to empty list."""
        shapely_polygon = shapely.Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 0.5), (0.0, 0.5)])
        stop_zone = StopZone(object_id=4, stop_zone_type=StopZoneType.TRAFFIC_LIGHT, shapely_polygon=shapely_polygon)
        assert stop_zone.lane_ids == []


class TestRoadEdge:
    def test_properties(self):
        """Test that the properties of the RoadEdge object are correct."""
        polyline = Polyline3D.from_array(np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]))
        road_edge = RoadEdge(object_id=5, road_edge_type=1, polyline=polyline)
        assert road_edge.layer == MapLayer.ROAD_EDGE
        assert road_edge.object_id == 5
        assert road_edge.road_edge_type == 1
        assert isinstance(road_edge.polyline, Polyline3D)

    def test_init_with_polyline2d(self):
        """Test initialization with Polyline2D."""
        polyline = Polyline2D.from_array(np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]]))
        road_edge = RoadEdge(object_id=5, road_edge_type=1, polyline=polyline)
        assert isinstance(road_edge.polyline, Polyline2D)
        assert road_edge.road_edge_type == 1

    def test_polyline_length(self):
        """Test that the polyline has correct number of points."""
        polyline = Polyline3D.from_array(np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]))
        road_edge = RoadEdge(object_id=5, road_edge_type=1, polyline=polyline)
        assert len(road_edge.polyline.array) == 3

    def test_different_road_edge_types(self):
        """Test different road edge types."""
        polyline = Polyline3D.from_array(np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]))
        for edge_type in RoadEdgeType:
            road_edge = RoadEdge(object_id=5, road_edge_type=edge_type, polyline=polyline)
            assert road_edge.road_edge_type == edge_type


class TestRoadLine:
    def test_properties(self):
        """Test that the properties of the RoadLine object are correct."""
        polyline = Polyline2D.from_array(np.array([[0.0, 1.0], [10.0, 1.0], [20.0, 1.0]]))
        road_line = RoadLine(object_id=6, road_line_type=2, polyline=polyline)
        assert road_line.layer == MapLayer.ROAD_LINE
        assert road_line.object_id == 6
        assert road_line.road_line_type == 2
        assert isinstance(road_line.polyline, Polyline2D)

    def test_init_with_polyline3d(self):
        """Test initialization with Polyline3D."""
        polyline = Polyline3D.from_array(np.array([[0.0, 1.0, 0.0], [10.0, 1.0, 0.0], [20.0, 1.0, 0.0]]))
        road_line = RoadLine(object_id=6, road_line_type=2, polyline=polyline)
        assert isinstance(road_line.polyline, Polyline3D)
        assert road_line.road_line_type == 2

    def test_polyline_length(self):
        """Test that the polyline has correct number of points."""
        polyline = Polyline2D.from_array(np.array([[0.0, 1.0], [10.0, 1.0], [20.0, 1.0], [30.0, 1.0]]))
        road_line = RoadLine(object_id=6, road_line_type=2, polyline=polyline)
        assert len(road_line.polyline.array) == 4

    def test_different_road_line_types(self):
        """Test different road line types."""
        polyline = Polyline2D.from_array(np.array([[0.0, 1.0], [10.0, 1.0]]))
        for line_type in RoadLineType:
            road_line = RoadLine(object_id=6, road_line_type=line_type, polyline=polyline)
            assert road_line.road_line_type == line_type
