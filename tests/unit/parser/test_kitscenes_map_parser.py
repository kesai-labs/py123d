import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pyproj
import pytest

from py123d.datatypes import Lane, LaneGroup, RoadEdge, RoadLine, RoadLineType, StopZone, StopZoneType
from py123d.parser.kitscenes.kitscenes_map_parser import REVERSED_LANE_ID_OFFSET, KITScenesMapParser

ORIGIN_LAT, ORIGIN_LON = 49.0, 8.4
SCENE_ID = "test-scene"

# Node positions in meters (x east, y north) relative to the map origin.
NODES: Dict[int, Tuple[float, float]] = {
    1: (0.0, 0.0),
    2: (10.0, 0.0),
    3: (20.0, 0.0),
    4: (0.0, 3.5),
    5: (10.0, 3.5),
    6: (20.0, 3.5),
    7: (0.0, 7.0),
    8: (10.0, 7.0),
    9: (0.0, 20.0),
    10: (10.0, 20.0),
    11: (0.0, 23.0),
    12: (10.0, 23.0),
    13: (0.0, 47.0),
    14: (10.0, 47.0),
    15: (0.0, 43.5),
    16: (10.0, 43.5),
    17: (0.0, 40.5),
    18: (10.0, 40.5),
    19: (0.0, 37.0),
    20: (10.0, 37.0),
    21: (0.0, 39.5),
    22: (10.0, 39.5),
}
WAYS: Dict[int, Tuple[List[int], Dict[str, str]]] = {
    20: ([5, 4], {"type": "line_thin", "subtype": "dashed"}),  # stored against the driving direction
    21: ([1, 2], {"type": "curbstone", "subtype": "high"}),
    22: ([5, 6], {"type": "line_thin", "subtype": "solid"}),
    23: ([2, 3], {"type": "curbstone", "subtype": "high"}),
    24: ([7, 8], {"type": "curbstone", "subtype": "high"}),
    25: ([11, 12], {"type": "line_thin", "subtype": "solid"}),
    26: ([9, 10], {"type": "line_thin", "subtype": "solid"}),
    27: ([2, 5], {"type": "stop_line"}),
    30: ([13, 14], {"type": "virtual"}),
    31: ([15, 16], {"type": "virtual"}),
    32: ([17, 18], {"type": "virtual"}),
    33: ([19, 20], {"type": "virtual"}),
    34: ([21, 22], {"type": "virtual"}),
}
# Lanelet ID -> (left way, right way, tags, regulatory elements)
LANELETS: Dict[int, Tuple[int, int, Dict[str, str], List[int]]] = {
    100: (20, 21, {"subtype": "road", "speed_limit": "50"}, [200]),
    101: (22, 23, {"subtype": "road"}, []),
    102: (24, 20, {"subtype": "road"}, []),
    103: (25, 26, {"subtype": "road", "one_way": "0"}, []),
    # Lane split: lanelets 111 and 113 overlap and share their left boundary with the right boundary of 110.
    110: (30, 31, {"subtype": "road"}, []),
    111: (31, 32, {"subtype": "road"}, []),
    112: (32, 33, {"subtype": "road"}, []),
    113: (31, 34, {"subtype": "road"}, []),
}


def _to_lat_lon(x: float, y: float) -> Tuple[float, float]:
    """Inverse of the parser's projection: UTM zone 32N, relative to the UTM coordinates of the origin."""
    to_utm = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
    origin_x, origin_y = to_utm.transform(ORIGIN_LON, ORIGIN_LAT)
    lon, lat = to_utm.transform(origin_x + x, origin_y + y, direction="INVERSE")
    return lat, lon


def _write_map(scene_dir: Path) -> None:
    lines = ['<?xml version="1.0"?>', '<osm version="0.6" generator="lanelet2">']
    for node_id, (x, y) in NODES.items():
        lat, lon = _to_lat_lon(x, y)
        lines.append(f'<node id="{node_id}" lat="{lat:.12f}" lon="{lon:.12f}"><tag k="ele" v="100.0"/></node>')
    for way_id, (node_ids, tags) in WAYS.items():
        lines.append(f'<way id="{way_id}">')
        lines += [f'<nd ref="{node_id}"/>' for node_id in node_ids]
        lines += [f'<tag k="{key}" v="{value}"/>' for key, value in tags.items()]
        lines.append("</way>")
    for lanelet_id, (left_way, right_way, tags, regulatory_elements) in LANELETS.items():
        lines.append(f'<relation id="{lanelet_id}">')
        lines.append(f'<member type="way" ref="{left_way}" role="left"/>')
        lines.append(f'<member type="way" ref="{right_way}" role="right"/>')
        lines += [f'<member type="relation" ref="{ref}" role="regulatory_element"/>' for ref in regulatory_elements]
        lines += [f'<tag k="{key}" v="{value}"/>' for key, value in {"type": "lanelet", **tags}.items()]
        lines.append("</relation>")
    lines += [
        '<relation id="200"><member type="way" ref="27" role="ref_line"/>',
        '<tag k="type" v="regulatory_element"/><tag k="subtype" v="traffic_light"/></relation>',
        "</osm>",
    ]
    (scene_dir / "maps").mkdir(parents=True)
    (scene_dir / "maps" / "map.osm").write_text("\n".join(lines))
    (scene_dir / "maps" / "origin.json").write_text(json.dumps({"latitude": ORIGIN_LAT, "longitude": ORIGIN_LON}))


@pytest.fixture
def map_objects(tmp_path: Path):
    _write_map(tmp_path / "data" / "val" / SCENE_ID)
    parser = KITScenesMapParser(tmp_path, "val", SCENE_ID, location="karlsruhe")
    objects = list(parser.iter_map_objects())
    lanes = {lane.object_id: lane for lane in objects if isinstance(lane, Lane)}
    return parser, objects, lanes


class TestKITScenesMapParser:
    def test_map_metadata_is_per_log(self, map_objects):
        parser, _, _ = map_objects
        metadata = parser.get_map_metadata()
        assert metadata.map_is_per_log
        assert metadata.split == "kitscenes_val"
        assert metadata.log_name == SCENE_ID

    def test_boundaries_follow_driving_direction(self, map_objects):
        """Lanelet 100 stores its left boundary backwards; both boundaries must run east."""
        _, _, lanes = map_objects
        left, right = lanes[100].left_boundary.array, lanes[100].right_boundary.array
        np.testing.assert_allclose(left[[0, -1], :2], [[0.0, 3.5], [10.0, 3.5]], atol=0.05)
        np.testing.assert_allclose(right[[0, -1], :2], [[0.0, 0.0], [10.0, 0.0]], atol=0.05)
        np.testing.assert_allclose(left[:, 2], 100.0)

    def test_topology_from_shared_geometry(self, map_objects):
        _, _, lanes = map_objects
        assert lanes[100].successor_ids == [101]
        assert lanes[101].predecessor_ids == [100]
        assert lanes[100].left_lane_id == 102
        assert lanes[102].right_lane_id == 100
        assert lanes[100].speed_limit_mps == pytest.approx(50 / 3.6)

    def test_bidirectional_lanelet_is_emitted_in_both_directions(self, map_objects):
        _, _, lanes = map_objects
        reversed_lane = lanes[103 + REVERSED_LANE_ID_OFFSET]
        np.testing.assert_allclose(reversed_lane.left_boundary.array[0, :2], [10.0, 20.0], atol=0.05)
        assert reversed_lane.left_lane_id is None

    def test_lane_groups_ordered_left_to_right(self, map_objects):
        _, objects, lanes = map_objects
        lane_groups = {group.object_id: group for group in objects if isinstance(group, LaneGroup)}
        assert lanes[100].lane_group_id == lanes[102].lane_group_id
        assert lane_groups[lanes[100].lane_group_id].lane_ids == [102, 100]

    def test_lanes_sharing_a_boundary_are_in_exactly_one_lane_group(self, map_objects):
        """Lanelets 111 and 113 both have 110 as left neighbor, which can only point back at one of them."""
        _, objects, lanes = map_objects
        lane_groups = {group.object_id: group for group in objects if isinstance(group, LaneGroup)}
        assert lanes[111].left_lane_id == lanes[113].left_lane_id == 110
        assert lanes[110].right_lane_id == 113

        grouped_lane_ids = [lane_id for group in lane_groups.values() for lane_id in group.lane_ids]
        assert sorted(grouped_lane_ids) == sorted(lanes)
        for lane_id, lane in lanes.items():
            assert lane_id in lane_groups[lane.lane_group_id].lane_ids
        assert lane_groups[lanes[110].lane_group_id].lane_ids == [110, 113]
        assert lane_groups[lanes[111].lane_group_id].lane_ids == [111, 112]

    def test_road_edges_road_lines_and_stop_zones(self, map_objects):
        _, objects, _ = map_objects
        assert {edge.object_id for edge in objects if isinstance(edge, RoadEdge)} == {21, 23, 24}
        road_lines = {line.object_id: line for line in objects if isinstance(line, RoadLine)}
        assert road_lines[20].road_line_type == RoadLineType.DASHED_WHITE
        stop_zones = [zone for zone in objects if isinstance(zone, StopZone)]
        assert len(stop_zones) == 1
        assert stop_zones[0].stop_zone_type == StopZoneType.TRAFFIC_LIGHT
        assert list(stop_zones[0].lane_ids) == [100]
