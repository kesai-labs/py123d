from pathlib import Path

import shapely

import py123d
from py123d.datatypes.map_objects import RoadEdge
from py123d.parser.opendrive.opendrive_map_parser import iter_xodr_map_objects

CARLA_MAPS = Path(py123d.__file__).parent / "parser" / "opendrive" / "carla_maps"
# Town03 street at x 106-137: from the eastbound lane centre (y -62.5) across its 3.5 m parking lane to the kerb strip
LANE_TO_PARKING = shapely.LineString([(120.7, -62.5), (120.7, -67.9)])
PARKING_LANE_POINT = [120.7, -65.96]


def _road_edges_crossing(segment: shapely.LineString, **parser_options) -> list:
    return [
        map_object
        for map_object in iter_xodr_map_objects(CARLA_MAPS / "Town03.xodr.gz", **parser_options)
        if isinstance(map_object, RoadEdge) and shapely.LineString(map_object.polyline.array[:, :2]).intersects(segment)
    ]


def test_town03_parking_lane_joins_the_drivable_envelope_only_when_enabled():
    assert _road_edges_crossing(LANE_TO_PARKING)
    assert not _road_edges_crossing(LANE_TO_PARKING, drivable_parking_lanes=True)


def test_non_drivable_point_closes_a_drivable_parking_lane_again():
    assert _road_edges_crossing(
        LANE_TO_PARKING, drivable_parking_lanes=True, road_edge_non_drivable_points=[PARKING_LANE_POINT]
    )
