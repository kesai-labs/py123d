from pathlib import Path

import shapely

import py123d
from py123d.parser.opendrive.opendrive_map_parser import _iter_intersections
from py123d.parser.opendrive.utils.collection import collect_element_helpers
from py123d.parser.opendrive.utils.id_system import road_id_from_lane_id
from py123d.parser.opendrive.xodr_parser.opendrive import XODR

CARLA_MAPS = Path(py123d.__file__).parent / "parser" / "opendrive" / "carla_maps"


def test_town04_junction_road_shoulders_lie_inside_their_intersection():
    xodr = XODR.parse_from_file(CARLA_MAPS / "Town04.xodr.gz")
    road_dict, junction_dict, lane_helper_dict, lane_group_helper_dict, _, _, _ = collect_element_helpers(
        xodr, 1.0, 0.1
    )

    intersections = {
        intersection.object_id: shapely.Polygon(intersection.outline_3d.array[:, :2])
        for intersection in _iter_intersections(junction_dict, lane_group_helper_dict, lane_helper_dict, road_dict)
    }
    assert len(intersections) == 27

    # Junction 255: the right-turn connector 271 runs over the shoulder lane -5 of junction road 267 (its inner
    # corner apron) and the none lane -4 (the gore); both belong to the junction surface.
    junction_255 = intersections[255]
    for lane_id, lane_helper in lane_helper_dict.items():
        if road_id_from_lane_id(lane_id) != "267" or lane_helper.type not in {"shoulder", "none"}:
            continue
        polygon = lane_helper.shapely_polygon
        assert polygon.intersection(junction_255).area > 0.99 * polygon.area, lane_id

    # Shoulders of the approach road 42 stay outside the junction
    for lane_id, lane_helper in lane_helper_dict.items():
        if road_id_from_lane_id(lane_id) == "42" and lane_helper.type == "shoulder":
            assert lane_helper.shapely_polygon.intersection(junction_255).area < 0.5
