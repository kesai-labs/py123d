from pathlib import Path

import py123d
from py123d.datatypes.map_objects import LaneType, MapLayer
from py123d.parser.opendrive.opendrive_map_parser import _extract_lanes, _extract_none_lanes, _extract_shoulders
from py123d.parser.opendrive.utils.collection import collect_element_helpers
from py123d.parser.opendrive.xodr_parser.opendrive import XODR

CARLA_MAPS = Path(py123d.__file__).parent / "parser" / "opendrive" / "carla_maps"


def test_town03_shoulders_and_none_lanes_become_typed_lanes():
    xodr = XODR.parse_from_file(CARLA_MAPS / "Town03.xodr.gz")
    _, _, lane_helper_dict, lane_group_helper_dict, _, _, _ = collect_element_helpers(xodr, 1.0, 0.1)

    driving_lanes = _extract_lanes(lane_group_helper_dict)
    shoulders = _extract_shoulders(lane_helper_dict)
    flat_none_lanes, curbed_none_lanes = _extract_none_lanes(lane_helper_dict)
    none_lanes = flat_none_lanes + curbed_none_lanes

    assert len(driving_lanes) == 443 and {lane.lane_type for lane in driving_lanes} == {LaneType.SURFACE_STREET}
    assert len(shoulders) == 586 and {lane.lane_type for lane in shoulders} == {LaneType.SHOULDER}
    assert len(none_lanes) == 137 and {lane.lane_type for lane in none_lanes} == {LaneType.UNDEFINED}

    # All of them share the lane layer, so their ids must not collide
    lane_ids = [lane.object_id for lane in driving_lanes + shoulders + none_lanes]
    assert len(set(lane_ids)) == len(lane_ids)

    # Lane groups only span driving lanes
    assert all(lane.lane_group_id is not None for lane in driving_lanes)
    for lane in shoulders + none_lanes:
        assert lane.layer == MapLayer.LANE
        assert lane.lane_group_id is None
        assert lane.left_lane_id is None and lane.right_lane_id is None
        assert lane.speed_limit_mps is None

    # Shoulders continue into shoulders, and no driving lane leads onto a shoulder or none lane
    shoulder_ids = {lane.object_id for lane in shoulders}
    assert sum(successor_id in shoulder_ids for lane in shoulders for successor_id in lane.successor_ids) == 416
    non_driving_ids = shoulder_ids | {lane.object_id for lane in none_lanes}
    for lane in driving_lanes:
        assert non_driving_ids.isdisjoint(lane.predecessor_ids + lane.successor_ids)
