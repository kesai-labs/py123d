from pathlib import Path

import py123d
from py123d.datatypes.map_objects import StopZoneType
from py123d.parser.opendrive.utils.collection import STOP_STENCIL_NAME, collect_element_helpers
from py123d.parser.opendrive.utils.stop_zone_helper import create_stop_zones_from_signals
from py123d.parser.opendrive.xodr_parser.opendrive import XODR

CARLA_MAPS = Path(py123d.__file__).parent / "parser" / "opendrive" / "carla_maps"


def test_town03_stop_stencils_become_stop_zones():
    xodr = XODR.parse_from_file(CARLA_MAPS / "Town03.xodr.gz")
    _, _, lane_helper_dict, _, _, _, signal_dict = collect_element_helpers(xodr, 1.0, 0.1)

    stencils = [obj for road in xodr.roads for obj in road.objects if obj.name == STOP_STENCIL_NAME]
    derived = {sid: helper for sid, helper in signal_dict.items() if helper.is_derived}
    assert len(stencils) == 22
    assert len(derived) == 20  # 2424 and 2370 face no driving lane, CARLA spawns no stop actor for them either
    assert all(helper.xodr_signal.type == "206" and len(helper.lane_ids) == 1 for helper in derived.values())

    zones = create_stop_zones_from_signals(signal_dict, lane_helper_dict)
    stop_zones = {sid: zone for sid, zone in zones.items() if zone.stop_zone_type == StopZoneType.STOP_SIGN}
    assert len(stop_zones) == 15  # one zone per approach, multi-lane approaches merge
    assert sum(zone.stop_zone_type == StopZoneType.TRAFFIC_LIGHT for zone in zones.values()) == 38
    assert stop_zones[2378].lane_ids == ["1933_1_right_-1"]  # mid-connector stencil splits the connector
    assert set(stop_zones[2160].lane_ids) == {
        "1165_0_right_-2",
        "1156_0_right_-1",
        "1158_0_right_-1",
        "1165_0_right_-1",
    }
