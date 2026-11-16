from pathlib import Path

import numpy as np

import py123d
from py123d.datatypes import Lane, StopZone, StopZoneType, TrafficLight
from py123d.parser.opendrive.opendrive_map_parser import iter_xodr_map_objects

CARLA_MAPS = Path(py123d.__file__).parent / "parser" / "opendrive" / "carla_maps"


def test_town03_traffic_lights_link_lanes_stop_zones_and_junctions():
    objects = list(iter_xodr_map_objects(CARLA_MAPS / "Town03.xodr.gz"))
    lights = [obj for obj in objects if isinstance(obj, TrafficLight)]
    lanes = {obj.object_id: obj for obj in objects if isinstance(obj, Lane)}
    stop_zones = {obj.object_id: obj for obj in objects if isinstance(obj, StopZone)}

    traffic_light_stop_zones = [sz for sz in stop_zones.values() if sz.stop_zone_type == StopZoneType.TRAFFIC_LIGHT]
    assert len(lights) == 38 and len(traffic_light_stop_zones) == 38
    assert len({light.object_id for light in lights}) == len(lights)

    for light in lights:
        assert light.lane_ids and all(lane_id in lanes for lane_id in light.lane_ids)
        assert len(light.stop_zone_ids) == 1 and light.stop_zone_ids[0] in stop_zones
        assert stop_zones[light.stop_zone_ids[0]].stop_zone_type == StopZoneType.TRAFFIC_LIGHT
        assert light.intersection_id is not None
        # CARLA signal faces: width and height from the <signal>, depth from the default
        box = light.bounding_box_se3
        assert box.length == 0.3 and 0.4 < box.width < 0.6 and 1.0 < box.height < 1.3
        # The housing faces the driving direction of the controlled lanes
        centerline = lanes[light.lane_ids[0]].centerline.array
        direction = centerline[1] - centerline[0]
        lane_yaw = np.arctan2(direction[1], direction[0])
        assert abs((light.pose_se3.yaw - lane_yaw + np.pi) % (2 * np.pi) - np.pi) < np.radians(5.0)
