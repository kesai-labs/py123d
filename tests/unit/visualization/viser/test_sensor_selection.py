"""Which lidar and radar sensors start checked in the viewer's per-sensor checklists."""

from typing import Dict, List
from unittest.mock import MagicMock

import pytest

from py123d.datatypes.sensors.lidar import LidarID
from py123d.datatypes.sensors.radar import RadarID
from py123d.visualization.viser.elements.lidar_element import LidarConfig, LidarElement
from py123d.visualization.viser.elements.radar_element import RadarConfig, RadarElement


def _checked_sensors(element_class, config, available_attribute: str, available_ids: List) -> Dict[str, bool]:
    """Builds the element's GUI on a mocked server, returns the initial value of every sensor checkbox."""
    context = MagicMock()
    setattr(context.scene, available_attribute, available_ids)
    server = MagicMock()
    element_class(context, config).create_gui(server)
    sensor_labels = {sensor_id.serialize(lower=False) for sensor_id in available_ids}
    return {
        call.args[0]: call.args[1] for call in server.gui.add_checkbox.call_args_list if call.args[0] in sensor_labels
    }


@pytest.mark.parametrize(
    "element_class, config, available_attribute, merged_id, sensor_ids",
    [
        (LidarElement, LidarConfig(), "available_lidar_ids", LidarID.LIDAR_MERGED, [LidarID.LIDAR_TOP]),
        (RadarElement, RadarConfig(), "available_radar_ids", RadarID.RADAR_MERGED, [RadarID.RADAR_FRONT]),
    ],
)
class TestInitialSensorSelection:
    def test_merged_sensor_is_the_default(self, element_class, config, available_attribute, merged_id, sensor_ids):
        """With a merged sensor in the scene, only that one starts checked."""
        checked = _checked_sensors(element_class, config, available_attribute, [*sensor_ids, merged_id])
        assert checked == {
            merged_id.serialize(lower=False): True,
            **{sensor_id.serialize(lower=False): False for sensor_id in sensor_ids},
        }

    def test_scene_without_merged_sensor(self, element_class, config, available_attribute, merged_id, sensor_ids):
        """Without the configured sensor in the scene, every available sensor starts checked."""
        checked = _checked_sensors(element_class, config, available_attribute, sensor_ids)
        assert checked == {sensor_id.serialize(lower=False): True for sensor_id in sensor_ids}
