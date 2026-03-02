from py123d.datatypes.detections import TrafficLightDetection, TrafficLights, TrafficLightStatus
from py123d.datatypes.time.time_point import Timestamp


class TestTrafficLightStatus:
    def test_status_values(self):
        """Test that TrafficLightStatus enum has correct values."""
        assert TrafficLightStatus.GREEN.value == 0
        assert TrafficLightStatus.YELLOW.value == 1
        assert TrafficLightStatus.RED.value == 2
        assert TrafficLightStatus.OFF.value == 3
        assert TrafficLightStatus.UNKNOWN.value == 4


class TestTrafficLightDetection:
    def test_creation_with_required_fields(self):
        """Test that TrafficLightDetection can be created with required fields."""
        detection = TrafficLightDetection(lane_id=1, status=TrafficLightStatus.GREEN)
        assert detection.lane_id == 1
        assert detection.status == TrafficLightStatus.GREEN
        assert detection.timestamp is None

    def test_creation_with_timestamp(self):
        """Test that TrafficLightDetection can be created with timestamp."""
        timestamp = Timestamp.from_s(0)
        detection = TrafficLightDetection(
            lane_id=2,
            status=TrafficLightStatus.RED,
            timestamp=timestamp,
        )
        assert detection.lane_id == 2
        assert detection.status == TrafficLightStatus.RED
        assert detection.timestamp == timestamp


class TestTrafficLights:
    def setup_method(self):
        self.detection1 = TrafficLightDetection(lane_id=1, status=TrafficLightStatus.GREEN)
        self.detection2 = TrafficLightDetection(lane_id=2, status=TrafficLightStatus.RED)
        self.detection3 = TrafficLightDetection(lane_id=3, status=TrafficLightStatus.YELLOW)
        self.wrapper = TrafficLights(traffic_light_detections=[self.detection1, self.detection2, self.detection3])

    def test_getitem(self):
        """Test __getitem__ method of TrafficLights."""
        assert self.wrapper[0] == self.detection1
        assert self.wrapper[1] == self.detection2
        assert self.wrapper[2] == self.detection3

    def test_len(self):
        """Test __len__ method of TrafficLights."""
        assert len(self.wrapper) == 3

    def test_iter(self):
        """Test __iter__ method of TrafficLights."""
        detections = list(self.wrapper)
        assert detections == [self.detection1, self.detection2, self.detection3]

    def test_get_detection_by_lane_id_found(self):
        """Test get_detection_by_lane_id method of TrafficLights."""
        result = self.wrapper.get_detection_by_lane_id(2)
        assert result == self.detection2
        assert result.status == TrafficLightStatus.RED

    def test_get_detection_by_lane_id_not_found(self):
        """Test get_detection_by_lane_id method of TrafficLights when not found."""
        result = self.wrapper.get_detection_by_lane_id(99)
        assert result is None

    def test_get_detection_by_lane_id_first_match(self):
        """Test get_detection_by_lane_id method returns first match."""
        duplicate = TrafficLightDetection(lane_id=1, status=TrafficLightStatus.OFF)
        wrapper = TrafficLights(traffic_light_detections=[self.detection1, duplicate])
        result = wrapper.get_detection_by_lane_id(1)
        assert result == self.detection1

    def test_empty_wrapper(self):
        """Test behavior of an empty TrafficLights."""
        empty_wrapper = TrafficLights(traffic_light_detections=[])
        assert len(empty_wrapper) == 0
        assert list(empty_wrapper) == []
        assert empty_wrapper.get_detection_by_lane_id(1) is None
