import pytest

from py123d.datatypes.detections import (
    BoxDetectionAttributes,
    BoxDetectionSE2,
    BoxDetectionSE3,
    BoxDetectionsSE2,
    BoxDetectionsSE3,
    BoxDetectionsSE3Metadata,
)
from py123d.datatypes.detections.box_detection_label import BoxDetectionLabel, DefaultBoxDetectionLabel
from py123d.datatypes.time.timestamp import Timestamp
from py123d.geometry import BoundingBoxSE2, BoundingBoxSE3, PoseSE2, PoseSE3, Vector2D, Vector3D


class DummyBoxDetectionLabel(BoxDetectionLabel):
    CAR = 1
    PEDESTRIAN = 2
    BICYCLE = 3

    def to_default(self):
        mapping = {
            DummyBoxDetectionLabel.CAR: DefaultBoxDetectionLabel.VEHICLE,
            DummyBoxDetectionLabel.PEDESTRIAN: DefaultBoxDetectionLabel.PERSON,
            DummyBoxDetectionLabel.BICYCLE: DefaultBoxDetectionLabel.BICYCLE,
        }
        return mapping[self]


DUMMY_TIMESTAMP = Timestamp.from_s(0.0)
DUMMY_BOX_DETECTIONS_METADATA = BoxDetectionsSE3Metadata(box_detection_label_class=DummyBoxDetectionLabel)

sample_attributes_args = {
    "label": DummyBoxDetectionLabel.CAR,
    "track_token": "sample_token",
    "num_lidar_points": 10,
}


class TestBoxDetectionAttributes:
    def test_initialization(self):
        attributes = BoxDetectionAttributes(**sample_attributes_args)
        assert isinstance(attributes, BoxDetectionAttributes)
        assert attributes.label == DummyBoxDetectionLabel.CAR
        assert attributes.track_token == "sample_token"
        assert attributes.num_lidar_points == 10

    def test_default_label(self):
        attributes = BoxDetectionAttributes(**sample_attributes_args)
        label = attributes.label
        default_label = attributes.default_label
        assert label == DummyBoxDetectionLabel.CAR
        assert label.to_default() == DefaultBoxDetectionLabel.VEHICLE
        assert default_label == DefaultBoxDetectionLabel.VEHICLE

    def test_default_label_with_default_label(self):
        sample_args = sample_attributes_args.copy()
        sample_args["label"] = DefaultBoxDetectionLabel.PERSON
        attributes = BoxDetectionAttributes(**sample_args)
        label = attributes.label
        default_label = attributes.default_label
        assert label == DefaultBoxDetectionLabel.PERSON
        assert default_label == DefaultBoxDetectionLabel.PERSON

    def test_optional_args(self):
        sample_args = {
            "label": DummyBoxDetectionLabel.BICYCLE,
            "track_token": "another_token",
        }
        attributes = BoxDetectionAttributes(**sample_args)
        assert isinstance(attributes, BoxDetectionAttributes)
        assert attributes.label == DummyBoxDetectionLabel.BICYCLE
        assert attributes.track_token == "another_token"
        assert attributes.num_lidar_points is None

    def test_missing_args(self):
        sample_args = {
            "label": DummyBoxDetectionLabel.CAR,
        }
        with pytest.raises(TypeError):
            BoxDetectionAttributes(**sample_args)

        sample_args = {
            "track_token": "token_only",
        }
        with pytest.raises(TypeError):
            BoxDetectionAttributes(**sample_args)


class TestBoxDetectionSE2:
    def setup_method(self):
        self.attributes = BoxDetectionAttributes(**sample_attributes_args)
        self.bounding_box_se2 = BoundingBoxSE2(
            center_se2=PoseSE2(x=0.0, y=0.0, yaw=0.0),
            length=4.0,
            width=2.0,
        )
        self.velocity = None

    def test_initialization(self):
        box_detection = BoxDetectionSE2(
            attributes=self.attributes,
            bounding_box_se2=self.bounding_box_se2,
            velocity_2d=self.velocity,
        )
        assert isinstance(box_detection, BoxDetectionSE2)
        assert box_detection.attributes == self.attributes
        assert box_detection.bounding_box_se2 == self.bounding_box_se2
        assert box_detection.velocity_2d is None

    def test_properties(self):
        box_detection = BoxDetectionSE2(
            attributes=self.attributes,
            bounding_box_se2=self.bounding_box_se2,
            velocity_2d=self.velocity,
        )
        assert box_detection.shapely_polygon == self.bounding_box_se2.shapely_polygon
        assert box_detection.center_se2 == self.bounding_box_se2.center_se2
        assert box_detection.bounding_box_se2 == self.bounding_box_se2

    def test_optional_velocity(self):
        box_detection_no_velo = BoxDetectionSE2(
            attributes=self.attributes,
            bounding_box_se2=self.bounding_box_se2,
        )
        assert isinstance(box_detection_no_velo, BoxDetectionSE2)
        assert box_detection_no_velo.velocity_2d is None

        box_detection_velo = BoxDetectionSE2(
            attributes=self.attributes,
            bounding_box_se2=self.bounding_box_se2,
            velocity_2d=Vector2D(x=1.0, y=0.0),
        )
        assert isinstance(box_detection_velo, BoxDetectionSE2)
        assert box_detection_velo.velocity_2d == Vector2D(x=1.0, y=0.0)


class TestBoxBoxDetectionSE3:
    def setup_method(self):
        self.attributes = BoxDetectionAttributes(**sample_attributes_args)
        self.bounding_box_se3 = BoundingBoxSE3(
            center_se3=PoseSE3(x=0.0, y=0.0, z=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0),
            length=4.0,
            width=2.0,
            height=1.5,
        )
        self.velocity = Vector3D(x=1.0, y=0.0, z=0.0)

    def test_initialization(self):
        box_detection = BoxDetectionSE3(
            attributes=self.attributes,
            bounding_box_se3=self.bounding_box_se3,
            velocity_3d=self.velocity,
        )
        assert isinstance(box_detection, BoxDetectionSE3)
        assert box_detection.attributes == self.attributes
        assert box_detection.bounding_box_se3 == self.bounding_box_se3
        assert box_detection.velocity_3d == self.velocity

    def test_properties(self):
        box_detection = BoxDetectionSE3(
            attributes=self.attributes,
            bounding_box_se3=self.bounding_box_se3,
            velocity_3d=self.velocity,
        )
        assert box_detection.shapely_polygon == self.bounding_box_se3.shapely_polygon
        assert box_detection.center_se3 == self.bounding_box_se3.center_se3
        assert box_detection.center_se2 == self.bounding_box_se3.center_se2
        assert box_detection.bounding_box_se3 == self.bounding_box_se3
        assert box_detection.bounding_box_se2 == self.bounding_box_se3.bounding_box_se2
        assert box_detection.velocity_3d == self.velocity
        assert box_detection.velocity_2d == self.velocity.vector_2d

    def test_box_detection_se2_conversion(self):
        box_detection = BoxDetectionSE3(
            attributes=self.attributes,
            bounding_box_se3=self.bounding_box_se3,
            velocity_3d=Vector3D(x=1.0, y=0.0, z=0.0),
        )
        box_detection_se2 = box_detection.box_detection_se2
        assert isinstance(box_detection_se2, BoxDetectionSE2)
        assert box_detection_se2.attributes == self.attributes
        assert box_detection_se2.bounding_box_se2 == self.bounding_box_se3.bounding_box_se2
        assert box_detection_se2.velocity_2d == Vector2D(x=1.0, y=0.0)

    def test_box_detection_se3_conversion(self):
        box_detection_se2 = BoxDetectionSE2(
            attributes=self.attributes,
            bounding_box_se2=self.bounding_box_se3.bounding_box_se2,
            velocity_2d=Vector2D(x=1.0, y=0.0),
        )
        box_detection_se3 = BoxDetectionSE3(
            attributes=box_detection_se2.attributes,
            bounding_box_se3=self.bounding_box_se3,
            velocity_3d=Vector3D(x=1.0, y=0.0, z=0.0),
        )
        assert isinstance(box_detection_se3, BoxDetectionSE3)
        assert box_detection_se3.attributes == box_detection_se2.attributes
        assert box_detection_se3.bounding_box_se3 == self.bounding_box_se3
        assert box_detection_se3.velocity_2d == Vector2D(x=1.0, y=0.0)

        box_detection_se3_converted = box_detection_se3.box_detection_se2
        assert isinstance(box_detection_se3_converted, BoxDetectionSE2)
        assert box_detection_se3_converted.attributes == box_detection_se2.attributes
        assert box_detection_se3_converted.bounding_box_se2 == box_detection_se2.bounding_box_se2
        assert box_detection_se3_converted.velocity_2d == box_detection_se2.velocity_2d

    def test_optional_velocity(self):
        box_detection_no_velo = BoxDetectionSE3(
            attributes=self.attributes,
            bounding_box_se3=self.bounding_box_se3,
        )
        assert isinstance(box_detection_no_velo, BoxDetectionSE3)
        assert box_detection_no_velo.velocity_3d is None

        box_detection_velo = BoxDetectionSE3(
            attributes=self.attributes,
            bounding_box_se3=self.bounding_box_se3,
            velocity_3d=Vector3D(x=1.0, y=0.0, z=0.0),
        )
        assert isinstance(box_detection_velo, BoxDetectionSE3)
        assert box_detection_velo.velocity_3d == Vector3D(x=1.0, y=0.0, z=0.0)


class TestBoxDetectionsSE2:
    def setup_method(self):
        self.attributes1 = BoxDetectionAttributes(
            label=DummyBoxDetectionLabel.CAR,
            track_token="token1",
            num_lidar_points=10,
        )
        self.attributes2 = BoxDetectionAttributes(
            label=DummyBoxDetectionLabel.PEDESTRIAN,
            track_token="token2",
            num_lidar_points=5,
        )

        self.box_detection1 = BoxDetectionSE2(
            attributes=self.attributes1,
            bounding_box_se2=BoundingBoxSE2(
                center_se2=PoseSE2(x=0.0, y=0.0, yaw=0.0),
                length=4.0,
                width=2.0,
            ),
            velocity_2d=Vector2D(x=1.0, y=0.0),
        )
        self.box_detection2 = BoxDetectionSE2(
            attributes=self.attributes2,
            bounding_box_se2=BoundingBoxSE2(
                center_se2=PoseSE2(x=5.0, y=5.0, yaw=0.0),
                length=1.0,
                width=0.5,
            ),
            velocity_2d=Vector2D(x=0.5, y=0.5),
        )

    def test_initialization(self):
        wrapper = BoxDetectionsSE2(box_detections=[self.box_detection1, self.box_detection2], timestamp=DUMMY_TIMESTAMP)
        assert isinstance(wrapper, BoxDetectionsSE2)
        assert len(wrapper.box_detections) == 2

    def test_empty_initialization(self):
        wrapper = BoxDetectionsSE2(box_detections=[], timestamp=DUMMY_TIMESTAMP)
        assert isinstance(wrapper, BoxDetectionsSE2)
        assert len(wrapper.box_detections) == 0

    def test_getitem(self):
        wrapper = BoxDetectionsSE2(box_detections=[self.box_detection1, self.box_detection2], timestamp=DUMMY_TIMESTAMP)
        assert wrapper[0] == self.box_detection1
        assert wrapper[1] == self.box_detection2

    def test_getitem_out_of_range(self):
        wrapper = BoxDetectionsSE2(box_detections=[self.box_detection1], timestamp=DUMMY_TIMESTAMP)
        with pytest.raises(IndexError):
            _ = wrapper[1]

    def test_len(self):
        wrapper = BoxDetectionsSE2(box_detections=[self.box_detection1, self.box_detection2], timestamp=DUMMY_TIMESTAMP)
        assert len(wrapper) == 2

    def test_len_empty(self):
        wrapper = BoxDetectionsSE2(box_detections=[], timestamp=DUMMY_TIMESTAMP)
        assert len(wrapper) == 0

    def test_iter(self):
        wrapper = BoxDetectionsSE2(box_detections=[self.box_detection1, self.box_detection2], timestamp=DUMMY_TIMESTAMP)
        detections = list(wrapper)
        assert len(detections) == 2
        assert detections[0] == self.box_detection1
        assert detections[1] == self.box_detection2

    def test_get_detection_by_track_token_found(self):
        wrapper = BoxDetectionsSE2(box_detections=[self.box_detection1, self.box_detection2], timestamp=DUMMY_TIMESTAMP)
        detection = wrapper.get_detection_by_track_token("token2")
        assert detection is not None
        assert detection == self.box_detection2
        assert detection.attributes.track_token == "token2"

    def test_get_detection_by_track_token_not_found(self):
        wrapper = BoxDetectionsSE2(box_detections=[self.box_detection1, self.box_detection2], timestamp=DUMMY_TIMESTAMP)
        detection = wrapper.get_detection_by_track_token("nonexistent_token")
        assert detection is None

    def test_get_detection_by_track_token_empty_wrapper(self):
        wrapper = BoxDetectionsSE2(box_detections=[], timestamp=DUMMY_TIMESTAMP)
        detection = wrapper.get_detection_by_track_token("token1")
        assert detection is None

    def test_occupancy_map(self):
        wrapper = BoxDetectionsSE2(box_detections=[self.box_detection1, self.box_detection2], timestamp=DUMMY_TIMESTAMP)
        occupancy_map = wrapper.occupancy_map_2d
        assert occupancy_map is not None
        assert len(occupancy_map.geometries) == 2
        assert len(occupancy_map.ids) == 2
        assert "token1" in occupancy_map.ids
        assert "token2" in occupancy_map.ids

    def test_occupancy_map_cached(self):
        wrapper = BoxDetectionsSE2(box_detections=[self.box_detection1, self.box_detection2], timestamp=DUMMY_TIMESTAMP)
        occupancy_map1 = wrapper.occupancy_map_2d
        occupancy_map2 = wrapper.occupancy_map_2d
        assert occupancy_map1 is occupancy_map2

    def test_occupancy_map_empty(self):
        wrapper = BoxDetectionsSE2(box_detections=[], timestamp=DUMMY_TIMESTAMP)
        occupancy_map = wrapper.occupancy_map_2d
        assert occupancy_map is not None
        assert len(occupancy_map.geometries) == 0
        assert len(occupancy_map.ids) == 0

    def test_timestamp(self):
        wrapper = BoxDetectionsSE2(box_detections=[self.box_detection1, self.box_detection2], timestamp=DUMMY_TIMESTAMP)
        assert wrapper.timestamp == DUMMY_TIMESTAMP

    def test_items_are_se2(self):
        wrapper = BoxDetectionsSE2(box_detections=[self.box_detection1, self.box_detection2], timestamp=DUMMY_TIMESTAMP)
        for detection in wrapper:
            assert isinstance(detection, BoxDetectionSE2)


class TestBoxDetectionsSE3:
    def setup_method(self):
        self.attributes1 = BoxDetectionAttributes(
            label=DummyBoxDetectionLabel.CAR,
            track_token="token1",
            num_lidar_points=10,
        )
        self.attributes2 = BoxDetectionAttributes(
            label=DummyBoxDetectionLabel.PEDESTRIAN,
            track_token="token2",
            num_lidar_points=5,
        )
        self.attributes3 = BoxDetectionAttributes(
            label=DummyBoxDetectionLabel.BICYCLE,
            track_token="token3",
            num_lidar_points=8,
        )

        self.box_detection1 = BoxDetectionSE3(
            attributes=self.attributes1,
            bounding_box_se3=BoundingBoxSE3(
                center_se3=PoseSE3(x=0.0, y=0.0, z=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0),
                length=4.0,
                width=2.0,
                height=1.5,
            ),
            velocity_3d=Vector3D(x=1.0, y=0.0, z=0.0),
        )
        self.box_detection2 = BoxDetectionSE3(
            attributes=self.attributes2,
            bounding_box_se3=BoundingBoxSE3(
                center_se3=PoseSE3(x=5.0, y=5.0, z=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0),
                length=1.0,
                width=0.5,
                height=1.8,
            ),
            velocity_3d=Vector3D(x=0.5, y=0.5, z=0.0),
        )
        self.box_detection3 = BoxDetectionSE3(
            attributes=self.attributes3,
            bounding_box_se3=BoundingBoxSE3(
                center_se3=PoseSE3(x=10.0, y=10.0, z=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0),
                length=2.0,
                width=1.0,
                height=1.5,
            ),
            velocity_3d=Vector3D(x=0.0, y=1.0, z=0.0),
        )

    def test_initialization(self):
        wrapper = BoxDetectionsSE3(
            box_detections=[self.box_detection1, self.box_detection2],
            timestamp=DUMMY_TIMESTAMP,
            metadata=DUMMY_BOX_DETECTIONS_METADATA,
        )
        assert isinstance(wrapper, BoxDetectionsSE3)
        assert len(wrapper.box_detections) == 2

    def test_empty_initialization(self):
        wrapper = BoxDetectionsSE3(box_detections=[], timestamp=DUMMY_TIMESTAMP, metadata=DUMMY_BOX_DETECTIONS_METADATA)
        assert isinstance(wrapper, BoxDetectionsSE3)
        assert len(wrapper.box_detections) == 0

    def test_getitem(self):
        wrapper = BoxDetectionsSE3(
            box_detections=[self.box_detection1, self.box_detection2],
            timestamp=DUMMY_TIMESTAMP,
            metadata=DUMMY_BOX_DETECTIONS_METADATA,
        )
        assert wrapper[0] == self.box_detection1
        assert wrapper[1] == self.box_detection2

    def test_getitem_out_of_range(self):
        wrapper = BoxDetectionsSE3(
            box_detections=[self.box_detection1], timestamp=DUMMY_TIMESTAMP, metadata=DUMMY_BOX_DETECTIONS_METADATA
        )
        with pytest.raises(IndexError):
            _ = wrapper[1]

    def test_len(self):
        wrapper = BoxDetectionsSE3(
            box_detections=[self.box_detection1, self.box_detection2, self.box_detection3],
            timestamp=DUMMY_TIMESTAMP,
            metadata=DUMMY_BOX_DETECTIONS_METADATA,
        )
        assert len(wrapper) == 3

    def test_len_empty(self):
        wrapper = BoxDetectionsSE3(box_detections=[], timestamp=DUMMY_TIMESTAMP, metadata=DUMMY_BOX_DETECTIONS_METADATA)
        assert len(wrapper) == 0

    def test_iter(self):
        wrapper = BoxDetectionsSE3(
            box_detections=[self.box_detection1, self.box_detection2],
            timestamp=DUMMY_TIMESTAMP,
            metadata=DUMMY_BOX_DETECTIONS_METADATA,
        )
        detections = list(wrapper)
        assert len(detections) == 2
        assert detections[0] == self.box_detection1
        assert detections[1] == self.box_detection2

    def test_get_detection_by_track_token_found(self):
        wrapper = BoxDetectionsSE3(
            box_detections=[self.box_detection1, self.box_detection2, self.box_detection3],
            timestamp=DUMMY_TIMESTAMP,
            metadata=DUMMY_BOX_DETECTIONS_METADATA,
        )
        detection = wrapper.get_detection_by_track_token("token2")
        assert detection is not None
        assert detection == self.box_detection2
        assert detection.attributes.track_token == "token2"

    def test_get_detection_by_track_token_not_found(self):
        wrapper = BoxDetectionsSE3(
            box_detections=[self.box_detection1, self.box_detection2],
            timestamp=DUMMY_TIMESTAMP,
            metadata=DUMMY_BOX_DETECTIONS_METADATA,
        )
        detection = wrapper.get_detection_by_track_token("nonexistent_token")
        assert detection is None

    def test_get_detection_by_track_token_empty_wrapper(self):
        wrapper = BoxDetectionsSE3(box_detections=[], timestamp=DUMMY_TIMESTAMP, metadata=DUMMY_BOX_DETECTIONS_METADATA)
        detection = wrapper.get_detection_by_track_token("token1")
        assert detection is None

    def test_occupancy_map(self):
        wrapper = BoxDetectionsSE3(
            box_detections=[self.box_detection1, self.box_detection2],
            timestamp=DUMMY_TIMESTAMP,
            metadata=DUMMY_BOX_DETECTIONS_METADATA,
        )
        occupancy_map = wrapper.occupancy_map_2d
        assert occupancy_map is not None
        assert len(occupancy_map.geometries) == 2
        assert len(occupancy_map.ids) == 2
        assert "token1" in occupancy_map.ids
        assert "token2" in occupancy_map.ids

    def test_occupancy_map_cached(self):
        wrapper = BoxDetectionsSE3(
            box_detections=[self.box_detection1, self.box_detection2],
            timestamp=DUMMY_TIMESTAMP,
            metadata=DUMMY_BOX_DETECTIONS_METADATA,
        )
        occupancy_map1 = wrapper.occupancy_map_2d
        occupancy_map2 = wrapper.occupancy_map_2d
        assert occupancy_map1 is occupancy_map2

    def test_occupancy_map_empty(self):
        wrapper = BoxDetectionsSE3(box_detections=[], timestamp=DUMMY_TIMESTAMP, metadata=DUMMY_BOX_DETECTIONS_METADATA)
        occupancy_map = wrapper.occupancy_map_2d
        assert occupancy_map is not None
        assert len(occupancy_map.geometries) == 0
        assert len(occupancy_map.ids) == 0

    def test_timestamp(self):
        wrapper = BoxDetectionsSE3(
            box_detections=[self.box_detection1, self.box_detection2],
            timestamp=DUMMY_TIMESTAMP,
            metadata=DUMMY_BOX_DETECTIONS_METADATA,
        )
        assert wrapper.timestamp == DUMMY_TIMESTAMP

    def test_items_are_se3(self):
        wrapper = BoxDetectionsSE3(
            box_detections=[self.box_detection1, self.box_detection2, self.box_detection3],
            timestamp=DUMMY_TIMESTAMP,
            metadata=DUMMY_BOX_DETECTIONS_METADATA,
        )
        for detection in wrapper:
            assert isinstance(detection, BoxDetectionSE3)
