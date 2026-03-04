from __future__ import annotations

from py123d.datatypes.detections.box_detection_label import (
    BOX_DETECTION_LABEL_REGISTRY,  # noqa: F401 — re-exported for backward compatibility
    BoxDetectionLabel,
    DefaultBoxDetectionLabel,
    register_box_detection_label,
)


@register_box_detection_label
class AV2SensorBoxDetectionLabel(BoxDetectionLabel):
    """Argoverse 2 Sensor dataset annotation categories."""

    ANIMAL = 0
    ARTICULATED_BUS = 1
    BICYCLE = 2
    BICYCLIST = 3
    BOLLARD = 4
    BOX_TRUCK = 5
    BUS = 6
    CONSTRUCTION_BARREL = 7
    CONSTRUCTION_CONE = 8
    DOG = 9
    LARGE_VEHICLE = 10
    MESSAGE_BOARD_TRAILER = 11
    MOBILE_PEDESTRIAN_CROSSING_SIGN = 12
    MOTORCYCLE = 13
    MOTORCYCLIST = 14
    OFFICIAL_SIGNALER = 15
    PEDESTRIAN = 16
    RAILED_VEHICLE = 17
    REGULAR_VEHICLE = 18
    SCHOOL_BUS = 19
    SIGN = 20
    STOP_SIGN = 21
    STROLLER = 22
    TRAFFIC_LIGHT_TRAILER = 23
    TRUCK = 24
    TRUCK_CAB = 25
    VEHICULAR_TRAILER = 26
    WHEELCHAIR = 27
    WHEELED_DEVICE = 28
    WHEELED_RIDER = 29

    def to_default(self) -> DefaultBoxDetectionLabel:
        """Inherited, see superclass."""
        mapping = {
            AV2SensorBoxDetectionLabel.ANIMAL: DefaultBoxDetectionLabel.ANIMAL,
            AV2SensorBoxDetectionLabel.ARTICULATED_BUS: DefaultBoxDetectionLabel.VEHICLE,
            AV2SensorBoxDetectionLabel.BICYCLE: DefaultBoxDetectionLabel.BICYCLE,
            AV2SensorBoxDetectionLabel.BICYCLIST: DefaultBoxDetectionLabel.PERSON,
            AV2SensorBoxDetectionLabel.BOLLARD: DefaultBoxDetectionLabel.GENERIC_OBJECT,
            AV2SensorBoxDetectionLabel.BOX_TRUCK: DefaultBoxDetectionLabel.VEHICLE,
            AV2SensorBoxDetectionLabel.BUS: DefaultBoxDetectionLabel.VEHICLE,
            AV2SensorBoxDetectionLabel.CONSTRUCTION_BARREL: DefaultBoxDetectionLabel.TRAFFIC_CONE,
            AV2SensorBoxDetectionLabel.CONSTRUCTION_CONE: DefaultBoxDetectionLabel.TRAFFIC_CONE,
            AV2SensorBoxDetectionLabel.DOG: DefaultBoxDetectionLabel.ANIMAL,
            AV2SensorBoxDetectionLabel.LARGE_VEHICLE: DefaultBoxDetectionLabel.VEHICLE,
            AV2SensorBoxDetectionLabel.MESSAGE_BOARD_TRAILER: DefaultBoxDetectionLabel.VEHICLE,
            AV2SensorBoxDetectionLabel.MOBILE_PEDESTRIAN_CROSSING_SIGN: DefaultBoxDetectionLabel.TRAFFIC_SIGN,
            AV2SensorBoxDetectionLabel.MOTORCYCLE: DefaultBoxDetectionLabel.BICYCLE,
            AV2SensorBoxDetectionLabel.MOTORCYCLIST: DefaultBoxDetectionLabel.PERSON,
            AV2SensorBoxDetectionLabel.OFFICIAL_SIGNALER: DefaultBoxDetectionLabel.PERSON,
            AV2SensorBoxDetectionLabel.PEDESTRIAN: DefaultBoxDetectionLabel.PERSON,
            AV2SensorBoxDetectionLabel.RAILED_VEHICLE: DefaultBoxDetectionLabel.TRAIN,
            AV2SensorBoxDetectionLabel.REGULAR_VEHICLE: DefaultBoxDetectionLabel.VEHICLE,
            AV2SensorBoxDetectionLabel.SCHOOL_BUS: DefaultBoxDetectionLabel.VEHICLE,
            AV2SensorBoxDetectionLabel.SIGN: DefaultBoxDetectionLabel.TRAFFIC_SIGN,
            AV2SensorBoxDetectionLabel.STOP_SIGN: DefaultBoxDetectionLabel.TRAFFIC_SIGN,
            AV2SensorBoxDetectionLabel.STROLLER: DefaultBoxDetectionLabel.PERSON,
            AV2SensorBoxDetectionLabel.TRAFFIC_LIGHT_TRAILER: DefaultBoxDetectionLabel.VEHICLE,
            AV2SensorBoxDetectionLabel.TRUCK: DefaultBoxDetectionLabel.VEHICLE,
            AV2SensorBoxDetectionLabel.TRUCK_CAB: DefaultBoxDetectionLabel.VEHICLE,
            AV2SensorBoxDetectionLabel.VEHICULAR_TRAILER: DefaultBoxDetectionLabel.VEHICLE,
            AV2SensorBoxDetectionLabel.WHEELCHAIR: DefaultBoxDetectionLabel.PERSON,
            AV2SensorBoxDetectionLabel.WHEELED_DEVICE: DefaultBoxDetectionLabel.PERSON,
            AV2SensorBoxDetectionLabel.WHEELED_RIDER: DefaultBoxDetectionLabel.PERSON,
        }
        return mapping[self]


@register_box_detection_label
class KITTI360BoxDetectionLabel(BoxDetectionLabel):
    """KITTI-360 dataset annotation categories."""

    BICYCLE = 0
    BOX = 1
    BUS = 2
    CAR = 3
    CARAVAN = 4
    LAMP = 5
    MOTORCYCLE = 6
    PERSON = 7
    POLE = 8
    RIDER = 9
    SMALLPOLE = 10
    STOP = 11
    TRAFFIC_LIGHT = 12
    TRAFFIC_SIGN = 13
    TRAILER = 14
    TRAIN = 15
    TRASH_BIN = 16
    TRUCK = 17
    VENDING_MACHINE = 18

    def to_default(self) -> DefaultBoxDetectionLabel:
        """Inherited, see superclass."""
        mapping = {
            KITTI360BoxDetectionLabel.BICYCLE: DefaultBoxDetectionLabel.BICYCLE,
            KITTI360BoxDetectionLabel.BOX: DefaultBoxDetectionLabel.GENERIC_OBJECT,
            KITTI360BoxDetectionLabel.BUS: DefaultBoxDetectionLabel.VEHICLE,
            KITTI360BoxDetectionLabel.CAR: DefaultBoxDetectionLabel.VEHICLE,
            KITTI360BoxDetectionLabel.CARAVAN: DefaultBoxDetectionLabel.VEHICLE,
            KITTI360BoxDetectionLabel.LAMP: DefaultBoxDetectionLabel.GENERIC_OBJECT,
            KITTI360BoxDetectionLabel.MOTORCYCLE: DefaultBoxDetectionLabel.BICYCLE,
            KITTI360BoxDetectionLabel.PERSON: DefaultBoxDetectionLabel.PERSON,
            KITTI360BoxDetectionLabel.POLE: DefaultBoxDetectionLabel.GENERIC_OBJECT,
            KITTI360BoxDetectionLabel.RIDER: DefaultBoxDetectionLabel.PERSON,
            KITTI360BoxDetectionLabel.SMALLPOLE: DefaultBoxDetectionLabel.GENERIC_OBJECT,
            KITTI360BoxDetectionLabel.STOP: DefaultBoxDetectionLabel.TRAFFIC_SIGN,
            KITTI360BoxDetectionLabel.TRAFFIC_LIGHT: DefaultBoxDetectionLabel.TRAFFIC_SIGN,
            KITTI360BoxDetectionLabel.TRAFFIC_SIGN: DefaultBoxDetectionLabel.TRAFFIC_SIGN,
            KITTI360BoxDetectionLabel.TRAILER: DefaultBoxDetectionLabel.VEHICLE,
            KITTI360BoxDetectionLabel.TRAIN: DefaultBoxDetectionLabel.VEHICLE,
            KITTI360BoxDetectionLabel.TRASH_BIN: DefaultBoxDetectionLabel.GENERIC_OBJECT,
            KITTI360BoxDetectionLabel.TRUCK: DefaultBoxDetectionLabel.VEHICLE,
            KITTI360BoxDetectionLabel.VENDING_MACHINE: DefaultBoxDetectionLabel.GENERIC_OBJECT,
        }
        return mapping[self]


@register_box_detection_label
class NuPlanBoxDetectionLabel(BoxDetectionLabel):
    """Semantic labels for nuPlan bounding box detections."""

    VEHICLE = 0
    """Includes all four or more wheeled vehicles, as well as trailers."""

    BICYCLE = 1
    """Includes bicycles, motorcycles and tricycles."""

    PEDESTRIAN = 2
    """All types of pedestrians, incl. strollers and wheelchairs."""

    TRAFFIC_CONE = 3
    """Cones that are temporarily placed to control the flow of traffic."""

    BARRIER = 4
    """Solid barriers that can be either temporary or permanent."""

    CZONE_SIGN = 5
    """Temporary signs that indicate construction zones."""

    GENERIC_OBJECT = 6
    """Animals, debris, pushable/pullable objects, permanent poles."""

    def to_default(self) -> DefaultBoxDetectionLabel:
        """Inherited, see superclass."""
        mapping = {
            NuPlanBoxDetectionLabel.VEHICLE: DefaultBoxDetectionLabel.VEHICLE,
            NuPlanBoxDetectionLabel.BICYCLE: DefaultBoxDetectionLabel.BICYCLE,
            NuPlanBoxDetectionLabel.PEDESTRIAN: DefaultBoxDetectionLabel.PERSON,
            NuPlanBoxDetectionLabel.TRAFFIC_CONE: DefaultBoxDetectionLabel.TRAFFIC_CONE,
            NuPlanBoxDetectionLabel.BARRIER: DefaultBoxDetectionLabel.BARRIER,
            NuPlanBoxDetectionLabel.CZONE_SIGN: DefaultBoxDetectionLabel.TRAFFIC_SIGN,
            NuPlanBoxDetectionLabel.GENERIC_OBJECT: DefaultBoxDetectionLabel.GENERIC_OBJECT,
        }
        return mapping[self]


@register_box_detection_label
class NuScenesBoxDetectionLabel(BoxDetectionLabel):
    """
    Semantic labels for nuScenes bounding box detections.
    [1] https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/instructions_nuscenes.md#labels
    """

    VEHICLE_CAR = 0
    VEHICLE_TRUCK = 1
    VEHICLE_BUS_BENDY = 2
    VEHICLE_BUS_RIGID = 3
    VEHICLE_CONSTRUCTION = 4
    VEHICLE_EMERGENCY_AMBULANCE = 5
    VEHICLE_EMERGENCY_POLICE = 6
    VEHICLE_TRAILER = 7
    VEHICLE_BICYCLE = 8
    VEHICLE_MOTORCYCLE = 9
    HUMAN_PEDESTRIAN_ADULT = 10
    HUMAN_PEDESTRIAN_CHILD = 11
    HUMAN_PEDESTRIAN_CONSTRUCTION_WORKER = 12
    HUMAN_PEDESTRIAN_PERSONAL_MOBILITY = 13
    HUMAN_PEDESTRIAN_POLICE_OFFICER = 14
    HUMAN_PEDESTRIAN_STROLLER = 15
    HUMAN_PEDESTRIAN_WHEELCHAIR = 16
    MOVABLE_OBJECT_TRAFFICCONE = 17
    MOVABLE_OBJECT_BARRIER = 18
    MOVABLE_OBJECT_PUSHABLE_PULLABLE = 19
    MOVABLE_OBJECT_DEBRIS = 20
    STATIC_OBJECT_BICYCLE_RACK = 21
    ANIMAL = 22

    def to_default(self):
        """Inherited, see superclass."""
        mapping = {
            NuScenesBoxDetectionLabel.VEHICLE_CAR: DefaultBoxDetectionLabel.VEHICLE,
            NuScenesBoxDetectionLabel.VEHICLE_TRUCK: DefaultBoxDetectionLabel.VEHICLE,
            NuScenesBoxDetectionLabel.VEHICLE_BUS_BENDY: DefaultBoxDetectionLabel.VEHICLE,
            NuScenesBoxDetectionLabel.VEHICLE_BUS_RIGID: DefaultBoxDetectionLabel.VEHICLE,
            NuScenesBoxDetectionLabel.VEHICLE_CONSTRUCTION: DefaultBoxDetectionLabel.VEHICLE,
            NuScenesBoxDetectionLabel.VEHICLE_EMERGENCY_AMBULANCE: DefaultBoxDetectionLabel.VEHICLE,
            NuScenesBoxDetectionLabel.VEHICLE_EMERGENCY_POLICE: DefaultBoxDetectionLabel.VEHICLE,
            NuScenesBoxDetectionLabel.VEHICLE_TRAILER: DefaultBoxDetectionLabel.VEHICLE,
            NuScenesBoxDetectionLabel.VEHICLE_BICYCLE: DefaultBoxDetectionLabel.BICYCLE,
            NuScenesBoxDetectionLabel.VEHICLE_MOTORCYCLE: DefaultBoxDetectionLabel.BICYCLE,
            NuScenesBoxDetectionLabel.HUMAN_PEDESTRIAN_ADULT: DefaultBoxDetectionLabel.PERSON,
            NuScenesBoxDetectionLabel.HUMAN_PEDESTRIAN_CHILD: DefaultBoxDetectionLabel.PERSON,
            NuScenesBoxDetectionLabel.HUMAN_PEDESTRIAN_CONSTRUCTION_WORKER: DefaultBoxDetectionLabel.PERSON,
            NuScenesBoxDetectionLabel.HUMAN_PEDESTRIAN_PERSONAL_MOBILITY: DefaultBoxDetectionLabel.PERSON,
            NuScenesBoxDetectionLabel.HUMAN_PEDESTRIAN_POLICE_OFFICER: DefaultBoxDetectionLabel.PERSON,
            NuScenesBoxDetectionLabel.HUMAN_PEDESTRIAN_STROLLER: DefaultBoxDetectionLabel.PERSON,
            NuScenesBoxDetectionLabel.HUMAN_PEDESTRIAN_WHEELCHAIR: DefaultBoxDetectionLabel.PERSON,
            NuScenesBoxDetectionLabel.MOVABLE_OBJECT_TRAFFICCONE: DefaultBoxDetectionLabel.TRAFFIC_CONE,
            NuScenesBoxDetectionLabel.MOVABLE_OBJECT_BARRIER: DefaultBoxDetectionLabel.BARRIER,
            NuScenesBoxDetectionLabel.MOVABLE_OBJECT_PUSHABLE_PULLABLE: DefaultBoxDetectionLabel.GENERIC_OBJECT,
            NuScenesBoxDetectionLabel.MOVABLE_OBJECT_DEBRIS: DefaultBoxDetectionLabel.GENERIC_OBJECT,
            NuScenesBoxDetectionLabel.STATIC_OBJECT_BICYCLE_RACK: DefaultBoxDetectionLabel.GENERIC_OBJECT,
            NuScenesBoxDetectionLabel.ANIMAL: DefaultBoxDetectionLabel.ANIMAL,
        }
        return mapping[self]


@register_box_detection_label
class PandasetBoxDetectionLabel(BoxDetectionLabel):
    """
    Semantic labels for Pandaset bounding box detections, see [1]_

    References
    ----------
    .. [1] https://github.com/scaleapi/pandaset-devkit/blob/master/docs/annotation_instructions_cuboids.pdf
    """

    ANIMALS_BIRD = 0
    ANIMALS_OTHER = 1
    BICYCLE = 2
    BUS = 3
    CAR = 4
    CONES = 5
    CONSTRUCTION_SIGNS = 6
    EMERGENCY_VEHICLE = 7
    MEDIUM_SIZED_TRUCK = 8
    MOTORCYCLE = 9
    MOTORIZED_SCOOTER = 10
    OTHER_VEHICLE_CONSTRUCTION_VEHICLE = 11
    OTHER_VEHICLE_PEDICAB = 12
    OTHER_VEHICLE_UNCOMMON = 13
    PEDESTRIAN = 14
    PEDESTRIAN_WITH_OBJECT = 15
    PERSONAL_MOBILITY_DEVICE = 16
    PICKUP_TRUCK = 17
    PYLONS = 18
    ROAD_BARRIERS = 19
    ROLLING_CONTAINERS = 20
    SEMI_TRUCK = 21
    SIGNS = 22
    TEMPORARY_CONSTRUCTION_BARRIERS = 23
    TOWED_OBJECT = 24
    TRAIN = 25
    TRAM_SUBWAY = 26

    def to_default(self) -> DefaultBoxDetectionLabel:
        """Inherited, see superclass."""
        mapping = {
            PandasetBoxDetectionLabel.ANIMALS_BIRD: DefaultBoxDetectionLabel.ANIMAL,
            PandasetBoxDetectionLabel.ANIMALS_OTHER: DefaultBoxDetectionLabel.ANIMAL,
            PandasetBoxDetectionLabel.BICYCLE: DefaultBoxDetectionLabel.BICYCLE,
            PandasetBoxDetectionLabel.BUS: DefaultBoxDetectionLabel.VEHICLE,
            PandasetBoxDetectionLabel.CAR: DefaultBoxDetectionLabel.VEHICLE,
            PandasetBoxDetectionLabel.CONES: DefaultBoxDetectionLabel.TRAFFIC_CONE,
            PandasetBoxDetectionLabel.CONSTRUCTION_SIGNS: DefaultBoxDetectionLabel.TRAFFIC_SIGN,
            PandasetBoxDetectionLabel.EMERGENCY_VEHICLE: DefaultBoxDetectionLabel.VEHICLE,
            PandasetBoxDetectionLabel.MEDIUM_SIZED_TRUCK: DefaultBoxDetectionLabel.VEHICLE,
            PandasetBoxDetectionLabel.MOTORCYCLE: DefaultBoxDetectionLabel.BICYCLE,
            PandasetBoxDetectionLabel.MOTORIZED_SCOOTER: DefaultBoxDetectionLabel.BICYCLE,
            PandasetBoxDetectionLabel.OTHER_VEHICLE_CONSTRUCTION_VEHICLE: DefaultBoxDetectionLabel.VEHICLE,
            PandasetBoxDetectionLabel.OTHER_VEHICLE_PEDICAB: DefaultBoxDetectionLabel.VEHICLE,
            PandasetBoxDetectionLabel.OTHER_VEHICLE_UNCOMMON: DefaultBoxDetectionLabel.VEHICLE,
            PandasetBoxDetectionLabel.PEDESTRIAN: DefaultBoxDetectionLabel.PERSON,
            PandasetBoxDetectionLabel.PEDESTRIAN_WITH_OBJECT: DefaultBoxDetectionLabel.PERSON,
            PandasetBoxDetectionLabel.PERSONAL_MOBILITY_DEVICE: DefaultBoxDetectionLabel.BICYCLE,
            PandasetBoxDetectionLabel.PICKUP_TRUCK: DefaultBoxDetectionLabel.VEHICLE,
            PandasetBoxDetectionLabel.PYLONS: DefaultBoxDetectionLabel.TRAFFIC_CONE,
            PandasetBoxDetectionLabel.ROAD_BARRIERS: DefaultBoxDetectionLabel.BARRIER,
            PandasetBoxDetectionLabel.ROLLING_CONTAINERS: DefaultBoxDetectionLabel.GENERIC_OBJECT,
            PandasetBoxDetectionLabel.SEMI_TRUCK: DefaultBoxDetectionLabel.VEHICLE,
            PandasetBoxDetectionLabel.SIGNS: DefaultBoxDetectionLabel.TRAFFIC_SIGN,
            PandasetBoxDetectionLabel.TEMPORARY_CONSTRUCTION_BARRIERS: DefaultBoxDetectionLabel.BARRIER,
            PandasetBoxDetectionLabel.TOWED_OBJECT: DefaultBoxDetectionLabel.VEHICLE,
            PandasetBoxDetectionLabel.TRAIN: DefaultBoxDetectionLabel.TRAIN,  # TODO: Adjust default types
            PandasetBoxDetectionLabel.TRAM_SUBWAY: DefaultBoxDetectionLabel.TRAIN,  # TODO: Adjust default types
        }
        return mapping[self]


@register_box_detection_label
class WODPerceptionBoxDetectionLabel(BoxDetectionLabel):
    """
    Semantic labels if bounding box detections in the WOD-Perception dataset, see [1]_ [2]_.

    References
    ----------
    .. [1] https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/labeling_specifications.md
    .. [2] https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/label.proto#L63-L69
    """

    TYPE_UNKNOWN = 0
    TYPE_VEHICLE = 1
    TYPE_PEDESTRIAN = 2
    TYPE_SIGN = 3
    TYPE_CYCLIST = 4

    def to_default(self) -> DefaultBoxDetectionLabel:
        """Inherited, see superclass."""
        mapping = {
            WODPerceptionBoxDetectionLabel.TYPE_UNKNOWN: DefaultBoxDetectionLabel.GENERIC_OBJECT,
            WODPerceptionBoxDetectionLabel.TYPE_VEHICLE: DefaultBoxDetectionLabel.VEHICLE,
            WODPerceptionBoxDetectionLabel.TYPE_PEDESTRIAN: DefaultBoxDetectionLabel.PERSON,
            WODPerceptionBoxDetectionLabel.TYPE_SIGN: DefaultBoxDetectionLabel.TRAFFIC_SIGN,
            WODPerceptionBoxDetectionLabel.TYPE_CYCLIST: DefaultBoxDetectionLabel.BICYCLE,
        }
        return mapping[self]


@register_box_detection_label
class WODMotionBoxDetectionLabel(BoxDetectionLabel):
    """
    Semantic labels if bounding box detections in the WOD-Motion dataset, see [1]_.

    References
    ----------
    .. [1] https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/scenario.proto#L56-L62
    """

    TYPE_UNSET = 0
    TYPE_VEHICLE = 1
    TYPE_PEDESTRIAN = 2
    TYPE_CYCLIST = 3
    TYPE_OTHER = 4

    def to_default(self) -> DefaultBoxDetectionLabel:
        """Inherited, see superclass."""
        mapping = {
            WODMotionBoxDetectionLabel.TYPE_UNSET: DefaultBoxDetectionLabel.GENERIC_OBJECT,
            WODMotionBoxDetectionLabel.TYPE_VEHICLE: DefaultBoxDetectionLabel.VEHICLE,
            WODMotionBoxDetectionLabel.TYPE_PEDESTRIAN: DefaultBoxDetectionLabel.PERSON,
            WODMotionBoxDetectionLabel.TYPE_OTHER: DefaultBoxDetectionLabel.GENERIC_OBJECT,
            WODMotionBoxDetectionLabel.TYPE_CYCLIST: DefaultBoxDetectionLabel.BICYCLE,
        }
        return mapping[self]
