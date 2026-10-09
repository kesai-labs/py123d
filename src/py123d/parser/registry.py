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
            AV2SensorBoxDetectionLabel.BICYCLE: DefaultBoxDetectionLabel.TWO_WHEELER,
            AV2SensorBoxDetectionLabel.BICYCLIST: DefaultBoxDetectionLabel.PERSON,
            AV2SensorBoxDetectionLabel.BOLLARD: DefaultBoxDetectionLabel.BARRIER,
            AV2SensorBoxDetectionLabel.BOX_TRUCK: DefaultBoxDetectionLabel.VEHICLE,
            AV2SensorBoxDetectionLabel.BUS: DefaultBoxDetectionLabel.VEHICLE,
            AV2SensorBoxDetectionLabel.CONSTRUCTION_BARREL: DefaultBoxDetectionLabel.TRAFFIC_CONE,
            AV2SensorBoxDetectionLabel.CONSTRUCTION_CONE: DefaultBoxDetectionLabel.TRAFFIC_CONE,
            AV2SensorBoxDetectionLabel.DOG: DefaultBoxDetectionLabel.ANIMAL,
            AV2SensorBoxDetectionLabel.LARGE_VEHICLE: DefaultBoxDetectionLabel.VEHICLE,
            AV2SensorBoxDetectionLabel.MESSAGE_BOARD_TRAILER: DefaultBoxDetectionLabel.VEHICLE,
            AV2SensorBoxDetectionLabel.MOBILE_PEDESTRIAN_CROSSING_SIGN: DefaultBoxDetectionLabel.TRAFFIC_SIGN,
            AV2SensorBoxDetectionLabel.MOTORCYCLE: DefaultBoxDetectionLabel.TWO_WHEELER,
            AV2SensorBoxDetectionLabel.MOTORCYCLIST: DefaultBoxDetectionLabel.PERSON,
            AV2SensorBoxDetectionLabel.OFFICIAL_SIGNALER: DefaultBoxDetectionLabel.PERSON,
            AV2SensorBoxDetectionLabel.PEDESTRIAN: DefaultBoxDetectionLabel.PERSON,
            AV2SensorBoxDetectionLabel.RAILED_VEHICLE: DefaultBoxDetectionLabel.TRAIN,
            AV2SensorBoxDetectionLabel.REGULAR_VEHICLE: DefaultBoxDetectionLabel.VEHICLE,
            AV2SensorBoxDetectionLabel.SCHOOL_BUS: DefaultBoxDetectionLabel.VEHICLE,
            AV2SensorBoxDetectionLabel.SIGN: DefaultBoxDetectionLabel.TRAFFIC_SIGN,
            AV2SensorBoxDetectionLabel.STOP_SIGN: DefaultBoxDetectionLabel.TRAFFIC_SIGN,
            AV2SensorBoxDetectionLabel.STROLLER: DefaultBoxDetectionLabel.OTHER,
            AV2SensorBoxDetectionLabel.TRAFFIC_LIGHT_TRAILER: DefaultBoxDetectionLabel.VEHICLE,
            AV2SensorBoxDetectionLabel.TRUCK: DefaultBoxDetectionLabel.VEHICLE,
            AV2SensorBoxDetectionLabel.TRUCK_CAB: DefaultBoxDetectionLabel.VEHICLE,
            AV2SensorBoxDetectionLabel.VEHICULAR_TRAILER: DefaultBoxDetectionLabel.VEHICLE,
            # NOTE @DanielDauner: Separate classes for wheelchairs and wheeled rider,
            # we thus map the wheelchair to OTHER and the wheeled rider to PERSON.
            AV2SensorBoxDetectionLabel.WHEELCHAIR: DefaultBoxDetectionLabel.OTHER,
            AV2SensorBoxDetectionLabel.WHEELED_DEVICE: DefaultBoxDetectionLabel.OTHER,
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
            KITTI360BoxDetectionLabel.BICYCLE: DefaultBoxDetectionLabel.TWO_WHEELER,
            KITTI360BoxDetectionLabel.BOX: DefaultBoxDetectionLabel.GENERIC_OBJECT,
            KITTI360BoxDetectionLabel.BUS: DefaultBoxDetectionLabel.VEHICLE,
            KITTI360BoxDetectionLabel.CAR: DefaultBoxDetectionLabel.VEHICLE,
            KITTI360BoxDetectionLabel.CARAVAN: DefaultBoxDetectionLabel.VEHICLE,
            KITTI360BoxDetectionLabel.LAMP: DefaultBoxDetectionLabel.GENERIC_OBJECT,
            KITTI360BoxDetectionLabel.MOTORCYCLE: DefaultBoxDetectionLabel.TWO_WHEELER,
            KITTI360BoxDetectionLabel.PERSON: DefaultBoxDetectionLabel.PERSON,
            KITTI360BoxDetectionLabel.POLE: DefaultBoxDetectionLabel.GENERIC_OBJECT,
            KITTI360BoxDetectionLabel.RIDER: DefaultBoxDetectionLabel.PERSON,
            KITTI360BoxDetectionLabel.SMALLPOLE: DefaultBoxDetectionLabel.GENERIC_OBJECT,
            KITTI360BoxDetectionLabel.STOP: DefaultBoxDetectionLabel.TRAFFIC_SIGN,
            KITTI360BoxDetectionLabel.TRAFFIC_LIGHT: DefaultBoxDetectionLabel.TRAFFIC_LIGHT,
            KITTI360BoxDetectionLabel.TRAFFIC_SIGN: DefaultBoxDetectionLabel.TRAFFIC_SIGN,
            KITTI360BoxDetectionLabel.TRAILER: DefaultBoxDetectionLabel.VEHICLE,
            KITTI360BoxDetectionLabel.TRAIN: DefaultBoxDetectionLabel.TRAIN,
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
            NuPlanBoxDetectionLabel.BICYCLE: DefaultBoxDetectionLabel.TWO_WHEELER,
            NuPlanBoxDetectionLabel.PEDESTRIAN: DefaultBoxDetectionLabel.PERSON,
            NuPlanBoxDetectionLabel.TRAFFIC_CONE: DefaultBoxDetectionLabel.TRAFFIC_CONE,
            NuPlanBoxDetectionLabel.BARRIER: DefaultBoxDetectionLabel.BARRIER,
            NuPlanBoxDetectionLabel.CZONE_SIGN: DefaultBoxDetectionLabel.TRAFFIC_SIGN,
            NuPlanBoxDetectionLabel.GENERIC_OBJECT: DefaultBoxDetectionLabel.GENERIC_OBJECT,
        }
        return mapping[self]


@register_box_detection_label
class NureasoningBoxDetectionLabel(BoxDetectionLabel):
    """Semantic labels for nuReasoning bounding box detections.

    nuReasoning does not publish an object taxonomy. The labels are the category strings observed in
    the released annotations, plus the ones the devkit lists. Categories outside this list fall back
    to :attr:`OTHER_OTHER` during conversion.
    """

    VEHICLE_CAR = 0
    """Cars and other four-or-more wheeled vehicles."""

    VEHICLE_PERSONAL_MOBILITY_BYCICLE = 1
    """Bicycles, motorcycles and other personal mobility devices."""

    HUMAN = 2
    """Humans / vulnerable road users."""

    OTHER_TRAFFICCONE = 3
    """Cones temporarily placed to control the flow of traffic."""

    OTHER_TEMPORARY_TRAFFICSIGN = 4
    """Temporary traffic signs (e.g. construction-zone signage)."""

    OTHER_OTHER = 5
    """Catch-all for uncategorized / miscellaneous objects."""

    VEHICLE_DOOR = 6
    """Vehicle doors (e.g. an opened car door)."""

    VEHICLE_TRUCK = 7
    """Trucks."""

    VEHICLE_BUS = 8
    """Buses."""

    VEHICLE_MOTORCYCLE = 9
    """Motorcycles."""

    VEHICLE_BICYCLE = 10
    """Bicycles."""

    HUMAN_PEDESTRIAN = 11
    """Pedestrians."""

    CONSTRUCTION_TRAFFIC_CONE = 12
    """Construction-zone traffic cones."""

    CONSTRUCTION_ZONE_AREA = 13
    """Extent of a construction zone. An area box (often 100 m or longer), not a physical object."""

    VEHICLE_CONSTRUCTION = 14
    """Construction vehicles."""

    VEHICLE_EMERGENCY = 15
    """Emergency vehicles."""

    VEHICLE_TRAILER = 16
    """Trailers."""

    ANIMAL = 17
    """Animals."""

    def to_default(self) -> DefaultBoxDetectionLabel:
        """Inherited, see superclass."""
        mapping = {
            NureasoningBoxDetectionLabel.VEHICLE_CAR: DefaultBoxDetectionLabel.VEHICLE,
            NureasoningBoxDetectionLabel.VEHICLE_PERSONAL_MOBILITY_BYCICLE: DefaultBoxDetectionLabel.TWO_WHEELER,
            NureasoningBoxDetectionLabel.HUMAN: DefaultBoxDetectionLabel.PERSON,
            NureasoningBoxDetectionLabel.OTHER_TRAFFICCONE: DefaultBoxDetectionLabel.TRAFFIC_CONE,
            NureasoningBoxDetectionLabel.OTHER_TEMPORARY_TRAFFICSIGN: DefaultBoxDetectionLabel.TRAFFIC_SIGN,
            NureasoningBoxDetectionLabel.OTHER_OTHER: DefaultBoxDetectionLabel.GENERIC_OBJECT,
            NureasoningBoxDetectionLabel.VEHICLE_DOOR: DefaultBoxDetectionLabel.GENERIC_OBJECT,
            NureasoningBoxDetectionLabel.VEHICLE_TRUCK: DefaultBoxDetectionLabel.VEHICLE,
            NureasoningBoxDetectionLabel.VEHICLE_BUS: DefaultBoxDetectionLabel.VEHICLE,
            NureasoningBoxDetectionLabel.VEHICLE_MOTORCYCLE: DefaultBoxDetectionLabel.TWO_WHEELER,
            NureasoningBoxDetectionLabel.VEHICLE_BICYCLE: DefaultBoxDetectionLabel.TWO_WHEELER,
            NureasoningBoxDetectionLabel.HUMAN_PEDESTRIAN: DefaultBoxDetectionLabel.PERSON,
            NureasoningBoxDetectionLabel.CONSTRUCTION_TRAFFIC_CONE: DefaultBoxDetectionLabel.TRAFFIC_CONE,
            NureasoningBoxDetectionLabel.CONSTRUCTION_ZONE_AREA: DefaultBoxDetectionLabel.OTHER,
            NureasoningBoxDetectionLabel.VEHICLE_CONSTRUCTION: DefaultBoxDetectionLabel.VEHICLE,
            NureasoningBoxDetectionLabel.VEHICLE_EMERGENCY: DefaultBoxDetectionLabel.VEHICLE,
            NureasoningBoxDetectionLabel.VEHICLE_TRAILER: DefaultBoxDetectionLabel.VEHICLE,
            NureasoningBoxDetectionLabel.ANIMAL: DefaultBoxDetectionLabel.ANIMAL,
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
            NuScenesBoxDetectionLabel.VEHICLE_BICYCLE: DefaultBoxDetectionLabel.TWO_WHEELER,
            NuScenesBoxDetectionLabel.VEHICLE_MOTORCYCLE: DefaultBoxDetectionLabel.TWO_WHEELER,
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
            PandasetBoxDetectionLabel.BICYCLE: DefaultBoxDetectionLabel.TWO_WHEELER,
            PandasetBoxDetectionLabel.BUS: DefaultBoxDetectionLabel.VEHICLE,
            PandasetBoxDetectionLabel.CAR: DefaultBoxDetectionLabel.VEHICLE,
            PandasetBoxDetectionLabel.CONES: DefaultBoxDetectionLabel.TRAFFIC_CONE,
            PandasetBoxDetectionLabel.CONSTRUCTION_SIGNS: DefaultBoxDetectionLabel.TRAFFIC_SIGN,
            PandasetBoxDetectionLabel.EMERGENCY_VEHICLE: DefaultBoxDetectionLabel.VEHICLE,
            PandasetBoxDetectionLabel.MEDIUM_SIZED_TRUCK: DefaultBoxDetectionLabel.VEHICLE,
            PandasetBoxDetectionLabel.MOTORCYCLE: DefaultBoxDetectionLabel.TWO_WHEELER,
            PandasetBoxDetectionLabel.MOTORIZED_SCOOTER: DefaultBoxDetectionLabel.TWO_WHEELER,
            PandasetBoxDetectionLabel.OTHER_VEHICLE_CONSTRUCTION_VEHICLE: DefaultBoxDetectionLabel.VEHICLE,
            PandasetBoxDetectionLabel.OTHER_VEHICLE_PEDICAB: DefaultBoxDetectionLabel.VEHICLE,
            PandasetBoxDetectionLabel.OTHER_VEHICLE_UNCOMMON: DefaultBoxDetectionLabel.VEHICLE,
            PandasetBoxDetectionLabel.PEDESTRIAN: DefaultBoxDetectionLabel.PERSON,
            PandasetBoxDetectionLabel.PEDESTRIAN_WITH_OBJECT: DefaultBoxDetectionLabel.PERSON,
            PandasetBoxDetectionLabel.PERSONAL_MOBILITY_DEVICE: DefaultBoxDetectionLabel.OTHER,
            PandasetBoxDetectionLabel.PICKUP_TRUCK: DefaultBoxDetectionLabel.VEHICLE,
            PandasetBoxDetectionLabel.PYLONS: DefaultBoxDetectionLabel.TRAFFIC_CONE,
            PandasetBoxDetectionLabel.ROAD_BARRIERS: DefaultBoxDetectionLabel.BARRIER,
            PandasetBoxDetectionLabel.ROLLING_CONTAINERS: DefaultBoxDetectionLabel.GENERIC_OBJECT,
            PandasetBoxDetectionLabel.SEMI_TRUCK: DefaultBoxDetectionLabel.VEHICLE,
            PandasetBoxDetectionLabel.SIGNS: DefaultBoxDetectionLabel.TRAFFIC_SIGN,
            PandasetBoxDetectionLabel.TEMPORARY_CONSTRUCTION_BARRIERS: DefaultBoxDetectionLabel.BARRIER,
            PandasetBoxDetectionLabel.TOWED_OBJECT: DefaultBoxDetectionLabel.VEHICLE,
            PandasetBoxDetectionLabel.TRAIN: DefaultBoxDetectionLabel.TRAIN,
            PandasetBoxDetectionLabel.TRAM_SUBWAY: DefaultBoxDetectionLabel.TRAIN,
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
            WODPerceptionBoxDetectionLabel.TYPE_UNKNOWN: DefaultBoxDetectionLabel.OTHER,
            WODPerceptionBoxDetectionLabel.TYPE_VEHICLE: DefaultBoxDetectionLabel.VEHICLE,
            WODPerceptionBoxDetectionLabel.TYPE_PEDESTRIAN: DefaultBoxDetectionLabel.PERSON,
            WODPerceptionBoxDetectionLabel.TYPE_SIGN: DefaultBoxDetectionLabel.TRAFFIC_SIGN,
            WODPerceptionBoxDetectionLabel.TYPE_CYCLIST: DefaultBoxDetectionLabel.TWO_WHEELER,
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
            WODMotionBoxDetectionLabel.TYPE_UNSET: DefaultBoxDetectionLabel.OTHER,
            WODMotionBoxDetectionLabel.TYPE_VEHICLE: DefaultBoxDetectionLabel.VEHICLE,
            WODMotionBoxDetectionLabel.TYPE_PEDESTRIAN: DefaultBoxDetectionLabel.PERSON,
            WODMotionBoxDetectionLabel.TYPE_OTHER: DefaultBoxDetectionLabel.OTHER,
            WODMotionBoxDetectionLabel.TYPE_CYCLIST: DefaultBoxDetectionLabel.TWO_WHEELER,
        }
        return mapping[self]


@register_box_detection_label
class PhysicalAIAVBoxDetectionLabel(BoxDetectionLabel):
    """Semantic labels for Physical AI AV dataset obstacle detections (auto-labeled).

    Matches the 12 dynamic object classes documented in the NVlabs physical_ai_av wiki:
    https://github.com/NVlabs/physical_ai_av/wiki/5.-Machine-Labels-(Coming-Soon)
    """

    AUTOMOBILE = 0
    PERSON = 1
    BUS = 2
    HEAVY_TRUCK = 3
    OTHER_VEHICLE = 4
    PROTRUDING_OBJECT = 5
    RIDER = 6
    STROLLER = 7
    TRAILER = 8
    ANIMAL = 9
    TRAIN_OR_TRAM_CAR = 10
    TROLLEY_BUS = 11

    def to_default(self) -> DefaultBoxDetectionLabel:
        """Inherited, see superclass."""
        mapping = {
            PhysicalAIAVBoxDetectionLabel.AUTOMOBILE: DefaultBoxDetectionLabel.VEHICLE,
            PhysicalAIAVBoxDetectionLabel.PERSON: DefaultBoxDetectionLabel.PERSON,
            PhysicalAIAVBoxDetectionLabel.BUS: DefaultBoxDetectionLabel.VEHICLE,
            PhysicalAIAVBoxDetectionLabel.HEAVY_TRUCK: DefaultBoxDetectionLabel.VEHICLE,
            PhysicalAIAVBoxDetectionLabel.OTHER_VEHICLE: DefaultBoxDetectionLabel.VEHICLE,
            PhysicalAIAVBoxDetectionLabel.PROTRUDING_OBJECT: DefaultBoxDetectionLabel.GENERIC_OBJECT,
            # NOTE @DanielDauner: The rider class includes the vehicle (e.g. bicycle),
            # we thus map to TWO_WHEELER instead of PERSON.
            PhysicalAIAVBoxDetectionLabel.RIDER: DefaultBoxDetectionLabel.TWO_WHEELER,
            PhysicalAIAVBoxDetectionLabel.STROLLER: DefaultBoxDetectionLabel.OTHER,
            PhysicalAIAVBoxDetectionLabel.TRAILER: DefaultBoxDetectionLabel.VEHICLE,
            PhysicalAIAVBoxDetectionLabel.ANIMAL: DefaultBoxDetectionLabel.ANIMAL,
            PhysicalAIAVBoxDetectionLabel.TRAIN_OR_TRAM_CAR: DefaultBoxDetectionLabel.TRAIN,
            PhysicalAIAVBoxDetectionLabel.TROLLEY_BUS: DefaultBoxDetectionLabel.VEHICLE,
        }
        return mapping[self]


@register_box_detection_label
class GriffinBoxDetectionLabel(BoxDetectionLabel):
    """Griffin (aerial-ground cooperative) dataset annotation categories.

    Griffin annotates dynamic traffic participants with CARLA-derived semantic
    types. The official benchmark collapses these into three evaluation classes
    (``car``, ``bicycle``, ``pedestrian``); we keep the native granularity here
    and map down to :class:`DefaultBoxDetectionLabel` in :meth:`to_default`.

    Category collapsing follows ``obj_type_mapping`` in the official converter.
    See Wang et al., "Griffin: Aerial-Ground Cooperative Detection and Tracking
    Dataset and Benchmark", arXiv preprint 2503.06983 (2025), and the
    official toolkit (https://github.com/wang-jh18-SVM/Griffin).
    """

    PEDESTRIAN = 0
    CAR = 1
    TRUCK = 2
    BUS = 3
    MOTORCYCLE = 4
    BICYCLE = 5

    def to_default(self) -> DefaultBoxDetectionLabel:
        """Inherited, see superclass."""
        mapping = {
            GriffinBoxDetectionLabel.PEDESTRIAN: DefaultBoxDetectionLabel.PERSON,
            GriffinBoxDetectionLabel.CAR: DefaultBoxDetectionLabel.VEHICLE,
            GriffinBoxDetectionLabel.TRUCK: DefaultBoxDetectionLabel.VEHICLE,
            GriffinBoxDetectionLabel.BUS: DefaultBoxDetectionLabel.VEHICLE,
            GriffinBoxDetectionLabel.MOTORCYCLE: DefaultBoxDetectionLabel.TWO_WHEELER,
            GriffinBoxDetectionLabel.BICYCLE: DefaultBoxDetectionLabel.TWO_WHEELER,
        }
        return mapping[self]
