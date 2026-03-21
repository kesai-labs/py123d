from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

from py123d.datatypes import (
    BoxDetectionAttributes,
    BoxDetectionSE3,
    BoxDetectionsSE3,
    CameraID,
    EgoStateSE3,
    LidarID,
    LidarMetadata,
    LogMetadata,
    PinholeCameraMetadata,
    PinholeDistortion,
    PinholeIntrinsics,
    Timestamp,
)
from py123d.datatypes.detections.box_detections_metadata import BoxDetectionsSE3Metadata
from py123d.datatypes.sensors.lidar import LidarMergedMetadata
from py123d.datatypes.vehicle_state.ego_state_metadata import EgoStateSE3Metadata
from py123d.geometry import (
    BoundingBoxSE3,
    BoundingBoxSE3Index,
    EulerAngles,
    EulerAnglesIndex,
    PoseSE3,
    PoseSE3Index,
    Vector3D,
    Vector3DIndex,
)
from py123d.geometry.transform import rel_to_abs_se3, rel_to_abs_se3_array
from py123d.geometry.utils.constants import DEFAULT_PITCH, DEFAULT_ROLL
from py123d.geometry.utils.rotation_utils import (
    get_euler_array_from_quaternion_array,
    get_quaternion_array_from_euler_array,
)
from py123d.parser.base_dataset_parser import (
    BaseDatasetParser,
    BaseLogParser,
    BaseMapParser,
    ModalitiesSync,
    ParsedCamera,
    ParsedLidar,
)
from py123d.parser.registry import WODPerceptionBoxDetectionLabel
from py123d.parser.utils.sensor_utils.camera_conventions import CameraConvention, convert_camera_convention
from py123d.parser.wod.utils.wod_constants import (
    WOD_PERCEPTION_AVAILABLE_SPLITS,
    WOD_PERCEPTION_CAMERA_IDS,
    WOD_PERCEPTION_LIDAR_IDS,
)
from py123d.parser.wod.wod_map_parser import WODMapParser

if TYPE_CHECKING:
    from py123d.parser.wod.waymo_open_dataset.protos import dataset_pb2

logger = logging.getLogger(__name__)

# WOD Perception lidar operates at 10Hz → one full 360° sweep takes ~100ms.
# frame.timestamp_micros marks the START of the TOP lidar spin.
# frame.pose corresponds to the vehicle pose near the MIDDLE of the lidar spin (~50ms later).
# [1] https://github.com/waymo-research/waymo-open-dataset/issues/464
# [2] https://github.com/waymo-research/waymo-open-dataset/issues/70#issuecomment-552548486
WOD_PERCEPTION_LIDAR_SWEEP_DURATION_US = 100_000
WOD_PERCEPTION_LIDAR_HALF_SWEEP_US = WOD_PERCEPTION_LIDAR_SWEEP_DURATION_US // 2

# NOTE: These parameters are estimates based on the vehicle model used in the WOD Perception dataset.
# The vehicle should be the same (or a similar) vehicle model to nuPlan and PandaSet [1].
# [1] https://en.wikipedia.org/wiki/Chrysler_Pacifica_(minivan)
WOD_PERCEPTION_EGO_STATE_SE3_METADATA = EgoStateSE3Metadata(
    vehicle_name="wod-perception_chrysler_pacifica",
    width=2.297,
    length=5.176,
    height=1.777,
    wheel_base=3.089,
    center_to_imu_se3=PoseSE3(x=1.461, y=0.0, z=1.777 / 2, qw=1.0, qx=0.0, qy=0.0, qz=0.0),
    rear_axle_to_imu_se3=PoseSE3.identity(),
)

WOD_PERCEPTION_BOX_DETECTIONS_SE3_METADATA = BoxDetectionsSE3Metadata(
    box_detection_label_class=WODPerceptionBoxDetectionLabel,
)


def _lazy_import_tf_and_pb2():
    """Lazy import of tensorflow and dataset_pb2 to avoid import errors at module load time."""
    import os

    from py123d.common.utils.dependencies import check_dependencies

    check_dependencies(modules=["tensorflow"], optional_name="waymo")
    import tensorflow as tf

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    tf.config.set_visible_devices(tf.config.list_physical_devices("CPU"))

    # Proto dependencies must be loaded in dependency order before dataset_pb2
    import importlib

    for _proto in ("vector_pb2", "keypoint_pb2", "label_pb2", "map_pb2"):
        importlib.import_module(f"py123d.parser.wod.waymo_open_dataset.protos.{_proto}")
    from py123d.parser.wod.waymo_open_dataset.protos import dataset_pb2

    return tf, dataset_pb2


class WODPerceptionParser(BaseDatasetParser):
    """Dataset parser for the Waymo Open Dataset - Perception."""

    def __init__(
        self,
        splits: List[str],
        wod_perception_data_root: Union[Path, str],
        zero_roll_pitch: bool,
        keep_polar_features: bool,
        add_map_pose_offset: bool,
        add_dummy_lane_groups: bool,
    ) -> None:
        """Initializes the :class:`WODPerceptionParser`.

        :param splits: List of splits to convert, e.g. ``["wod-perception_train", "wod-perception_val"]``.
        :param wod_perception_data_root: Path to the root directory of the WOD Perception dataset
        :param zero_roll_pitch: Whether to zero out roll and pitch angles in the vehicle pose
        :param keep_polar_features: Whether to keep polar features in the Lidar point clouds
        :param add_map_pose_offset: Whether to add a pose offset to the map
        :param add_dummy_lane_groups: Whether to add dummy lane groups. \
            If True, creates a lane group for each lane since WOD does not provide lane groups.
        """
        for split in splits:
            assert split in WOD_PERCEPTION_AVAILABLE_SPLITS, (
                f"Split {split} is not available. Available splits: {WOD_PERCEPTION_AVAILABLE_SPLITS}"
            )

        self._splits: List[str] = splits
        self._wod_perception_data_root: Path = Path(wod_perception_data_root)
        self._zero_roll_pitch: bool = zero_roll_pitch
        self._keep_polar_features: bool = keep_polar_features
        self._add_map_pose_offset: bool = add_map_pose_offset
        self._add_dummy_lane_groups: bool = add_dummy_lane_groups

        self._split_tf_record_pairs: List[Tuple[str, Path]] = self._collect_split_tf_record_pairs()

    def _collect_split_tf_record_pairs(self) -> List[Tuple[str, Path]]:
        """Helper to collect the pairings of the split names and the corresponding tf record file."""
        split_tf_record_pairs: List[Tuple[str, Path]] = []
        split_name_mapping: Dict[str, str] = {
            "wod-perception_train": "training",
            "wod-perception_val": "validation",
            "wod-perception_test": "testing",
        }

        for split in self._splits:
            assert split in split_name_mapping.keys()
            split_folder = self._wod_perception_data_root / split_name_mapping[split]
            source_log_paths = [log_file for log_file in split_folder.glob("*.tfrecord")]
            for source_log_path in source_log_paths:
                split_tf_record_pairs.append((split, source_log_path))

        return split_tf_record_pairs

    def get_map_parsers(self) -> List[BaseMapParser]:
        """Inherited, see superclass."""
        map_parsers: List[BaseMapParser] = []
        for split, source_tf_record_path in self._split_tf_record_pairs:
            initial_frame = _get_initial_frame_from_tfrecord(source_tf_record_path)
            map_parsers.append(
                WODMapParser(
                    dataset="wod_perception",
                    split=split,
                    log_name=str(initial_frame.context.name),
                    source_tf_record_path=source_tf_record_path,
                    add_dummy_lane_groups=self._add_dummy_lane_groups,
                )
            )
        return map_parsers

    def get_log_parsers(self) -> List[BaseLogParser]:
        """Inherited, see superclass."""
        return [
            WODPerceptionLogParser(
                split=split,
                source_tf_record_path=source_tf_record_path,
                wod_perception_data_root=self._wod_perception_data_root,
                zero_roll_pitch=self._zero_roll_pitch,
                keep_polar_features=self._keep_polar_features,
                add_map_pose_offset=self._add_map_pose_offset,
            )
            for split, source_tf_record_path in self._split_tf_record_pairs
        ]


class WODPerceptionLogParser(BaseLogParser):
    """Lightweight, picklable handle to one WOD Perception log."""

    def __init__(
        self,
        split: str,
        source_tf_record_path: Path,
        wod_perception_data_root: Path,
        zero_roll_pitch: bool,
        keep_polar_features: bool,
        add_map_pose_offset: bool,
    ) -> None:
        self._split = split
        self._source_tf_record_path = source_tf_record_path
        self._wod_perception_data_root = wod_perception_data_root
        self._zero_roll_pitch = zero_roll_pitch
        self._keep_polar_features = keep_polar_features
        self._add_map_pose_offset = add_map_pose_offset

    def get_log_metadata(self) -> LogMetadata:
        """Inherited, see superclass."""
        initial_frame = _get_initial_frame_from_tfrecord(self._source_tf_record_path)
        return LogMetadata(
            dataset="wod_perception",
            split=self._split,
            log_name=str(initial_frame.context.name),
            location=str(initial_frame.context.stats.location),
        )

    def iter_modalities_sync(self) -> Iterator[ModalitiesSync]:
        """Inherited, see superclass."""
        initial_frame = _get_initial_frame_from_tfrecord(self._source_tf_record_path)
        pinhole_cameras_metadata = _get_wod_perception_camera_metadata(initial_frame)
        lidar_merged_metadata = _get_wod_perception_lidar_merged_metadata(initial_frame)

        tf, dataset_pb2 = _lazy_import_tf_and_pb2()
        dataset = tf.data.TFRecordDataset(self._source_tf_record_path, compression_type="")

        for frame_idx, data in enumerate(dataset):
            frame = dataset_pb2.Frame()
            frame.ParseFromString(data.numpy())

            map_pose_offset: Vector3D = Vector3D(0.0, 0.0, 0.0)
            if self._add_map_pose_offset:
                map_pose_offset = Vector3D(
                    x=frame.map_pose_offset.x,
                    y=frame.map_pose_offset.y,
                    z=frame.map_pose_offset.z,
                )

            # frame.timestamp_micros is the start of the lidar spin.
            # frame.pose is the vehicle pose near the middle of the spin (~50ms later).
            frame_timestamp = Timestamp.from_us(frame.timestamp_micros)
            ego_pose_timestamp = Timestamp.from_us(frame.timestamp_micros + WOD_PERCEPTION_LIDAR_HALF_SWEEP_US)

            ego_state_se3 = _extract_wod_perception_ego_state(
                frame, map_pose_offset, WOD_PERCEPTION_EGO_STATE_SE3_METADATA, timestamp=ego_pose_timestamp
            )
            box_detections_se3 = _extract_wod_perception_box_detections(
                frame,
                map_pose_offset,
                WOD_PERCEPTION_BOX_DETECTIONS_SE3_METADATA,
                self._zero_roll_pitch,
                ego_pose_timestamp,
            )
            parsed_pinhole_cameras = _extract_wod_perception_cameras(frame, pinhole_cameras_metadata, map_pose_offset)
            parsed_lidar = _extract_wod_perception_lidar(
                frame,
                frame_idx,
                self._source_tf_record_path,
                self._wod_perception_data_root,
                lidar_merged_metadata,
            )

            yield ModalitiesSync(
                timestamp=frame_timestamp,
                modalities=[
                    ego_state_se3,
                    box_detections_se3,
                    parsed_lidar,
                    *parsed_pinhole_cameras,
                ],
            )


def _get_initial_frame_from_tfrecord(tf_record_path: Path) -> dataset_pb2.Frame:
    """Helper to get the initial frame from a tf record file."""
    tf, dataset_pb2 = _lazy_import_tf_and_pb2()

    dataset = tf.data.TFRecordDataset(tf_record_path, compression_type="")
    for data in dataset:
        initial_frame = dataset_pb2.Frame()
        initial_frame.ParseFromString(data.numpy())
        break

    del dataset
    return initial_frame


# ------------------------------------------------------------------------------------------------------------------
# Metadata helpers
# ------------------------------------------------------------------------------------------------------------------


def _get_wod_perception_camera_metadata(
    initial_frame: dataset_pb2.Frame,
) -> Dict[CameraID, PinholeCameraMetadata]:
    """Get the WOD Perception camera metadata from the initial frame."""
    camera_metadata_dict: Dict[CameraID, PinholeCameraMetadata] = {}
    for calibration in initial_frame.context.camera_calibrations:
        camera_type = WOD_PERCEPTION_CAMERA_IDS[calibration.name]

        # Intrinsic & distortion parameters
        # https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/dataset.proto#L96
        # https://github.com/waymo-research/waymo-open-dataset/issues/834#issuecomment-2134995440
        fx, fy, cx, cy, k1, k2, p1, p2, k3 = calibration.intrinsic
        intrinsics = PinholeIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy)
        distortion = PinholeDistortion(k1=k1, k2=k2, p1=p1, p2=p2, k3=k3)

        # Static extrinsic parameters (from calibration)
        camera_to_imu_se3_matrix = np.array(calibration.extrinsic.transform, dtype=np.float64).reshape(4, 4)
        camera_to_imu_se3 = PoseSE3.from_transformation_matrix(camera_to_imu_se3_matrix)
        camera_to_imu_se3 = convert_camera_convention(
            camera_to_imu_se3,
            from_convention=CameraConvention.pXpZmY,
            to_convention=CameraConvention.pZmYpX,
        )

        if camera_type in WOD_PERCEPTION_CAMERA_IDS.values():
            camera_metadata_dict[camera_type] = PinholeCameraMetadata(
                camera_name=str(calibration.name),
                camera_id=camera_type,
                width=calibration.width,
                height=calibration.height,
                intrinsics=intrinsics,
                distortion=distortion,
                camera_to_imu_se3=camera_to_imu_se3,
            )

    return camera_metadata_dict


def _get_wod_perception_lidar_merged_metadata(
    initial_frame: dataset_pb2.Frame,
) -> LidarMergedMetadata:
    """Get the WOD Perception Lidar merged metadata from the initial frame.

    WOD provides individual calibrations for 5 lidars (TOP, FRONT, SIDE_LEFT, SIDE_RIGHT, REAR)
    but the point cloud data is stored as a single merged sweep per frame.
    """
    laser_metadatas: Dict[LidarID, LidarMetadata] = {}
    for laser_calibration in initial_frame.context.laser_calibrations:
        lidar_type = WOD_PERCEPTION_LIDAR_IDS[laser_calibration.name]
        extrinsic: Optional[PoseSE3] = None
        if laser_calibration.extrinsic:
            extrinsic_transform = np.array(laser_calibration.extrinsic.transform, dtype=np.float64).reshape(4, 4)
            extrinsic = PoseSE3.from_transformation_matrix(extrinsic_transform)

        laser_metadatas[lidar_type] = LidarMetadata(
            lidar_name=str(laser_calibration.name),
            lidar_id=lidar_type,
            lidar_to_imu_se3=extrinsic,
        )

    return LidarMergedMetadata(laser_metadatas)


# ------------------------------------------------------------------------------------------------------------------
# Sensor extraction helpers
# ------------------------------------------------------------------------------------------------------------------


def _get_ego_pose_se3(frame: dataset_pb2.Frame, map_pose_offset: Vector3D) -> PoseSE3:
    """Helper to get the ego pose SE3 from a WOD Perception frame."""
    ego_pose_matrix = np.array(frame.pose.transform, dtype=np.float64).reshape(4, 4)
    ego_pose_se3 = PoseSE3.from_transformation_matrix(ego_pose_matrix)
    ego_pose_se3.array[PoseSE3Index.XYZ] += map_pose_offset.array[Vector3DIndex.XYZ]
    return ego_pose_se3


def _extract_wod_perception_ego_state(
    frame: dataset_pb2.Frame, map_pose_offset: Vector3D, ego_metadata: EgoStateSE3Metadata, timestamp: Timestamp
) -> EgoStateSE3:
    """Extracts the ego state from a WOD Perception frame.

    ``frame.pose`` is the vehicle pose near the middle of the TOP lidar spin (~50ms after
    ``frame.timestamp_micros``). The caller should pass the mid-sweep timestamp.
    """
    imu_se3 = _get_ego_pose_se3(frame, map_pose_offset)
    return EgoStateSE3.from_imu(
        imu_se3=imu_se3,
        dynamic_state_se3=None,
        metadata=ego_metadata,
        timestamp=timestamp,
    )


def _extract_wod_perception_box_detections(
    frame: dataset_pb2.Frame,
    map_pose_offset: Vector3D,
    box_detections_metadata: BoxDetectionsSE3Metadata,
    zero_roll_pitch: bool = True,
    timestamp: Optional[Timestamp] = None,
) -> BoxDetectionsSE3:
    """Extracts the box detections from a WOD Perception frame.

    ``laser_labels`` are defined w.r.t. the ``frame.pose`` coordinate system, which
    corresponds to the mid-sweep vehicle pose. The caller should pass the mid-sweep timestamp.
    """
    ego_pose_se3 = _get_ego_pose_se3(frame, map_pose_offset)

    num_detections = len(frame.laser_labels)
    detections_state = np.zeros((num_detections, len(BoundingBoxSE3Index)), dtype=np.float64)
    detections_velocity = np.zeros((num_detections, len(Vector3DIndex)), dtype=np.float64)
    detections_types: List[WODPerceptionBoxDetectionLabel] = []
    detections_token: List[str] = []

    for detection_idx, detection in enumerate(frame.laser_labels):
        detection_quaternion = EulerAngles(
            roll=DEFAULT_ROLL,
            pitch=DEFAULT_PITCH,
            yaw=detection.box.heading,
        ).quaternion

        detections_state[detection_idx, BoundingBoxSE3Index.X] = detection.box.center_x
        detections_state[detection_idx, BoundingBoxSE3Index.Y] = detection.box.center_y
        detections_state[detection_idx, BoundingBoxSE3Index.Z] = detection.box.center_z
        detections_state[detection_idx, BoundingBoxSE3Index.QUATERNION] = detection_quaternion
        detections_state[detection_idx, BoundingBoxSE3Index.LENGTH] = detection.box.length
        detections_state[detection_idx, BoundingBoxSE3Index.WIDTH] = detection.box.width
        detections_state[detection_idx, BoundingBoxSE3Index.HEIGHT] = detection.box.height

        # TODO: check if velocity needs to be rotated
        detections_velocity[detection_idx] = Vector3D(
            x=detection.metadata.speed_x,
            y=detection.metadata.speed_y,
            z=detection.metadata.speed_z,
        ).array

        detections_types.append(WODPerceptionBoxDetectionLabel(detection.type))
        detections_token.append(str(detection.id))

    detections_state[:, BoundingBoxSE3Index.SE3] = rel_to_abs_se3_array(
        origin=ego_pose_se3, pose_se3_array=detections_state[:, BoundingBoxSE3Index.SE3]
    )
    if zero_roll_pitch:
        euler_array = get_euler_array_from_quaternion_array(detections_state[:, BoundingBoxSE3Index.QUATERNION])
        euler_array[..., EulerAnglesIndex.ROLL] = DEFAULT_ROLL
        euler_array[..., EulerAnglesIndex.PITCH] = DEFAULT_PITCH
        detections_state[..., BoundingBoxSE3Index.QUATERNION] = get_quaternion_array_from_euler_array(euler_array)

    box_detections: List[BoxDetectionSE3] = []
    for detection_idx in range(num_detections):
        box_detections.append(
            BoxDetectionSE3(
                attributes=BoxDetectionAttributes(
                    label=detections_types[detection_idx],
                    track_token=detections_token[detection_idx],
                ),
                bounding_box_se3=BoundingBoxSE3.from_array(detections_state[detection_idx]),
                velocity_3d=Vector3D.from_array(detections_velocity[detection_idx]),
            )
        )
    return BoxDetectionsSE3(box_detections=box_detections, timestamp=timestamp, metadata=box_detections_metadata)


def _extract_wod_perception_cameras(
    frame: dataset_pb2.Frame,
    camera_metadatas: Dict[CameraID, PinholeCameraMetadata],
    map_pose_offset: Vector3D,
) -> List[ParsedCamera]:
    """Extracts the camera data from a WOD Perception frame.

    Each camera has its own ``pose_timestamp`` (seconds since epoch) from the proto,
    which is the time of the ego pose when the camera triggers. The global camera pose
    is computed by composing the per-camera ego pose (``image_proto.pose``) with the
    static camera-to-vehicle extrinsic from calibration (already stored in
    ``camera_metadatas[...].camera_to_imu_se3`` with convention conversion applied).

    The ``map_pose_offset`` is applied to the camera ego pose translation to align
    camera global poses with map features, consistent with how ego and detection poses
    are offset (see :func:`_get_ego_pose_se3`). In the official Waymo tutorial the
    offset is added to global-frame coordinates after the vehicle-to-world transform;
    applying it to the ego pose translation is equivalent.

    This follows the same pattern as AV2, nuPlan, PandaSet, and KITTI-360 parsers.
    """
    camera_data_list: List[ParsedCamera] = []

    for image_proto in frame.images:
        camera_type = WOD_PERCEPTION_CAMERA_IDS[image_proto.name]
        metadata = camera_metadatas[camera_type]

        # image_proto.pose is the ego pose at the camera trigger time.
        # Apply the same map_pose_offset as for frame.pose (both are in the same global frame).
        camera_ego_pose = PoseSE3.from_transformation_matrix(
            np.array(image_proto.pose.transform, dtype=np.float64).reshape(4, 4)
        )
        camera_ego_pose.array[PoseSE3Index.XYZ] += map_pose_offset.array[Vector3DIndex.XYZ]

        # Compute global camera pose: ego_pose @ camera_to_imu_se3
        # camera_to_imu_se3 already has the pXpZmY -> pZmYpX convention conversion applied.
        camera_to_global_se3 = rel_to_abs_se3(
            origin=camera_ego_pose,
            pose_se3=metadata.camera_to_imu_se3,
        )

        # NOTE: WOD also provides {shutter, camera_trigger_time, camera_readout_done_time}
        camera_data_list.append(
            ParsedCamera(
                metadata=metadata,
                timestamp=Timestamp.from_s(image_proto.pose_timestamp),
                camera_to_global_se3=camera_to_global_se3,
                byte_string=image_proto.image,
            )
        )

    return camera_data_list


def _extract_wod_perception_lidar(
    frame: dataset_pb2.Frame,
    frame_idx: int,
    absolute_tf_record_path: Path,
    wod_perception_data_root: Path,
    lidar_merged_metadata: LidarMergedMetadata,
) -> ParsedLidar:
    """Extracts the merged Lidar data from a WOD Perception frame.

    WOD lidar operates at 10Hz (one full 360° rotation per frame, ~100ms).
    ``frame.timestamp_micros`` marks the start of the TOP lidar spin [1].

    [1] https://github.com/waymo-research/waymo-open-dataset/issues/464
    """
    relative_path = absolute_tf_record_path.relative_to(wod_perception_data_root)
    start_timestamp = Timestamp.from_us(frame.timestamp_micros)
    end_timestamp = Timestamp.from_us(frame.timestamp_micros + WOD_PERCEPTION_LIDAR_SWEEP_DURATION_US)
    return ParsedLidar(
        metadata=lidar_merged_metadata,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        dataset_root=wod_perception_data_root,
        relative_path=relative_path,
        iteration=frame_idx,
    )
