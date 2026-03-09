from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

from py123d.datatypes import (
    BoxDetectionAttributes,
    BoxDetectionSE3,
    BoxDetectionsSE3,
    BoxDetectionsSE3Metadata,
    EgoStateSE3,
    EgoStateSE3Metadata,
    FisheyeMEICameraMetadatas,
    LidarID,
    LidarMetadata,
    LidarMetadatas,
    LogMetadata,
    PinholeCameraID,
    PinholeCameraMetadata,
    PinholeCameraMetadatas,
    PinholeDistortion,
    PinholeIntrinsics,
    Timestamp,
)
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
from py123d.geometry.transform import rel_to_abs_se3_array
from py123d.geometry.utils.constants import DEFAULT_PITCH, DEFAULT_ROLL
from py123d.geometry.utils.rotation_utils import (
    get_euler_array_from_quaternion_array,
    get_quaternion_array_from_euler_array,
)
from py123d.parser.abstract_dataset_parser import (
    DatasetParser,
    LogParser,
    MapParser,
    ParsedCamera,
    ParsedFrame,
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


class WODPerceptionParser(DatasetParser):
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

    def get_map_parsers(self) -> List[MapParser]:
        """Inherited, see superclass."""
        map_parsers: List[MapParser] = []
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

    def get_log_parsers(self) -> List[LogParser]:
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


class WODPerceptionLogParser(LogParser):
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
            timestep_seconds=0.1,
        )

    def get_ego_metadata(self) -> Optional[EgoStateSE3Metadata]:
        """Inherited, see superclass."""
        # NOTE: These parameters are estimates based on the vehicle model used in the WOD Perception dataset.
        # The vehicle should be the same (or a similar) vehicle model to nuPlan and PandaSet [1].
        # [1] https://en.wikipedia.org/wiki/Chrysler_Pacifica_(minivan)
        return EgoStateSE3Metadata(
            vehicle_name="wod-perception_chrysler_pacifica",
            width=2.297,
            length=5.176,
            height=1.777,
            wheel_base=3.089,
            center_to_imu_se3=PoseSE3(x=1.461, y=0.0, z=1.777 / 2, qw=1.0, qx=0.0, qy=0.0, qz=0.0),
            rear_axle_to_imu_se3=PoseSE3.identity(),
        )

    def get_box_detection_metadata(self) -> Optional[BoxDetectionsSE3Metadata]:
        """Inherited, see superclass."""
        return BoxDetectionsSE3Metadata(box_detection_label_class=WODPerceptionBoxDetectionLabel)

    def get_pinhole_camera_metadatas(self) -> Optional[PinholeCameraMetadatas]:
        """Inherited, see superclass."""
        initial_frame = _get_initial_frame_from_tfrecord(self._source_tf_record_path)
        return _get_wod_perception_camera_metadata(initial_frame)

    def get_fisheye_mei_camera_metadatas(self) -> Optional[FisheyeMEICameraMetadatas]:
        """Inherited, see superclass."""
        return None

    def get_lidar_metadatas(self) -> Optional[LidarMetadatas]:
        """Inherited, see superclass."""
        initial_frame = _get_initial_frame_from_tfrecord(self._source_tf_record_path)
        return _get_wod_perception_lidar_metadata(initial_frame, self._keep_polar_features)

    def iter_frames(self) -> Iterator[ParsedFrame]:
        """Yields one FrameData per frame in the TFRecord."""
        tf, dataset_pb2 = _lazy_import_tf_and_pb2()

        dataset = tf.data.TFRecordDataset(self._source_tf_record_path, compression_type="")
        ego_metadata = self.get_ego_metadata()
        assert ego_metadata is not None, "Ego metadata must be available to iterate frames."

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

            timestamp = Timestamp.from_us(frame.timestamp_micros)

            yield ParsedFrame(
                timestamp=timestamp,
                ego_state_se3=_extract_wod_perception_ego_state(
                    frame, map_pose_offset, ego_metadata, timestamp=timestamp
                ),
                box_detections_se3=_extract_wod_perception_box_detections(
                    frame, map_pose_offset, self._zero_roll_pitch, timestamp
                ),
                pinhole_cameras=_extract_wod_perception_cameras(frame),
                lidars=_extract_wod_perception_lidars(
                    frame,
                    self._keep_polar_features,
                    frame_idx,
                    self._source_tf_record_path,
                    self._wod_perception_data_root,
                    timestamp,
                ),
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


def _get_wod_perception_camera_metadata(initial_frame: dataset_pb2.Frame) -> Optional[PinholeCameraMetadatas]:
    """Get the WOD Perception camera metadata from the initial frame."""
    camera_metadata_dict: Dict[PinholeCameraID, PinholeCameraMetadata] = {}
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

    return PinholeCameraMetadatas(camera_metadata_dict) if camera_metadata_dict else None


def _get_wod_perception_lidar_metadata(
    initial_frame: dataset_pb2.Frame,
    keep_polar_features: bool,
) -> Optional[LidarMetadatas]:
    """Get the WOD Perception Lidar metadata from the initial frame."""
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

    return LidarMetadatas(laser_metadatas) if laser_metadatas else None


def _get_ego_pose_se3(frame: dataset_pb2.Frame, map_pose_offset: Vector3D) -> PoseSE3:
    """Helper to get the ego pose SE3 from a WOD Perception frame."""
    ego_pose_matrix = np.array(frame.pose.transform, dtype=np.float64).reshape(4, 4)
    ego_pose_se3 = PoseSE3.from_transformation_matrix(ego_pose_matrix)
    ego_pose_se3.array[PoseSE3Index.XYZ] += map_pose_offset.array[Vector3DIndex.XYZ]
    return ego_pose_se3


def _extract_wod_perception_ego_state(
    frame: dataset_pb2.Frame, map_pose_offset: Vector3D, ego_metadata: EgoStateSE3Metadata, timestamp: Timestamp
) -> EgoStateSE3:
    """Extracts the ego state from a WOD Perception frame."""
    imu_se3 = _get_ego_pose_se3(frame, map_pose_offset)
    dynamic_state_se3 = None
    return EgoStateSE3.from_imu(
        imu_se3=imu_se3,
        dynamic_state_se3=dynamic_state_se3,
        ego_metadata=ego_metadata,
        timestamp=timestamp,
    )


def _extract_wod_perception_box_detections(
    frame: dataset_pb2.Frame,
    map_pose_offset: Vector3D,
    zero_roll_pitch: bool = True,
    timestamp: Optional[Timestamp] = None,
) -> BoxDetectionsSE3:
    """Extracts the box detections from a WOD Perception frame."""

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

        # 2. Fill SE3 Bounding Box
        detections_state[detection_idx, BoundingBoxSE3Index.X] = detection.box.center_x
        detections_state[detection_idx, BoundingBoxSE3Index.Y] = detection.box.center_y
        detections_state[detection_idx, BoundingBoxSE3Index.Z] = detection.box.center_z
        detections_state[detection_idx, BoundingBoxSE3Index.QUATERNION] = detection_quaternion
        detections_state[detection_idx, BoundingBoxSE3Index.LENGTH] = detection.box.length
        detections_state[detection_idx, BoundingBoxSE3Index.WIDTH] = detection.box.width
        detections_state[detection_idx, BoundingBoxSE3Index.HEIGHT] = detection.box.height

        # 2. Velocity TODO: check if velocity needs to be rotated
        detections_velocity[detection_idx] = Vector3D(
            x=detection.metadata.speed_x,
            y=detection.metadata.speed_y,
            z=detection.metadata.speed_z,
        ).array

        # 3. Type and track token
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
                metadata=BoxDetectionAttributes(
                    label=detections_types[detection_idx],
                    track_token=detections_token[detection_idx],
                ),
                bounding_box_se3=BoundingBoxSE3.from_array(detections_state[detection_idx]),
                velocity_3d=Vector3D.from_array(detections_velocity[detection_idx]),
            )
        )
    return BoxDetectionsSE3(box_detections=box_detections, timestamp=timestamp)


def _extract_wod_perception_cameras(frame: dataset_pb2.Frame) -> List[ParsedCamera]:
    """Extracts the camera data from a WOD Perception frame."""
    camera_data_list: List[ParsedCamera] = []

    # NOTE @DanielDauner: The extrinsic matrix in frame.context.camera_calibration is fixed to model the ego to camera transformation.
    # The poses in frame.images[idx] are the motion compensated ego poses when the camera triggers.
    camera_extrinsic: Dict[PinholeCameraID, PoseSE3] = {}
    for calibration in frame.context.camera_calibrations:
        camera_type = WOD_PERCEPTION_CAMERA_IDS[calibration.name]
        camera_transform = np.array(calibration.extrinsic.transform, dtype=np.float64).reshape(4, 4)
        camera_pose = PoseSE3.from_transformation_matrix(camera_transform)
        # NOTE: WOD Perception uses a different camera convention than py123d
        # https://arxiv.org/pdf/1912.04838 (Figure 1.)
        camera_pose = convert_camera_convention(
            camera_pose,
            from_convention=CameraConvention.pXpZmY,
            to_convention=CameraConvention.pZmYpX,
        )
        camera_extrinsic[camera_type] = camera_pose

    for image_proto in frame.images:
        camera_type = WOD_PERCEPTION_CAMERA_IDS[image_proto.name]

        # NOTE @DanielDauner: We store the pose_timestamp of each camera. WOD-Perception also provides:
        # {shutter, camera_trigger_time, camera_readout_done_time}
        camera_data_list.append(
            ParsedCamera(
                camera_name=str(image_proto.name),
                camera_id=camera_type,
                extrinsic=camera_extrinsic[camera_type],
                jpeg_binary=image_proto.image,
                timestamp=Timestamp.from_s(image_proto.pose_timestamp),
            )
        )

    return camera_data_list


def _extract_wod_perception_lidars(
    frame: dataset_pb2.Frame,
    keep_polar_features: bool,
    frame_idx: int,
    absolute_tf_record_path: Path,
    wod_perception_data_root: Path,
    timestamp: Timestamp,
) -> Optional[ParsedLidar]:
    """Extracts the Lidar data from a WOD Perception frame."""
    relative_path = absolute_tf_record_path.relative_to(wod_perception_data_root)
    return ParsedLidar(
        lidar_name=LidarID.LIDAR_MERGED.serialize(),
        lidar_type=LidarID.LIDAR_MERGED,
        start_timestamp=timestamp,
        end_timestamp=timestamp,
        iteration=frame_idx,
        dataset_root=wod_perception_data_root,
        relative_path=relative_path,
    )
