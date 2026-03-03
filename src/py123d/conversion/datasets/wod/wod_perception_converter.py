import logging
import os
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from py123d.api.map.abstract_map_writer import AbstractMapWriter
from py123d.api.scene.abstract_log_writer import AbstractLogWriter, CameraData, LidarData
from py123d.common.utils.dependencies import check_dependencies
from py123d.conversion.abstract_dataset_converter import AbstractDatasetConverter
from py123d.conversion.dataset_converter_config import DatasetConverterConfig
from py123d.conversion.datasets.wod.utils.wod_constants import (
    WOD_PERCEPTION_AVAILABLE_SPLITS,
    WOD_PERCEPTION_CAMERA_IDS,
    WOD_PERCEPTION_LIDAR_IDS,
)
from py123d.conversion.datasets.wod.wod_map_conversion import convert_wod_map
from py123d.conversion.datasets.wod.wod_perception_sensor_io import load_wod_perception_point_cloud_data_from_frame
from py123d.conversion.registry.box_detection_label_registry import WODPerceptionBoxDetectionLabel
from py123d.conversion.utils.sensor_utils.camera_conventions import CameraConvention, convert_camera_convention
from py123d.datatypes import (
    BoxDetectionMetadata,
    BoxDetectionSE3,
    BoxDetectionsSE3,
    EgoStateSE3,
    LidarID,
    LidarMetadata,
    LogMetadata,
    MapMetadata,
    PinholeCameraID,
    PinholeCameraMetadata,
    PinholeDistortion,
    PinholeIntrinsics,
    Timestamp,
)
from py123d.datatypes.vehicle_state.vehicle_parameters import get_wod_perception_chrysler_pacifica_parameters
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

check_dependencies(modules=["tensorflow"], optional_name="waymo")
import tensorflow as tf

from py123d.conversion.datasets.wod.waymo_open_dataset.protos import dataset_pb2

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.set_visible_devices(tf.config.list_physical_devices("CPU"))

logger = logging.getLogger(__name__)


class WODPerceptionConverter(AbstractDatasetConverter):
    """Converter for the Waymo Open Dataset - Perception."""

    def __init__(
        self,
        splits: List[str],
        wod_perception_data_root: Union[Path, str],
        zero_roll_pitch: bool,
        keep_polar_features: bool,
        add_map_pose_offset: bool,
        dataset_converter_config: DatasetConverterConfig,
    ) -> None:
        """Initializes the :class:`WODPConverter`.

        :param splits: List of splits to convert, i.e. ``["wod_perception_train", "wod_perception_val", "wod_perception_test"]``.
        :param wod_perception_data_root: Path to the root directory of the WODP dataset
        :param zero_roll_pitch: Whether to zero out roll and pitch angles in the vehicle pose
        :param keep_polar_features: Whether to keep polar features in the Lidar point clouds
        :param add_map_pose_offset: Whether to add a pose offset to the map
        :param dataset_converter_config: Configuration for the dataset converter
        """

        super().__init__(dataset_converter_config)
        for split in splits:
            assert split in WOD_PERCEPTION_AVAILABLE_SPLITS, (
                f"Split {split} is not available. Available splits: {WOD_PERCEPTION_AVAILABLE_SPLITS}"
            )

        self._splits: List[str] = splits
        self._wod_perception_data_root: Path = Path(wod_perception_data_root)
        self._zero_roll_pitch: bool = zero_roll_pitch
        self._keep_polar_features: bool = keep_polar_features
        self._add_map_pose_offset: bool = add_map_pose_offset

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

    def get_number_of_maps(self) -> int:
        """Inherited, see superclass."""
        return len(self._split_tf_record_pairs)

    def get_number_of_logs(self) -> int:
        """Inherited, see superclass."""
        return len(self._split_tf_record_pairs)

    def convert_map(self, map_index: int, map_writer: AbstractMapWriter) -> None:
        """Inherited, see superclass."""
        split, source_tf_record_path = self._split_tf_record_pairs[map_index]
        initial_frame = _get_initial_frame_from_tfrecord(source_tf_record_path)

        map_metadata = _get_wod_perception_map_metadata(initial_frame, split)
        map_needs_writing = map_writer.reset(self.dataset_converter_config, map_metadata)
        if map_needs_writing:
            convert_wod_map(initial_frame.map_features, map_writer)  # type: ignore

        map_writer.close()

    def convert_log(self, log_index: int, log_writer: AbstractLogWriter) -> None:
        """Inherited, see superclass."""

        split, source_tf_record_path = self._split_tf_record_pairs[log_index]

        initial_frame = _get_initial_frame_from_tfrecord(source_tf_record_path, keep_dataset=False)
        log_name = str(initial_frame.context.name)
        dataset = tf.data.TFRecordDataset(source_tf_record_path, compression_type="")

        # 1. Initialize Metadata
        log_metadata = LogMetadata(
            dataset="wod_perception",
            split=split,
            log_name=log_name,
            location=str(initial_frame.context.stats.location),
            timestep_seconds=0.1,
            vehicle_parameters=get_wod_perception_chrysler_pacifica_parameters(),
            box_detection_label_class=WODPerceptionBoxDetectionLabel,
            pinhole_camera_metadata=_get_wod_perception_camera_metadata(
                initial_frame,
                self.dataset_converter_config,
            ),
            lidar_metadata=_get_wod_perception_lidar_metadata(
                initial_frame,
                self._keep_polar_features,
                self.dataset_converter_config,
            ),
            map_metadata=_get_wod_perception_map_metadata(initial_frame, split),
        )

        # 2. Prepare log writer
        log_needs_writing = log_writer.reset(self.dataset_converter_config, log_metadata)

        # 3. Process source log data
        if log_needs_writing:
            try:
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

                    log_writer.write(
                        timestamp=Timestamp.from_us(frame.timestamp_micros),
                        ego_state=_extract_wod_perception_ego_state(frame, map_pose_offset),
                        box_detections=_extract_wod_perception_box_detections(
                            frame, map_pose_offset, self._zero_roll_pitch
                        ),
                        traffic_lights=None,  # NOTE: traffic lights are in the map proto, but only found in motion dataset.
                        pinhole_cameras=_extract_wod_perception_cameras(frame, self.dataset_converter_config),
                        lidar=_extract_wod_perception_lidars(
                            frame,
                            self._keep_polar_features,
                            frame_idx,
                            self.dataset_converter_config,
                            source_tf_record_path,
                            self._wod_perception_data_root,
                        ),
                    )
            except Exception as e:
                logger.error(f"Error processing log {log_name}: {e}")
                traceback.print_exc()

        log_writer.close()


def _get_initial_frame_from_tfrecord(
    tf_record_path: Path,
    keep_dataset: bool = False,
) -> Union[dataset_pb2.Frame, Tuple[dataset_pb2.Frame, tf.data.TFRecordDataset]]:
    """Helper to get the initial frame from a tf record file."""

    dataset = tf.data.TFRecordDataset(tf_record_path, compression_type="")
    for data in dataset:
        initial_frame = dataset_pb2.Frame()
        initial_frame.ParseFromString(data.numpy())
        break

    if keep_dataset:
        return initial_frame, dataset

    del dataset
    return initial_frame


def _get_wod_perception_map_metadata(initial_frame: dataset_pb2.Frame, split: str) -> MapMetadata:
    """Gets the WOD Perception map metadata from the initial frame."""
    map_metadata = MapMetadata(
        dataset="wod_perception",
        split=split,
        log_name=str(initial_frame.context.name),
        location=None,  # TODO: Add location information.
        map_has_z=True,
        map_is_local=True,
    )
    return map_metadata


def _get_wod_perception_camera_metadata(
    initial_frame: dataset_pb2.Frame, dataset_converter_config: DatasetConverterConfig
) -> Dict[PinholeCameraID, PinholeCameraMetadata]:
    """Get the WODP camera metadata from the initial frame."""
    camera_metadata_dict: Dict[PinholeCameraID, PinholeCameraMetadata] = {}
    if dataset_converter_config.pinhole_camera_store_option is not None:
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


def _get_wod_perception_lidar_metadata(
    initial_frame: dataset_pb2.Frame,
    keep_polar_features: bool,
    dataset_converter_config: DatasetConverterConfig,
) -> Dict[LidarID, LidarMetadata]:
    """Get the WODP Lidar metadata from the initial frame."""

    laser_metadatas: Dict[LidarID, LidarMetadata] = {}
    if dataset_converter_config.lidar_store_option is not None:
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

    return laser_metadatas


def _get_ego_pose_se3(frame: dataset_pb2.Frame, map_pose_offset: Vector3D) -> PoseSE3:
    """Helper to get the ego pose SE3 from a WODP frame."""
    ego_pose_matrix = np.array(frame.pose.transform, dtype=np.float64).reshape(4, 4)
    ego_pose_se3 = PoseSE3.from_transformation_matrix(ego_pose_matrix)
    ego_pose_se3.array[PoseSE3Index.XYZ] += map_pose_offset.array[Vector3DIndex.XYZ]
    return ego_pose_se3


def _extract_wod_perception_ego_state(frame: dataset_pb2.Frame, map_pose_offset: Vector3D) -> EgoStateSE3:
    """Extracts the ego state from a WODP frame."""
    imu_se3 = _get_ego_pose_se3(frame, map_pose_offset)
    vehicle_parameters = get_wod_perception_chrysler_pacifica_parameters()
    dynamic_state_se3 = None
    return EgoStateSE3.from_imu(
        imu_se3=imu_se3,
        dynamic_state_se3=dynamic_state_se3,
        vehicle_parameters=vehicle_parameters,
        timestamp=None,
    )


def _extract_wod_perception_box_detections(
    frame: dataset_pb2.Frame, map_pose_offset: Vector3D, zero_roll_pitch: bool = True
) -> BoxDetectionsSE3:
    """Extracts the box detections from a WODP frame."""

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
                metadata=BoxDetectionMetadata(
                    label=detections_types[detection_idx],
                    timestamp=None,
                    track_token=detections_token[detection_idx],
                ),
                bounding_box_se3=BoundingBoxSE3.from_array(detections_state[detection_idx]),
                velocity_3d=Vector3D.from_array(detections_velocity[detection_idx]),
            )
        )
    return BoxDetectionsSE3(box_detections=box_detections)


def _extract_wod_perception_cameras(
    frame: dataset_pb2.Frame, dataset_converter_config: DatasetConverterConfig
) -> List[CameraData]:
    """Extracts the camera data from a WODP frame."""

    camera_data_list: List[CameraData] = []
    if dataset_converter_config.include_pinhole_cameras:
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
                CameraData(
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
    dataset_converter_config: DatasetConverterConfig,
    absolute_tf_record_path: Path,
    wod_perception_data_root: Path,
) -> Optional[LidarData]:
    """Extracts the Lidar data from a WODP frame."""

    lidar: Optional[LidarData] = None
    if dataset_converter_config.include_lidars:
        if dataset_converter_config.lidar_store_option == "path":
            relative_path = absolute_tf_record_path.relative_to(wod_perception_data_root)
            lidar = LidarData(
                lidar_name=LidarID.LIDAR_MERGED.serialize(),
                lidar_type=LidarID.LIDAR_MERGED,
                iteration=frame_idx,
                dataset_root=wod_perception_data_root,
                relative_path=relative_path,
            )
        else:
            point_cloud_3d, point_cloud_features = load_wod_perception_point_cloud_data_from_frame(
                frame, keep_polar_features=keep_polar_features
            )
            lidar = LidarData(
                lidar_name=LidarID.LIDAR_MERGED.serialize(),
                lidar_type=LidarID.LIDAR_MERGED,
                iteration=frame_idx,
                point_cloud_3d=point_cloud_3d,
                point_cloud_features=point_cloud_features,
            )
    return lidar
