from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from py123d.api.map.abstract_map_writer import AbstractMapWriter
from py123d.api.scene.abstract_log_writer import AbstractLogWriter, CameraData, LidarData
from py123d.conversion.abstract_dataset_converter import AbstractDatasetConverter
from py123d.conversion.dataset_converter_config import DatasetConverterConfig
from py123d.conversion.datasets.pandaset.utils.pandaset_constants import (
    PANDASET_BOX_DETECTION_FROM_STR,
    PANDASET_CAMERA_DISTORTIONS,
    PANDASET_CAMERA_EXTRINSICS,
    PANDASET_CAMERA_MAPPING,
    PANDASET_LIDAR_EXTRINSICS,
    PANDASET_LIDAR_MAPPING,
    PANDASET_LOG_NAMES,
    PANDASET_SPLITS,
)
from py123d.conversion.datasets.pandaset.utils.pandaset_utlis import (
    extrinsic_to_imu,
    global_main_lidar_to_global_imu,
    pandaset_pose_dict_to_pose_se3,
    read_json,
    read_pkl_gz,
    rotate_pandaset_pose_to_iso_coordinates,
)
from py123d.conversion.registry import PandasetBoxDetectionLabel
from py123d.datatypes import (
    BoxDetectionMetadata,
    BoxDetectionSE3,
    BoxDetectionsSE3,
    EgoStateSE3,
    LidarID,
    LidarMetadata,
    LogMetadata,
    PinholeCameraID,
    PinholeCameraMetadata,
    PinholeIntrinsics,
    Timestamp,
)
from py123d.datatypes.vehicle_state.vehicle_parameters import get_pandaset_chrysler_pacifica_parameters
from py123d.geometry import BoundingBoxSE3, BoundingBoxSE3Index, EulerAnglesIndex, PoseSE3
from py123d.geometry.transform import abs_to_rel_se3_array
from py123d.geometry.utils.constants import DEFAULT_PITCH, DEFAULT_ROLL
from py123d.geometry.utils.rotation_utils import get_quaternion_array_from_euler_array


class PandasetConverter(AbstractDatasetConverter):
    """Converter for the Pandaset dataset."""

    def __init__(
        self,
        splits: List[str],
        pandaset_data_root: Union[Path, str],
        dataset_converter_config: DatasetConverterConfig,
        train_log_names: List[str],
        val_log_names: List[str],
        test_log_names: List[str],
    ) -> None:
        """Initializes the :class:`PandasetConverter`.

        :param splits: List of splits to include in the conversion. \
            Available splits: 'pandaset_train', 'pandaset_val', 'pandaset_test'.
        :param pandaset_data_root: Path to the root directory of the Pandaset dataset
        :param dataset_converter_config: Configuration for the dataset converter
        :param train_log_names: List of log names to include in the training split
        :param val_log_names: List of log names to include in the validation split
        :param test_log_names: List of log names to include in the test split
        """
        super().__init__(dataset_converter_config)
        for split in splits:
            assert split in PANDASET_SPLITS, f"Split {split} is not available. Available splits: {PANDASET_SPLITS}"
        assert pandaset_data_root is not None, "The variable `pandaset_data_root` must be provided."

        self._splits: List[str] = splits
        self._pandaset_data_root: Path = Path(pandaset_data_root)

        self._train_log_names: List[str] = train_log_names
        self._val_log_names: List[str] = val_log_names
        self._test_log_names: List[str] = test_log_names
        self._log_paths_and_split: List[Tuple[Path, str]] = self._collect_log_paths()

    def _collect_log_paths(self) -> List[Tuple[Path, str]]:
        log_paths_and_split: List[Tuple[Path, str]] = []

        for log_folder in self._pandaset_data_root.iterdir():
            if not log_folder.is_dir():
                continue

            log_name = log_folder.name
            assert log_name in PANDASET_LOG_NAMES, f"Log name {log_name} is not recognized."
            if (log_name in self._train_log_names) and ("pandaset_train" in self._splits):
                log_paths_and_split.append((log_folder, "pandaset_train"))
            elif (log_name in self._val_log_names) and ("pandaset_val" in self._splits):
                log_paths_and_split.append((log_folder, "pandaset_val"))
            elif (log_name in self._test_log_names) and ("pandaset_test" in self._splits):
                log_paths_and_split.append((log_folder, "pandaset_test"))

        return log_paths_and_split

    def get_number_of_maps(self) -> int:
        """Inherited, see superclass."""
        return 0  # NOTE @DanielDauner: Pandaset does not have maps.

    def get_number_of_logs(self) -> int:
        """Inherited, see superclass."""
        return len(self._log_paths_and_split)

    def convert_map(self, map_index: int, map_writer: AbstractMapWriter) -> None:
        """Inherited, see superclass."""
        return None  # NOTE @DanielDauner: Pandaset does not have maps.

    def convert_log(self, log_index: int, log_writer: AbstractLogWriter) -> None:
        """Inherited, see superclass."""

        source_log_path, split = self._log_paths_and_split[log_index]

        # 1. Initialize Metadata
        log_metadata = LogMetadata(
            dataset="pandaset",
            split=split,
            log_name=source_log_path.name,
            location=None,  # TODO: Add location information.
            timestep_seconds=0.1,
            vehicle_parameters=get_pandaset_chrysler_pacifica_parameters(),
            box_detection_label_class=PandasetBoxDetectionLabel,
            pinhole_camera_metadata=_get_pandaset_camera_metadata(source_log_path, self.dataset_converter_config),
            lidar_metadata=_get_pandaset_lidar_metadata(self.dataset_converter_config),
            map_metadata=None,  # NOTE @DanielDauner:  Pandaset does not have maps.
        )

        # 2. Prepare log writer
        log_needs_writing = log_writer.reset(self.dataset_converter_config, log_metadata)

        # 3. Process source log data
        if log_needs_writing:
            # Read files from pandaset
            timesteps = read_json(source_log_path / "meta" / "timestamps.json")
            gps: List[Dict[str, float]] = read_json(source_log_path / "meta" / "gps.json")
            lidar_poses: List[Dict[str, Dict[str, float]]] = read_json(source_log_path / "lidar" / "poses.json")
            camera_poses: Dict[str, List[Dict[str, Dict[str, float]]]] = {
                camera_name: read_json(source_log_path / "camera" / camera_name / "poses.json")
                for camera_name in PANDASET_CAMERA_MAPPING.keys()
            }
            camera_timestamps: Dict[str, List[float]] = {
                camera_name: read_json(source_log_path / "camera" / camera_name / "timestamps.json")
                for camera_name in PANDASET_CAMERA_MAPPING.keys()
            }

            # Write data to log writer
            for iteration, timestep_s in enumerate(timesteps):
                ego_state = _extract_pandaset_sensor_ego_state(gps[iteration], lidar_poses[iteration])
                log_writer.write(
                    timestamp=Timestamp.from_s(timestep_s),
                    ego_state=ego_state,
                    box_detections=_extract_pandaset_box_detections(source_log_path, iteration),
                    pinhole_cameras=_extract_pandaset_pinhole_cameras(
                        source_log_path,
                        iteration,
                        ego_state,
                        camera_poses,
                        camera_timestamps,
                        self.dataset_converter_config,
                    ),
                    lidar=_extract_pandaset_lidar(
                        source_log_path,
                        iteration,
                        self.dataset_converter_config,
                    ),
                )

        # 4. Finalize log writing
        log_writer.close()


def _get_pandaset_camera_metadata(
    source_log_path: Path, dataset_config: DatasetConverterConfig
) -> Dict[PinholeCameraID, PinholeCameraMetadata]:
    """Extracts the pinhole camera metadata from a Pandaset log folder."""

    camera_metadata: Dict[PinholeCameraID, PinholeCameraMetadata] = {}
    if dataset_config.include_pinhole_cameras:
        all_cameras_folder = source_log_path / "camera"
        for camera_folder in all_cameras_folder.iterdir():
            camera_name = camera_folder.name
            assert camera_name in PANDASET_CAMERA_MAPPING.keys(), f"Camera name {camera_name} is not recognized."

            camera_type = PANDASET_CAMERA_MAPPING[camera_name]
            intrinsics_file = camera_folder / "intrinsics.json"
            assert intrinsics_file.exists(), f"Camera intrinsics file {intrinsics_file} does not exist."

            intrinsics_data = read_json(intrinsics_file)
            camera_metadata[camera_type] = PinholeCameraMetadata(
                camera_name=camera_name,
                camera_id=camera_type,
                width=1920,
                height=1080,
                intrinsics=PinholeIntrinsics(
                    fx=intrinsics_data["fx"],
                    fy=intrinsics_data["fy"],
                    cx=intrinsics_data["cx"],
                    cy=intrinsics_data["cy"],
                ),
                distortion=PANDASET_CAMERA_DISTORTIONS[camera_name],
                camera_to_imu_se3=extrinsic_to_imu(PANDASET_CAMERA_EXTRINSICS[camera_name]),
            )

    return camera_metadata


def _get_pandaset_lidar_metadata(dataset_config: DatasetConverterConfig) -> Dict[LidarID, LidarMetadata]:
    """Extracts the Lidar metadata from a Pandaset log folder."""
    lidar_metadata: Dict[LidarID, LidarMetadata] = {}
    if dataset_config.include_lidars:
        for lidar_name, lidar_type in PANDASET_LIDAR_MAPPING.items():
            lidar_metadata[lidar_type] = LidarMetadata(
                lidar_name=lidar_name,
                lidar_id=lidar_type,
                lidar_to_imu_se3=extrinsic_to_imu(PANDASET_LIDAR_EXTRINSICS[lidar_name]),
            )

    return lidar_metadata


def _extract_pandaset_sensor_ego_state(gps: Dict[str, float], lidar_pose: Dict[str, Dict[str, float]]) -> EgoStateSE3:
    """Extracts the ego state from Pandaset GPS and Lidar pose data."""
    imu_se3 = global_main_lidar_to_global_imu(pandaset_pose_dict_to_pose_se3(lidar_pose))
    vehicle_parameters = get_pandaset_chrysler_pacifica_parameters()
    dynamic_state_se3 = None
    return EgoStateSE3.from_imu(
        imu_se3=imu_se3,
        vehicle_parameters=vehicle_parameters,
        dynamic_state_se3=dynamic_state_se3,
        timestamp=None,
    )


def _extract_pandaset_box_detections(source_log_path: Path, iteration: int) -> BoxDetectionsSE3:
    """Extracts the box detections from a Pandaset log folder at a given iteration."""

    # NOTE @DanielDauner: The following provided quboids annotations are not stored in 123D
    # - stationary
    # - camera_used
    # - attributes.object_motion
    # - cuboids.sibling_id
    # - cuboids.sensor_id
    # - attributes.pedestrian_behavior
    # - attributes.pedestrian_age
    # - attributes.rider_status
    # https://github.com/scaleapi/pandaset-devkit/blob/master/README.md?plain=1#L288

    iteration_str = f"{iteration:02d}"
    cuboids_file = source_log_path / "annotations" / "cuboids" / f"{iteration_str}.pkl.gz"

    if not cuboids_file.exists():
        return BoxDetectionsSE3(box_detections=[])

    cuboid_df: pd.DataFrame = read_pkl_gz(cuboids_file)

    # Read cuboid data
    box_label_names = list(cuboid_df["label"])
    box_uuids = list(cuboid_df["uuid"])
    num_boxes = len(box_uuids)

    box_position_x = np.array(cuboid_df["position.x"], dtype=np.float64)
    box_position_y = np.array(cuboid_df["position.y"], dtype=np.float64)
    box_position_z = np.array(cuboid_df["position.z"], dtype=np.float64)
    box_points = np.stack([box_position_x, box_position_y, box_position_z], axis=-1)
    box_yaws = np.array(cuboid_df["yaw"], dtype=np.float64)

    # NOTE: Rather strange format to have dimensions.x as width, dimensions.y as length
    box_widths = np.array(cuboid_df["dimensions.x"], dtype=np.float64)
    box_lengths = np.array(cuboid_df["dimensions.y"], dtype=np.float64)
    box_heights = np.array(cuboid_df["dimensions.z"], dtype=np.float64)

    # Create se3 array for boxes (i.e. convert rotation to quaternion)
    box_euler_angles_array = np.zeros((num_boxes, len(EulerAnglesIndex)), dtype=np.float64)
    box_euler_angles_array[..., EulerAnglesIndex.ROLL] = DEFAULT_ROLL
    box_euler_angles_array[..., EulerAnglesIndex.PITCH] = DEFAULT_PITCH
    box_euler_angles_array[..., EulerAnglesIndex.YAW] = box_yaws

    box_se3_array = np.zeros((num_boxes, len(BoundingBoxSE3Index)), dtype=np.float64)
    box_se3_array[:, BoundingBoxSE3Index.XYZ] = box_points
    box_se3_array[:, BoundingBoxSE3Index.QUATERNION] = get_quaternion_array_from_euler_array(box_euler_angles_array)
    box_se3_array[:, BoundingBoxSE3Index.EXTENT] = np.stack([box_lengths, box_widths, box_heights], axis=-1)

    # NOTE @DanielDauner: Pandaset annotates moving bounding boxes twice (for synchronization reasons),
    # if they are in the overlap area between the top 360° lidar and the front-facing lidar (and moving).
    # The value in `cuboids.sensor_id` is either
    # - `0` (mechanical 360° Lidar)
    # - `1` (front-facing Lidar).
    # - All other cuboids have value `-1`.

    # To avoid duplicate bounding boxes, we only keep boxes from the front lidar (sensor_id == 1), if they do not
    # have a sibling box in the top lidar (sensor_id == 0). Otherwise, all boxes with sensor_id == {-1, 0} are kept.
    # https://github.com/scaleapi/pandaset-devkit/blob/master/python/pandaset/annotations.py#L166
    # https://github.com/scaleapi/pandaset-devkit/issues/26

    top_lidar_uuids = set(cuboid_df[cuboid_df["cuboids.sensor_id"] == 0]["uuid"])
    sensor_ids = cuboid_df["cuboids.sensor_id"].to_list()
    sibling_ids = cuboid_df["cuboids.sibling_id"].to_list()

    # Fill bounding box detections and return
    box_detections: List[BoxDetectionSE3] = []
    for box_idx in range(num_boxes):
        # Skip duplicate box detections from front lidar if sibling exists in top lidar
        if sensor_ids[box_idx] == 1 and sibling_ids[box_idx] in top_lidar_uuids:
            continue

        pandaset_box_detection_label = PANDASET_BOX_DETECTION_FROM_STR[box_label_names[box_idx]]

        # Convert coordinates to ISO 8855
        # NOTE: This would be faster over a batch operation.
        box_se3_array[box_idx, BoundingBoxSE3Index.SE3] = rotate_pandaset_pose_to_iso_coordinates(
            PoseSE3.from_array(box_se3_array[box_idx, BoundingBoxSE3Index.SE3], copy=False)
        ).array

        box_detection_se3 = BoxDetectionSE3(
            metadata=BoxDetectionMetadata(
                label=pandaset_box_detection_label,
                track_token=box_uuids[box_idx],
            ),
            bounding_box_se3=BoundingBoxSE3.from_array(box_se3_array[box_idx]),
            velocity_3d=None,
        )
        box_detections.append(box_detection_se3)

    return BoxDetectionsSE3(box_detections=box_detections)  # type: ignore


def _extract_pandaset_pinhole_cameras(
    source_log_path: Path,
    iteration: int,
    ego_state_se3: EgoStateSE3,
    camera_poses: Dict[str, List[Dict[str, Dict[str, float]]]],
    camera_timestamps: Dict[str, List[float]],
    dataset_converter_config: DatasetConverterConfig,
) -> List[CameraData]:
    """Extracts the pinhole camera metadata from a Pandaset scene at a given iteration."""
    camera_data_list: List[CameraData] = []
    iteration_str = f"{iteration:02d}"

    if dataset_converter_config.include_pinhole_cameras:
        for camera_name, camera_type in PANDASET_CAMERA_MAPPING.items():
            image_abs_path = source_log_path / f"camera/{camera_name}/{iteration_str}.jpg"
            assert image_abs_path.exists(), f"Camera image file {str(image_abs_path)} does not exist."

            camera_pose_dict = camera_poses[camera_name][iteration]
            camera_extrinsic = pandaset_pose_dict_to_pose_se3(camera_pose_dict)
            camera_extrinsic = PoseSE3.from_array(
                abs_to_rel_se3_array(ego_state_se3.rear_axle_se3, camera_extrinsic.array), copy=True
            )
            camera_timestamp = Timestamp.from_s(camera_timestamps[camera_name][iteration])

            camera_data_list.append(
                CameraData(
                    camera_name=camera_name,
                    camera_id=camera_type,
                    timestamp=camera_timestamp,
                    extrinsic=camera_extrinsic,
                    dataset_root=source_log_path.parent,
                    relative_path=image_abs_path.relative_to(source_log_path.parent),
                )
            )

    return camera_data_list


def _extract_pandaset_lidar(
    source_log_path: Path, iteration: int, dataset_converter_config: DatasetConverterConfig
) -> Optional[LidarData]:
    """Extracts the Lidar data from a Pandaset scene at a given iteration."""

    lidar_data: Optional[LidarData] = None
    if dataset_converter_config.include_lidars:
        iteration_str = f"{iteration:02d}"
        lidar_absolute_path = source_log_path / "lidar" / f"{iteration_str}.pkl.gz"
        assert lidar_absolute_path.exists(), f"Lidar file {str(lidar_absolute_path)} does not exist."
        lidar_data = LidarData(
            lidar_name=LidarID.LIDAR_MERGED.serialize(),
            lidar_type=LidarID.LIDAR_MERGED,
            timestamp=None,
            iteration=iteration,
            dataset_root=source_log_path.parent,
            relative_path=str(lidar_absolute_path.relative_to(source_log_path.parent)),
        )

    return lidar_data
