from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from py123d.datatypes import (
    BoxDetectionAttributes,
    BoxDetectionSE3,
    BoxDetectionsSE3,
    CameraID,
    EgoStateSE3,
    LogMetadata,
    PinholeCameraMetadata,
    PinholeIntrinsics,
    Timestamp,
)
from py123d.datatypes.detections.box_detections_metadata import BoxDetectionsSE3Metadata
from py123d.datatypes.modalities.base_modality import BaseModality
from py123d.datatypes.sensors.lidar import LidarMergedMetadata
from py123d.datatypes.vehicle_state.ego_state_metadata import EgoStateSE3Metadata
from py123d.geometry import BoundingBoxSE3, BoundingBoxSE3Index, EulerAnglesIndex, PoseSE3
from py123d.geometry.utils.constants import DEFAULT_PITCH, DEFAULT_ROLL
from py123d.geometry.utils.rotation_utils import get_quaternion_array_from_euler_array
from py123d.parser.base_dataset_parser import (
    BaseDatasetParser,
    BaseLogParser,
    BaseMapParser,
    ModalitiesSync,
    ParsedCamera,
    ParsedLidar,
)
from py123d.parser.pandaset.utils.pandaset_constants import (
    PANDASET_BOX_DETECTION_FROM_STR,
    PANDASET_BOX_DETECTIONS_SE3_METADATA,
    PANDASET_CAMERA_DISTORTIONS,
    PANDASET_CAMERA_EXTRINSICS,
    PANDASET_CAMERA_MAPPING,
    PANDASET_EGO_STATE_SE3_METADATA,
    PANDASET_LIDAR_MERGED_METADATA,
    PANDASET_LOG_NAMES,
    PANDASET_SPLITS,
)
from py123d.parser.pandaset.utils.pandaset_utils import (
    extrinsic_to_imu,
    global_main_lidar_to_global_imu,
    pandaset_pose_dict_to_pose_se3,
    read_json,
    read_pkl_gz,
    rotate_pandaset_pose_to_iso_coordinates,
)


class PandasetParser(BaseDatasetParser):
    """Dataset parser for the Pandaset dataset."""

    def __init__(
        self,
        splits: List[str],
        pandaset_data_root: Union[Path, str],
        train_log_names: List[str],
        val_log_names: List[str],
        test_log_names: List[str],
    ) -> None:
        """Initializes the :class:`PandasetParser`.

        :param splits: List of splits to include in the conversion. \
            Available splits: 'pandaset_train', 'pandaset_val', 'pandaset_test'.
        :param pandaset_data_root: Path to the root directory of the Pandaset dataset
        :param train_log_names: List of log names to include in the training split
        :param val_log_names: List of log names to include in the validation split
        :param test_log_names: List of log names to include in the test split
        """
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

    def get_log_parsers(self) -> List[PandasetLogParser]:  # type: ignore
        """Inherited, see superclass."""
        return [
            PandasetLogParser(source_log_path=source_log_path, split=split)
            for source_log_path, split in self._log_paths_and_split
        ]

    def get_map_parsers(self) -> List[BaseMapParser]:
        """Inherited, see superclass."""
        return []  # NOTE @DanielDauner: Pandaset does not have maps.


class PandasetLogParser(BaseLogParser):
    """Lightweight, picklable handle to one Pandaset log."""

    def __init__(self, source_log_path: Path, split: str) -> None:
        self._source_log_path = source_log_path
        self._split = split

    def get_log_metadata(self) -> LogMetadata:
        """Inherited, see superclass."""
        return LogMetadata(
            dataset="pandaset",
            split=self._split,
            log_name=self._source_log_path.name,
            location=None,  # TODO: Add location information.
            timestep_seconds=0.1,
        )

    def iter_modalities_sync(self) -> Iterator[ModalitiesSync]:
        """Inherited, see superclass."""
        source_log_path = self._source_log_path

        ego_state_se3_metadata = PANDASET_EGO_STATE_SE3_METADATA
        box_detections_se3_metadata = PANDASET_BOX_DETECTIONS_SE3_METADATA
        pinhole_cameras_metadata = _get_pandaset_camera_metadata(source_log_path)
        lidar_merged_metadata = PANDASET_LIDAR_MERGED_METADATA

        # Read files from pandaset
        lidar_timestamps_s = read_json(source_log_path / "meta" / "timestamps.json")

        lidar_poses: List[Dict[str, Dict[str, float]]] = read_json(source_log_path / "lidar" / "poses.json")
        camera_poses: Dict[str, List[Dict[str, Dict[str, float]]]] = {
            camera_name: read_json(source_log_path / "camera" / camera_name / "poses.json")
            for camera_name in PANDASET_CAMERA_MAPPING.keys()
        }
        camera_timestamps_s: Dict[str, List[float]] = {
            camera_name: read_json(source_log_path / "camera" / camera_name / "timestamps.json")
            for camera_name in PANDASET_CAMERA_MAPPING.keys()
        }

        for iteration, timestep_s in enumerate(lidar_timestamps_s):
            timestamp = Timestamp.from_s(timestep_s)
            ego_state = _extract_pandaset_sensor_ego_state(
                lidar_pose=lidar_poses[iteration],
                ego_metadata=ego_state_se3_metadata,
                timestamp=timestamp,
            )
            box_detections = _extract_pandaset_box_detections(
                source_log_path, iteration, timestamp, box_detections_se3_metadata
            )
            parsed_cameras = _extract_pandaset_pinhole_cameras(
                source_log_path,
                iteration,
                camera_poses,
                camera_timestamps_s,
                pinhole_cameras_metadata,
            )
            parsed_lidar = _extract_pandaset_lidar(source_log_path, iteration, timestamp, lidar_merged_metadata)

            yield ModalitiesSync(
                timestamp=timestamp,
                modalities=[
                    ego_state,
                    box_detections,
                    parsed_lidar,
                    *parsed_cameras,
                ],
            )

    def iter_modalities_async(self) -> Iterator[BaseModality]:
        """Not implemented — Pandaset parser only supports sync conversion."""
        raise NotImplementedError("Pandaset parser only supports sync conversion via iter_modalities_sync().")


def _get_pandaset_camera_metadata(source_log_path: Path) -> Optional[Dict[CameraID, PinholeCameraMetadata]]:
    """Extracts the pinhole camera metadata from a Pandaset log folder."""
    all_cameras_folder = source_log_path / "camera"
    if not all_cameras_folder.exists():
        return None

    camera_metadata: Dict[CameraID, PinholeCameraMetadata] = {}
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

    return camera_metadata if camera_metadata else None


def _extract_pandaset_sensor_ego_state(
    lidar_pose: Dict[str, Dict[str, float]],
    ego_metadata: EgoStateSE3Metadata,
    timestamp: Timestamp,
) -> EgoStateSE3:
    """Extracts the ego state from PandaSet lidar pose data."""
    imu_se3 = global_main_lidar_to_global_imu(pandaset_pose_dict_to_pose_se3(lidar_pose))

    return EgoStateSE3.from_imu(
        imu_se3=imu_se3,
        metadata=ego_metadata,
        dynamic_state_se3=None,
        timestamp=timestamp,
    )


def _extract_pandaset_box_detections(
    source_log_path: Path,
    iteration: int,
    timestamp: Timestamp,
    box_detections_se3_metadata: BoxDetectionsSE3Metadata,
) -> BoxDetectionsSE3:
    """Extracts the box detections from a Pandaset log folder at a given iteration."""

    # NOTE @DanielDauner: The following provided cuboids annotations are not stored in 123D
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
        return BoxDetectionsSE3(box_detections=[], timestamp=timestamp, metadata=box_detections_se3_metadata)

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
            attributes=BoxDetectionAttributes(
                label=pandaset_box_detection_label,
                track_token=box_uuids[box_idx],
            ),
            bounding_box_se3=BoundingBoxSE3.from_array(box_se3_array[box_idx]),
            velocity_3d=None,
        )
        box_detections.append(box_detection_se3)

    return BoxDetectionsSE3(box_detections=box_detections, timestamp=timestamp, metadata=box_detections_se3_metadata)  # type: ignore


def _extract_pandaset_pinhole_cameras(
    source_log_path: Path,
    iteration: int,
    camera_poses: Dict[str, List[Dict[str, Dict[str, float]]]],
    camera_timestamps_s: Dict[str, List[float]],
    pinhole_cameras_metadata: Optional[Dict[CameraID, PinholeCameraMetadata]],
) -> List[ParsedCamera]:
    """Extracts the pinhole camera data from a PandaSet scene at a given iteration.

    PandaSet provides per-frame global camera poses directly in ``camera/{name}/poses.json``,
    so we use those as ``camera_to_global_se3`` without any intermediate transforms.
    """
    if pinhole_cameras_metadata is None:
        return []

    camera_data_list: List[ParsedCamera] = []
    iteration_str = f"{iteration:02d}"

    for camera_name, camera_type in PANDASET_CAMERA_MAPPING.items():
        image_abs_path = source_log_path / f"camera/{camera_name}/{iteration_str}.jpg"
        assert image_abs_path.exists(), f"Camera image file {str(image_abs_path)} does not exist."

        camera_to_global_se3 = pandaset_pose_dict_to_pose_se3(camera_poses[camera_name][iteration])
        camera_timestamp = Timestamp.from_s(camera_timestamps_s[camera_name][iteration])

        camera_data_list.append(
            ParsedCamera(
                metadata=pinhole_cameras_metadata[camera_type],
                timestamp=camera_timestamp,
                camera_to_global_se3=camera_to_global_se3,
                dataset_root=source_log_path.parent,
                relative_path=image_abs_path.relative_to(source_log_path.parent),
            )
        )

    return camera_data_list


def _extract_pandaset_lidar(
    source_log_path: Path,
    iteration: int,
    timestamp: Timestamp,
    lidar_merged_metadata: LidarMergedMetadata,
) -> ParsedLidar:
    """Extracts the Lidar data from a Pandaset scene at a given iteration."""
    iteration_str = f"{iteration:02d}"
    lidar_absolute_path = source_log_path / "lidar" / f"{iteration_str}.pkl.gz"
    assert lidar_absolute_path.exists(), f"Lidar file {str(lidar_absolute_path)} does not exist."

    return ParsedLidar(
        metadata=lidar_merged_metadata,
        start_timestamp=timestamp,
        end_timestamp=Timestamp.from_us(
            timestamp.time_us + 100_000
        ),  # NOTE: Pandaset lidars have a frequency of 10Hz, i.e. 100ms between frames
        iteration=iteration,
        dataset_root=source_log_path.parent,
        relative_path=str(lidar_absolute_path.relative_to(source_log_path.parent)),
    )
