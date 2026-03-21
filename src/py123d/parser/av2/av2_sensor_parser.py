from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

from py123d.datatypes import (
    BoxDetectionAttributes,
    BoxDetectionSE3,
    BoxDetectionsSE3,
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
from py123d.datatypes.modalities.base_modality import BaseModality
from py123d.datatypes.sensors.lidar import LidarMergedMetadata
from py123d.datatypes.sensors.pinhole_camera import CameraID
from py123d.datatypes.vehicle_state.ego_state_metadata import EgoStateSE3Metadata
from py123d.geometry import BoundingBoxSE3, BoundingBoxSE3Index, Vector3D, Vector3DIndex
from py123d.geometry.transform import rel_to_abs_se3_array
from py123d.geometry.transform.transform_se3 import rel_to_abs_se3
from py123d.parser.av2.av2_map_parser import Av2MapParser, get_av2_map_metadata
from py123d.parser.av2.utils.av2_constants import (
    AV2_CAMERA_ID_MAPPING,
    AV2_SENSOR_BOX_DETECTIONS_SE3_METADATA,
    AV2_SENSOR_EGO_STATE_SE3_METADATA,
    AV2_SENSOR_SPLITS,
)
from py123d.parser.av2.utils.av2_helper import (
    av2_row_dict_to_pose_se3,
    build_sensor_dataframe,
    build_synchronization_dataframe,
    find_closest_target_fpath,
    get_slice_with_timestamp_ns,
)
from py123d.parser.base_dataset_parser import (
    BaseDatasetParser,
    BaseLogParser,
    ModalitiesSync,
    ParsedCamera,
    ParsedLidar,
)
from py123d.parser.registry import AV2SensorBoxDetectionLabel


class Av2SensorParser(BaseDatasetParser):
    """Dataset parser for the AV2 sensor dataset."""

    def __init__(
        self,
        splits: List[str],
        av2_data_root: Union[Path, str],
        lidar_camera_matching: Literal["nearest", "sweep"] = "sweep",
    ) -> None:
        """Initializes the Av2SensorParser.

        :param splits: List of dataset splits, e.g. ["av2-sensor_train", "av2-sensor_val"].
        :param av2_data_root: Root directory of the AV2 sensor dataset.
        :param lidar_camera_matching: Criterion for matching lidar-to-camera timestamps.
        """
        assert av2_data_root is not None, "The variable `av2_data_root` must be provided."
        assert Path(av2_data_root).exists(), f"The provided `av2_data_root` path {av2_data_root} does not exist."
        for split in splits:
            assert split in AV2_SENSOR_SPLITS, f"Split {split} is not available. Available splits: {AV2_SENSOR_SPLITS}"

        self._splits = splits
        self._av2_data_root = Path(av2_data_root)
        self._lidar_camera_matching: Literal["nearest", "sweep"] = lidar_camera_matching
        self._log_paths_and_split: List[Tuple[Path, str]] = self._collect_log_paths()

    def _collect_log_paths(self) -> List[Tuple[Path, str]]:
        """Collects source log folder paths for the specified splits."""
        log_paths_and_split: List[Tuple[Path, str]] = []
        for split in self._splits:
            dataset_name = split.split("_")[0]
            split_type = split.split("_")[-1]
            assert split_type in {"train", "val", "test"}, f"Split type {split_type} is not valid."
            if "av2-sensor" == dataset_name:
                log_folder = self._av2_data_root / "sensor" / split_type
            else:
                raise ValueError(f"Unknown dataset name {dataset_name} in split {split}.")
            log_paths_and_split.extend([(log_path, split) for log_path in log_folder.iterdir()])
        return log_paths_and_split

    def get_log_parsers(self) -> List[Av2SensorLogParser]:  # type: ignore[override]
        """Inherited, see superclass."""
        return [
            Av2SensorLogParser(
                source_log_path=source_log_path,
                split=split,
                lidar_camera_matching=self._lidar_camera_matching,
            )
            for source_log_path, split in self._log_paths_and_split
        ]

    def get_map_parsers(self) -> List[Av2MapParser]:  # type: ignore[override]
        """Inherited, see superclass."""
        return [
            Av2MapParser(source_log_path=source_log_path, split=split, dataset="av2-sensor")
            for source_log_path, split in self._log_paths_and_split
        ]


class Av2SensorLogParser(BaseLogParser):
    """Lightweight, picklable handle to one AV2 sensor log."""

    def __init__(
        self,
        source_log_path: Path,
        split: str,
        lidar_camera_matching: Literal["nearest", "sweep"],
    ) -> None:
        self._source_log_path = source_log_path
        self._split = split
        self._lidar_camera_matching: Literal["nearest", "sweep"] = lidar_camera_matching

    def get_log_metadata(self) -> LogMetadata:
        """Inherited, see superclass."""
        map_metadata = get_av2_map_metadata(self._split, self._source_log_path, dataset="av2-sensor")
        return LogMetadata(
            dataset="av2-sensor",
            split=self._split,
            log_name=self._source_log_path.name,
            location=map_metadata.location,
            map_metadata=map_metadata,
        )

    def iter_modalities_sync(self) -> Iterator[ModalitiesSync]:
        """Inherited, see superclass."""

        ego_state_se3_metadata = AV2_SENSOR_EGO_STATE_SE3_METADATA
        box_detections_se3_metadata = AV2_SENSOR_BOX_DETECTIONS_SE3_METADATA
        pinhole_camera_metadatas = _get_av2_pinhole_camera_metadatas(self._source_log_path)
        lidar_merged_metadata = _get_av2_lidar_merged_metadata(self._source_log_path)

        sensor_df = build_sensor_dataframe(self._source_log_path)
        synchronization_df = build_synchronization_dataframe(
            sensor_df,
            camera_camera_matching="nearest",
            lidar_camera_matching=self._lidar_camera_matching,
        )

        lidar_sensor = sensor_df.xs(key="lidar", level=2)
        lidar_timestamps_ns = np.sort([int(idx_tuple[2]) for idx_tuple in lidar_sensor.index])

        annotations_df = (
            pd.read_feather(self._source_log_path / "annotations.feather")
            if (self._source_log_path / "annotations.feather").exists()
            else None
        )
        city_se3_egovehicle_df = pd.read_feather(self._source_log_path / "city_SE3_egovehicle.feather")
        egovehicle_se3_sensor_df = pd.read_feather(
            self._source_log_path / "calibration" / "egovehicle_SE3_sensor.feather"
        )

        for lidar_timestamp_ns in lidar_timestamps_ns:
            timestamp = Timestamp.from_ns(int(lidar_timestamp_ns))
            ego_state_se3 = _extract_av2_sensor_ego_state(
                city_se3_egovehicle_df, lidar_timestamp_ns, ego_state_se3_metadata
            )
            annotations_slice = (
                get_slice_with_timestamp_ns(annotations_df, lidar_timestamp_ns) if annotations_df is not None else None
            )
            box_detections_se3 = _extract_av2_sensor_box_detections(
                annotations_slice,
                lidar_timestamp_ns,
                ego_state_se3,
                box_detections_se3_metadata,
            )
            parsed_pinhole_cameras = _extract_av2_sensor_pinhole_cameras(
                lidar_timestamp_ns,
                egovehicle_se3_sensor_df,
                city_se3_egovehicle_df,
                synchronization_df,
                self._source_log_path,
                pinhole_camera_metadatas,
            )
            parsed_lidar = _extract_av2_sensor_lidar(
                self._source_log_path,
                lidar_timestamp_ns,
                lidar_merged_metadata,
            )
            yield ModalitiesSync(
                timestamp=timestamp,
                modalities=[
                    ego_state_se3,
                    box_detections_se3,
                    parsed_lidar,
                    *parsed_pinhole_cameras,
                ],
            )

    def iter_modalities_async(self) -> Iterator[BaseModality]:
        """Inherited, see superclass."""

        ego_state_se3_metadata = AV2_SENSOR_EGO_STATE_SE3_METADATA
        box_detections_se3_metadata = AV2_SENSOR_BOX_DETECTIONS_SE3_METADATA
        pinhole_camera_metadatas = _get_av2_pinhole_camera_metadatas(self._source_log_path)
        lidar_merged_metadata = _get_av2_lidar_merged_metadata(self._source_log_path)

        yield from self._iter_ego_states_se3(ego_state_se3_metadata)
        yield from self._iter_box_detections_se3(box_detections_se3_metadata)
        yield from self._iter_lidar_merged(lidar_merged_metadata)
        for pinhole_camera_metadata in pinhole_camera_metadatas.values():
            yield from self._iter_pinhole_camera(pinhole_camera_metadata)

    # ------------------------------------------------------------------------------------------------------------------
    # Per-modality iterators (async / native-rate)
    # ------------------------------------------------------------------------------------------------------------------

    def _iter_ego_states_se3(self, modality_metadata: EgoStateSE3Metadata) -> Iterator[EgoStateSE3]:
        """Yields all ego state observations at native rate from city_SE3_egovehicle.feather."""
        city_se3_egovehicle_df = pd.read_feather(self._source_log_path / "city_SE3_egovehicle.feather")

        for _, row in city_se3_egovehicle_df.iterrows():
            row_dict = row.to_dict()
            ego_imu_se3 = av2_row_dict_to_pose_se3(row_dict)
            yield EgoStateSE3.from_imu(
                imu_se3=ego_imu_se3,
                metadata=modality_metadata,
                dynamic_state_se3=None,
                timestamp=Timestamp.from_ns(int(row_dict["timestamp_ns"])),
            )

    def _iter_box_detections_se3(self, modality_metadata: BoxDetectionsSE3Metadata) -> Iterator[BoxDetectionsSE3]:
        """Yields box detections at each annotated timestamp."""
        if not (self._source_log_path / "annotations.feather").exists():
            return

        ego_metadata = AV2_SENSOR_EGO_STATE_SE3_METADATA
        annotations_df = pd.read_feather(self._source_log_path / "annotations.feather")
        city_se3_egovehicle_df = pd.read_feather(self._source_log_path / "city_SE3_egovehicle.feather")

        for timestamp_ns_key, group_df in annotations_df.groupby("timestamp_ns"):
            timestamp_ns = int(timestamp_ns_key)  # type: ignore[arg-type]
            ego_state = _extract_av2_sensor_ego_state(city_se3_egovehicle_df, timestamp_ns, ego_metadata)
            yield _extract_av2_sensor_box_detections(group_df, timestamp_ns, ego_state, modality_metadata)

    def _iter_pinhole_camera(self, pinhole_camera_metadata: PinholeCameraMetadata) -> Iterator[ParsedCamera]:
        """Yields pinhole camera observations for a specific camera at native rate (~20Hz)."""
        target_camera_name = pinhole_camera_metadata.camera_name

        egovehicle_se3_sensor_df = pd.read_feather(
            self._source_log_path / "calibration" / "egovehicle_SE3_sensor.feather"
        )
        city_se3_egovehicle_df = pd.read_feather(self._source_log_path / "city_SE3_egovehicle.feather")
        av2_sensor_data_root = self._source_log_path.parent.parent
        split_type = self._source_log_path.parent.name
        log_name = self._source_log_path.name

        # Find the calibration row for this specific camera
        camera_row = egovehicle_se3_sensor_df[egovehicle_se3_sensor_df["sensor_name"] == target_camera_name]
        if camera_row.empty:
            return

        camera_dir = self._source_log_path / "sensors" / "cameras" / target_camera_name
        if camera_dir.exists():
            image_files = sorted(camera_dir.glob("*.jpg"))
            for image_file in image_files:
                timestamp_ns = int(image_file.stem)
                relative_path = f"{split_type}/{log_name}/sensors/cameras/{target_camera_name}/{image_file.name}"

                ego_pose_slice = get_slice_with_timestamp_ns(city_se3_egovehicle_df, timestamp_ns)
                if ego_pose_slice.empty:
                    continue  # Skip images without a matching ego pose
                nearest_pose = ego_pose_slice.iloc[0].to_dict()
                nearest_pose_se3 = av2_row_dict_to_pose_se3(nearest_pose)  # type: ignore[arg-type]
                camera_to_global_se3 = rel_to_abs_se3(
                    origin=nearest_pose_se3,
                    pose_se3=pinhole_camera_metadata.camera_to_imu_se3,
                )

                yield ParsedCamera(
                    metadata=pinhole_camera_metadata,
                    timestamp=Timestamp.from_ns(timestamp_ns),
                    camera_to_global_se3=camera_to_global_se3,
                    dataset_root=av2_sensor_data_root,
                    relative_path=relative_path,
                )

    def _iter_lidar_merged(self, modality_metadata: LidarMergedMetadata) -> Iterator[ParsedLidar]:
        """Yields all lidar sweeps at native rate (~10Hz)."""
        lidar_dir = self._source_log_path / "sensors" / "lidar"
        av2_sensor_data_root = self._source_log_path.parent.parent
        split_type = self._source_log_path.parent.name
        log_name = self._source_log_path.name

        feather_files = sorted(lidar_dir.glob("*.feather"))
        for feather_file in feather_files:
            timestamp_ns = int(feather_file.stem)
            relative_path = f"{split_type}/{log_name}/sensors/lidar/{feather_file.name}"

            start_timestamp_ns = timestamp_ns
            end_timestamp_ns = timestamp_ns + 100_000_000  # Assume each sweep covers 100ms, consistent with AV2 API

            yield ParsedLidar(
                metadata=modality_metadata,
                start_timestamp=Timestamp.from_ns(start_timestamp_ns),
                end_timestamp=Timestamp.from_ns(end_timestamp_ns),
                dataset_root=av2_sensor_data_root,
                relative_path=relative_path,
            )


# ------------------------------------------------------------------------------------------------------------------
# Metadata helpers
# ------------------------------------------------------------------------------------------------------------------


def _get_av2_pinhole_camera_metadatas(source_log_path: Path) -> Dict[CameraID, PinholeCameraMetadata]:
    """Returns a list of pinhole camera metadata for AV2 sensor dataset (one per camera)."""
    metadatas: Dict[CameraID, PinholeCameraMetadata] = {}
    intrinsics_file = source_log_path / "calibration" / "intrinsics.feather"
    intrinsics_df = pd.read_feather(intrinsics_file)

    egovehicle_se3_sensor_file = source_log_path / "calibration" / "egovehicle_SE3_sensor.feather"
    egovehicle_se3_sensor_df = pd.read_feather(egovehicle_se3_sensor_file)

    for _, calibration_row in egovehicle_se3_sensor_df.iterrows():
        calibration_dict = calibration_row.to_dict()
        if calibration_dict["sensor_name"] in AV2_CAMERA_ID_MAPPING.keys():
            intrinsics_rows = intrinsics_df[intrinsics_df["sensor_name"] == calibration_dict["sensor_name"]]
            if intrinsics_rows.empty:
                continue  # Skip cameras without intrinsics (known AV2 dataset issue)
            intrinsics_dict = intrinsics_rows.iloc[0].to_dict()
            camera_id = AV2_CAMERA_ID_MAPPING[calibration_dict["sensor_name"]]
            metadatas[camera_id] = PinholeCameraMetadata(
                camera_name=str(calibration_dict["sensor_name"]),
                camera_id=camera_id,
                width=intrinsics_dict["width_px"],
                height=intrinsics_dict["height_px"],
                intrinsics=PinholeIntrinsics(
                    fx=intrinsics_dict["fx_px"],
                    fy=intrinsics_dict["fy_px"],
                    cx=intrinsics_dict["cx_px"],
                    cy=intrinsics_dict["cy_px"],
                ),
                distortion=PinholeDistortion(
                    k1=intrinsics_dict["k1"],
                    k2=intrinsics_dict["k2"],
                    p1=0.0,
                    p2=0.0,
                    k3=intrinsics_dict["k3"],
                ),
                camera_to_imu_se3=av2_row_dict_to_pose_se3(calibration_dict),  # type: ignore[arg-type]
                is_undistorted=True,
            )

    return metadatas


def _get_av2_lidar_merged_metadata(source_log_path: Path) -> LidarMergedMetadata:
    """Helper to get Lidar metadata for AV2 sensor dataset."""
    metadata_dict: Dict[LidarID, LidarMetadata] = {}
    calibration_file = source_log_path / "calibration" / "egovehicle_SE3_sensor.feather"
    calibration_df = pd.read_feather(calibration_file)

    metadata_dict[LidarID.LIDAR_TOP] = LidarMetadata(
        lidar_name="up_lidar",
        lidar_id=LidarID.LIDAR_TOP,
        lidar_to_imu_se3=av2_row_dict_to_pose_se3(
            calibration_df[calibration_df["sensor_name"] == "up_lidar"].iloc[0].to_dict()
        ),
    )
    metadata_dict[LidarID.LIDAR_DOWN] = LidarMetadata(
        lidar_name="down_lidar",
        lidar_id=LidarID.LIDAR_DOWN,
        lidar_to_imu_se3=av2_row_dict_to_pose_se3(
            calibration_df[calibration_df["sensor_name"] == "down_lidar"].iloc[0].to_dict()
        ),
    )
    return LidarMergedMetadata(metadata_dict)


# ------------------------------------------------------------------------------------------------------------------
# Sensor extraction helpers
# ------------------------------------------------------------------------------------------------------------------


def _extract_av2_sensor_box_detections(
    annotations_slice: Optional[pd.DataFrame],
    timestamp_ns: int,
    ego_state_se3: EgoStateSE3,
    box_detections_se3_metadata: BoxDetectionsSE3Metadata,
) -> BoxDetectionsSE3:
    """Extract box detections from AV2 sensor dataset annotations.

    :param annotations_slice: Pre-sliced annotations DataFrame for a single timestamp, or None if no annotations.
    :param timestamp_ns: The timestamp in nanoseconds.
    :param ego_state_se3: The ego state at this timestamp.
    :param box_detections_se3_metadata: Metadata for the box detections.
    :return: BoxDetectionsSE3 containing all detections at this timestamp.
    """

    timestamp = Timestamp.from_ns(int(timestamp_ns))

    if annotations_slice is None or len(annotations_slice) == 0:
        return BoxDetectionsSE3(box_detections=[], timestamp=timestamp, metadata=box_detections_se3_metadata)

    num_detections = len(annotations_slice)

    detections_state = np.zeros((num_detections, len(BoundingBoxSE3Index)), dtype=np.float64)
    detections_velocity = np.zeros((num_detections, len(Vector3DIndex)), dtype=np.float64)
    detections_token: List[str] = annotations_slice["track_uuid"].tolist()
    detections_labels: List[AV2SensorBoxDetectionLabel] = []
    detections_num_lidar_points: List[int] = []

    for detection_idx, (_, detection_series) in enumerate(annotations_slice.iterrows()):
        detection = detection_series.to_dict()
        detections_state[detection_idx, BoundingBoxSE3Index.XYZ] = [
            detection["tx_m"],
            detection["ty_m"],
            detection["tz_m"],
        ]
        detections_state[detection_idx, BoundingBoxSE3Index.QUATERNION] = [
            detection["qw"],
            detection["qx"],
            detection["qy"],
            detection["qz"],
        ]
        detections_state[detection_idx, BoundingBoxSE3Index.EXTENT] = [
            detection["length_m"],
            detection["width_m"],
            detection["height_m"],
        ]
        detections_labels.append(AV2SensorBoxDetectionLabel.deserialize(detection["category"]))  # type: ignore[arg-type]
        detections_num_lidar_points.append(int(detection["num_interior_pts"]))

    detections_state[:, BoundingBoxSE3Index.SE3] = rel_to_abs_se3_array(
        origin=ego_state_se3.rear_axle_se3,
        pose_se3_array=detections_state[:, BoundingBoxSE3Index.SE3],
    )

    box_detections: List[BoxDetectionSE3] = []
    for detection_idx in range(num_detections):
        box_detections.append(
            BoxDetectionSE3(
                attributes=BoxDetectionAttributes(
                    label=detections_labels[detection_idx],
                    track_token=detections_token[detection_idx],
                    num_lidar_points=detections_num_lidar_points[detection_idx],
                ),
                bounding_box_se3=BoundingBoxSE3.from_array(detections_state[detection_idx]),
                velocity_3d=Vector3D.from_array(detections_velocity[detection_idx]),
            )
        )

    return BoxDetectionsSE3(box_detections=box_detections, timestamp=timestamp, metadata=box_detections_se3_metadata)


def _extract_av2_sensor_ego_state(
    city_se3_egovehicle_df: pd.DataFrame,
    lidar_timestamp_ns: int,
    ego_metadata: EgoStateSE3Metadata,
) -> EgoStateSE3:
    """Extract ego state from AV2 sensor dataset city_SE3_egovehicle dataframe."""
    ego_state_slice = get_slice_with_timestamp_ns(city_se3_egovehicle_df, lidar_timestamp_ns)
    assert len(ego_state_slice) == 1, (
        f"Expected exactly one ego state for timestamp {lidar_timestamp_ns}, got {len(ego_state_slice)}."
    )
    ego_pose_dict = ego_state_slice.iloc[0].to_dict()
    ego_imu_se3 = av2_row_dict_to_pose_se3(ego_pose_dict)
    return EgoStateSE3.from_imu(
        imu_se3=ego_imu_se3,
        metadata=ego_metadata,
        dynamic_state_se3=None,
        timestamp=Timestamp.from_ns(lidar_timestamp_ns),
    )


def _extract_av2_sensor_pinhole_cameras(
    lidar_timestamp_ns: int,
    egovehicle_se3_sensor_df: pd.DataFrame,
    city_se3_egovehicle_df: pd.DataFrame,
    synchronization_df: pd.DataFrame,
    source_log_path: Path,
    metadatas: Dict[CameraID, PinholeCameraMetadata],
) -> List[ParsedCamera]:
    """Extract pinhole camera data from AV2 sensor dataset."""
    camera_data_list: List[ParsedCamera] = []
    split = source_log_path.parent.name
    log_id = source_log_path.name

    av2_sensor_data_root = source_log_path.parent.parent

    current_ego_pose_slice = get_slice_with_timestamp_ns(city_se3_egovehicle_df, lidar_timestamp_ns)
    assert len(current_ego_pose_slice) == 1

    for _, sensor_series in egovehicle_se3_sensor_df.iterrows():
        sensor_dict = sensor_series.to_dict()
        if sensor_dict["sensor_name"] not in AV2_CAMERA_ID_MAPPING:
            continue
        pinhole_camera_name = sensor_dict["sensor_name"]
        pinhole_camera_id = AV2_CAMERA_ID_MAPPING[pinhole_camera_name]
        camera_metadata = metadatas[pinhole_camera_id]

        relative_image_path = find_closest_target_fpath(
            split=split,
            log_id=log_id,
            src_sensor_name="lidar",
            src_timestamp_ns=lidar_timestamp_ns,
            target_sensor_name=pinhole_camera_name,
            synchronization_df=synchronization_df,
        )
        if relative_image_path is not None:
            absolute_image_path = av2_sensor_data_root / relative_image_path
            assert absolute_image_path.exists()
            timestamp_ns_str = absolute_image_path.stem

            ego_pose_slice = get_slice_with_timestamp_ns(city_se3_egovehicle_df, int(timestamp_ns_str))
            if ego_pose_slice.empty:
                continue  # Skip images without a matching ego pose
            nearest_pose = ego_pose_slice.iloc[0].to_dict()
            nearest_pose_se3 = av2_row_dict_to_pose_se3(nearest_pose)
            camera_to_global_se3 = rel_to_abs_se3(
                origin=nearest_pose_se3,
                pose_se3=camera_metadata.camera_to_imu_se3,
            )
            camera_data = ParsedCamera(
                metadata=camera_metadata,
                timestamp=Timestamp.from_ns(int(timestamp_ns_str)),
                camera_to_global_se3=camera_to_global_se3,
                dataset_root=av2_sensor_data_root,
                relative_path=relative_image_path,
            )
            camera_data_list.append(camera_data)

    return camera_data_list


def _extract_av2_sensor_lidar(
    source_log_path: Path,
    lidar_timestamp_ns: int,
    metadata: LidarMergedMetadata,
) -> ParsedLidar:
    """Extract Lidar data from AV2 sensor dataset. Returns None if lidars not included."""
    av2_sensor_data_root = source_log_path.parent.parent
    split_type = source_log_path.parent.name
    log_name = source_log_path.name

    relative_feather_path = f"{split_type}/{log_name}/sensors/lidar/{lidar_timestamp_ns}.feather"
    lidar_feather_path = av2_sensor_data_root / relative_feather_path
    assert lidar_feather_path.exists(), f"Lidar feather file not found: {lidar_feather_path}"

    return ParsedLidar(
        metadata=metadata,
        start_timestamp=Timestamp.from_ns(int(lidar_timestamp_ns)),
        end_timestamp=Timestamp.from_ns(
            int(lidar_timestamp_ns) + 100_000_000
        ),  # Assume each sweep covers 100ms, consistent with AV2 API
        dataset_root=av2_sensor_data_root,
        relative_path=relative_feather_path,
    )
