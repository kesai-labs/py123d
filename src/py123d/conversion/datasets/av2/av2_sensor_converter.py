from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

from py123d.api import AbstractLogWriter, AbstractMapWriter, CameraData, LidarData
from py123d.conversion.abstract_dataset_converter import AbstractDatasetConverter
from py123d.conversion.dataset_converter_config import DatasetConverterConfig
from py123d.conversion.datasets.av2.av2_map_conversion import convert_av2_map
from py123d.conversion.datasets.av2.utils.av2_constants import AV2_CAMERA_ID_MAPPING, AV2_SENSOR_SPLITS
from py123d.conversion.datasets.av2.utils.av2_helper import (
    build_sensor_dataframe,
    build_synchronization_dataframe,
    find_closest_target_fpath,
    get_slice_with_timestamp_ns,
)
from py123d.conversion.registry import AV2SensorBoxDetectionLabel
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
from py123d.datatypes.vehicle_state.vehicle_parameters import get_av2_ford_fusion_hybrid_parameters
from py123d.geometry import BoundingBoxSE3, BoundingBoxSE3Index, PoseSE3, Vector3D, Vector3DIndex
from py123d.geometry.transform import reframe_se3_array, rel_to_abs_se3_array


class AV2SensorConverter(AbstractDatasetConverter):
    """Dataset converter for the AV2 sensor dataset."""

    def __init__(
        self,
        splits: List[str],
        av2_data_root: Union[Path, str],
        dataset_converter_config: DatasetConverterConfig,
        lidar_camera_matching: Literal["nearest", "sweep"] = "sweep",
    ) -> None:
        """Initializes the AV2SensorConverter.

        :param splits: List of dataset splits to convert, e.g. ["av2-sensor_train", "av2-sensor_val", "av2-sensor_test"]
        :param av2_data_root: Root directory of the AV2 sensor dataset.
        :param dataset_converter_config: Configuration for the dataset converter.
        :param lidar_camera_matching: Criterion for matching lidar-to-camera timestamps. "sweep" (default) uses
            forward/backward matching to find cameras captured during the lidar sweep window. "nearest" finds the
            closest camera frame regardless of temporal ordering.
        """
        super().__init__(dataset_converter_config)
        assert av2_data_root is not None, "The variable `av2_data_root` must be provided."
        assert Path(av2_data_root).exists(), f"The provided `av2_data_root` path {av2_data_root} does not exist."
        for split in splits:
            assert split in AV2_SENSOR_SPLITS, f"Split {split} is not available. Available splits: {AV2_SENSOR_SPLITS}"

        self._splits: List[str] = splits
        self._av2_data_root: Path = Path(av2_data_root)
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

    def get_number_of_maps(self) -> int:
        """Inherited, see superclass."""
        return len(self._log_paths_and_split)

    def get_number_of_logs(self) -> int:
        """Inherited, see superclass."""
        return len(self._log_paths_and_split)

    def convert_map(self, map_index: int, map_writer: AbstractMapWriter) -> None:
        """Inherited, see superclass."""

        source_log_path, split = self._log_paths_and_split[map_index]

        # 1. Initialize map metadata
        map_metadata = _get_av2_sensor_map_metadata(split, source_log_path)

        # 2. Prepare map writer
        map_needs_writing = map_writer.reset(self.dataset_converter_config, map_metadata)

        # 3. Process source map data
        if map_needs_writing:
            convert_av2_map(source_log_path, map_writer)

        # 4. Finalize map writing
        map_writer.close()

    def convert_log(self, log_index: int, log_writer: AbstractLogWriter) -> None:
        """Inherited, see superclass."""

        source_log_path, split = self._log_paths_and_split[log_index]

        # 1. Initialize Metadata
        map_metadata = _get_av2_sensor_map_metadata(split, source_log_path)
        log_metadata = LogMetadata(
            dataset="av2-sensor",
            split=split,
            log_name=source_log_path.name,
            location=map_metadata.location,
            timestep_seconds=0.1,
            box_detection_label_class=AV2SensorBoxDetectionLabel,
            vehicle_parameters=get_av2_ford_fusion_hybrid_parameters(),
            pinhole_camera_metadata=_get_av2_pinhole_camera_metadata(source_log_path, self.dataset_converter_config),
            lidar_metadata=_get_av2_lidar_metadata(source_log_path, self.dataset_converter_config),
            map_metadata=map_metadata,
        )

        # 2. Prepare log writer
        log_needs_writing = log_writer.reset(self.dataset_converter_config, log_metadata)

        # 3. Process source log data
        if log_needs_writing:
            sensor_df = build_sensor_dataframe(source_log_path)
            synchronization_df = build_synchronization_dataframe(
                sensor_df,
                camera_camera_matching="nearest",
                lidar_camera_matching=self._lidar_camera_matching,
            )

            lidar_sensor = sensor_df.xs(key="lidar", level=2)
            lidar_timestamps_ns = np.sort([int(idx_tuple[2]) for idx_tuple in lidar_sensor.index])

            annotations_df = (
                pd.read_feather(source_log_path / "annotations.feather")
                if (source_log_path / "annotations.feather").exists()
                else None
            )
            city_se3_egovehicle_df = pd.read_feather(source_log_path / "city_SE3_egovehicle.feather")
            egovehicle_se3_sensor_df = pd.read_feather(
                source_log_path / "calibration" / "egovehicle_SE3_sensor.feather"
            )

            for lidar_timestamp_ns in lidar_timestamps_ns:
                ego_state = _extract_av2_sensor_ego_state(city_se3_egovehicle_df, lidar_timestamp_ns)
                log_writer.write(
                    timestamp=Timestamp.from_ns(int(lidar_timestamp_ns)),
                    ego_state_se3=ego_state,
                    box_detections_se3=_extract_av2_sensor_box_detections(
                        annotations_df,
                        lidar_timestamp_ns,
                        ego_state,
                    ),
                    pinhole_cameras=_extract_av2_sensor_pinhole_cameras(
                        lidar_timestamp_ns,
                        egovehicle_se3_sensor_df,
                        city_se3_egovehicle_df,
                        synchronization_df,
                        source_log_path,
                        self.dataset_converter_config,
                    ),
                    lidar=_extract_av2_sensor_lidars(
                        source_log_path,
                        lidar_timestamp_ns,
                        self.dataset_converter_config,
                    )[0],
                )

        # 4. Finalize log writing
        log_writer.close()


def _get_av2_sensor_map_metadata(split: str, source_log_path: Path) -> MapMetadata:
    """Helper to get map metadata for AV2 sensor dataset."""
    # NOTE: We need to get the city name from the map folder.
    # see: https://github.com/argoverse/av2-api/blob/main/src/av2/datasets/sensor/av2_sensor_dataloader.py#L163

    map_folder = source_log_path / "map"
    log_map_archive_path = next(map_folder.glob("log_map_archive_*.json"), None)
    assert log_map_archive_path is not None, f"Log map archive file not found in {map_folder}."
    location = log_map_archive_path.name.split("____")[1].split("_")[0]
    return MapMetadata(
        dataset="av2-sensor",
        split=split,
        log_name=source_log_path.name,
        location=location,  # TODO: Add location information, e.g. city name.
        map_has_z=True,
        map_is_local=True,
    )


def _get_av2_pinhole_camera_metadata(
    source_log_path: Path,
    dataset_converter_config: DatasetConverterConfig,
) -> Dict[PinholeCameraID, PinholeCameraMetadata]:
    """Helper to get pinhole camera metadata for AV2 sensor dataset."""
    pinhole_camera_metadata: Dict[PinholeCameraID, PinholeCameraMetadata] = {}
    if dataset_converter_config.include_pinhole_cameras:
        intrinsics_file = source_log_path / "calibration" / "intrinsics.feather"
        intrinsics_df = pd.read_feather(intrinsics_file)

        egovehicle_se3_sensor_file = source_log_path / "calibration" / "egovehicle_SE3_sensor.feather"
        egovehicle_se3_sensor_df = pd.read_feather(egovehicle_se3_sensor_file)

        for _, row_callib in egovehicle_se3_sensor_df.iterrows():
            row_callib = row_callib.to_dict()
            if row_callib["sensor_name"] in AV2_CAMERA_ID_MAPPING.keys():
                row_intrinsics = (
                    intrinsics_df[intrinsics_df["sensor_name"] == row_callib["sensor_name"]].iloc[0].to_dict()
                )
                camera_id = AV2_CAMERA_ID_MAPPING[row_callib["sensor_name"]]
                pinhole_camera_metadata[camera_id] = PinholeCameraMetadata(
                    camera_name=str(row_callib["sensor_name"]),
                    camera_id=camera_id,
                    width=row_intrinsics["width_px"],
                    height=row_intrinsics["height_px"],
                    intrinsics=PinholeIntrinsics(
                        fx=row_intrinsics["fx_px"],
                        fy=row_intrinsics["fy_px"],
                        cx=row_intrinsics["cx_px"],
                        cy=row_intrinsics["cy_px"],
                    ),
                    distortion=PinholeDistortion(
                        k1=row_intrinsics["k1"],
                        k2=row_intrinsics["k2"],
                        p1=0.0,
                        p2=0.0,
                        k3=row_intrinsics["k3"],
                    ),
                    camera_to_imu_se3=_row_dict_to_pose_se3(row_callib),
                    is_undistorted=True,
                )

    return pinhole_camera_metadata


def _get_av2_lidar_metadata(
    source_log_path: Path, dataset_converter_config: DatasetConverterConfig
) -> Dict[LidarID, LidarMetadata]:
    """Helper to get Lidar metadata for AV2 sensor dataset."""

    metadata: Dict[LidarID, LidarMetadata] = {}
    if dataset_converter_config.include_lidars:
        # Load calibration feather file
        calibration_file = source_log_path / "calibration" / "egovehicle_SE3_sensor.feather"
        calibration_df = pd.read_feather(calibration_file)

        # NOTE: AV2 has two two stacked lidars: up_lidar and down_lidar.
        # We store these as separate LidarID entries.

        # top lidar:
        metadata[LidarID.LIDAR_TOP] = LidarMetadata(
            lidar_name="up_lidar",
            lidar_id=LidarID.LIDAR_TOP,
            lidar_to_imu_se3=_row_dict_to_pose_se3(
                calibration_df[calibration_df["sensor_name"] == "up_lidar"].iloc[0].to_dict()
            ),
        )
        # down lidar:
        metadata[LidarID.LIDAR_DOWN] = LidarMetadata(
            lidar_name="down_lidar",
            lidar_id=LidarID.LIDAR_DOWN,
            lidar_to_imu_se3=_row_dict_to_pose_se3(
                calibration_df[calibration_df["sensor_name"] == "down_lidar"].iloc[0].to_dict()
            ),
        )
    return metadata


def _extract_av2_sensor_box_detections(
    annotations_df: Optional[pd.DataFrame], lidar_timestamp_ns: int, ego_state_se3: EgoStateSE3
) -> BoxDetectionsSE3:
    """Extract box detections from AV2 sensor dataset annotations."""

    # TODO: Extract velocity from annotations_df if available.

    if annotations_df is None:
        return BoxDetectionsSE3(box_detections=[])

    annotations_slice = get_slice_with_timestamp_ns(annotations_df, lidar_timestamp_ns)
    num_detections = len(annotations_slice)

    detections_state = np.zeros((num_detections, len(BoundingBoxSE3Index)), dtype=np.float64)
    detections_velocity = np.zeros((num_detections, len(Vector3DIndex)), dtype=np.float64)
    detections_token: List[str] = annotations_slice["track_uuid"].tolist()
    detections_labels: List[AV2SensorBoxDetectionLabel] = []
    detections_num_lidar_points: List[int] = []

    for detection_idx, (_, row) in enumerate(annotations_slice.iterrows()):
        row = row.to_dict()
        detections_state[detection_idx, BoundingBoxSE3Index.XYZ] = [row["tx_m"], row["ty_m"], row["tz_m"]]
        detections_state[detection_idx, BoundingBoxSE3Index.QUATERNION] = [row["qw"], row["qx"], row["qy"], row["qz"]]
        detections_state[detection_idx, BoundingBoxSE3Index.EXTENT] = [row["length_m"], row["width_m"], row["height_m"]]
        detections_labels.append(AV2SensorBoxDetectionLabel.deserialize(row["category"]))  # type: ignore
        detections_num_lidar_points.append(int(row["num_interior_pts"]))

    detections_state[:, BoundingBoxSE3Index.SE3] = rel_to_abs_se3_array(
        origin=ego_state_se3.rear_axle_se3,
        pose_se3_array=detections_state[:, BoundingBoxSE3Index.SE3],
    )

    box_detections: List[BoxDetectionSE3] = []
    for detection_idx in range(num_detections):
        box_detections.append(
            BoxDetectionSE3(
                metadata=BoxDetectionMetadata(
                    label=detections_labels[detection_idx],
                    track_token=detections_token[detection_idx],
                    num_lidar_points=detections_num_lidar_points[detection_idx],
                ),
                bounding_box_se3=BoundingBoxSE3.from_array(detections_state[detection_idx]),
                velocity_3d=Vector3D.from_array(detections_velocity[detection_idx]),
            )
        )

    return BoxDetectionsSE3(box_detections=box_detections)  # type: ignore


def _extract_av2_sensor_ego_state(city_se3_egovehicle_df: pd.DataFrame, lidar_timestamp_ns: int) -> EgoStateSE3:
    """Extract ego state from AV2 sensor dataset city_SE3_egovehicle dataframe."""
    ego_state_slice = get_slice_with_timestamp_ns(city_se3_egovehicle_df, lidar_timestamp_ns)
    assert len(ego_state_slice) == 1, (
        f"Expected exactly one ego state for timestamp {lidar_timestamp_ns}, got {len(ego_state_slice)}."
    )
    ego_pose_dict = ego_state_slice.iloc[0].to_dict()
    ego_imu_se3 = _row_dict_to_pose_se3(ego_pose_dict)
    return EgoStateSE3.from_imu(
        imu_se3=ego_imu_se3,
        vehicle_parameters=get_av2_ford_fusion_hybrid_parameters(),
        dynamic_state_se3=None,
        timestamp=Timestamp.from_ns(lidar_timestamp_ns),
    )


def _extract_av2_sensor_pinhole_cameras(
    lidar_timestamp_ns: int,
    egovehicle_se3_sensor_df: pd.DataFrame,
    city_se3_egovehicle_df: pd.DataFrame,
    synchronization_df: pd.DataFrame,
    source_log_path: Path,
    dataset_converter_config: DatasetConverterConfig,
) -> List[CameraData]:
    """Extract pinhole camera data from AV2 sensor dataset."""

    camera_data_list: List[CameraData] = []
    split = source_log_path.parent.name
    log_id = source_log_path.name

    if dataset_converter_config.include_pinhole_cameras:
        av2_sensor_data_root = source_log_path.parent.parent

        current_ego_pose_slice = get_slice_with_timestamp_ns(city_se3_egovehicle_df, lidar_timestamp_ns)
        assert len(current_ego_pose_slice) == 1
        current_ego_pose_se3 = _row_dict_to_pose_se3(current_ego_pose_slice.iloc[0].to_dict())

        for _, row in egovehicle_se3_sensor_df.iterrows():
            row = row.to_dict()
            if row["sensor_name"] not in AV2_CAMERA_ID_MAPPING:
                continue
            camera_to_imu_se3 = _row_dict_to_pose_se3(row)
            pinhole_camera_name = row["sensor_name"]
            pinhole_camera_id = AV2_CAMERA_ID_MAPPING[pinhole_camera_name]

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

                # Motion compensation of the camera extrinsic to the lidar timestamp:
                nearest_pose = (
                    get_slice_with_timestamp_ns(city_se3_egovehicle_df, int(timestamp_ns_str)).iloc[0].to_dict()
                )
                nearest_pose_se3 = _row_dict_to_pose_se3(nearest_pose)
                compensated_extrinsic_se3_array = reframe_se3_array(
                    from_origin=nearest_pose_se3,
                    to_origin=current_ego_pose_se3,
                    pose_se3_array=camera_to_imu_se3.array,
                )
                camera_data = CameraData(
                    camera_name=str(pinhole_camera_name),
                    camera_id=pinhole_camera_id,
                    timestamp=Timestamp.from_ns(int(timestamp_ns_str)),
                    extrinsic=PoseSE3.from_array(compensated_extrinsic_se3_array),
                    dataset_root=av2_sensor_data_root,
                    relative_path=relative_image_path,
                )
                camera_data_list.append(camera_data)

    return camera_data_list


def _extract_av2_sensor_lidars(
    source_log_path: Path,
    lidar_timestamp_ns: int,
    dataset_converter_config: DatasetConverterConfig,
) -> List[LidarData]:
    """Extract Lidar data from AV2 sensor dataset."""
    lidars: List[LidarData] = []
    if dataset_converter_config.include_lidars:
        av2_sensor_data_root = source_log_path.parent.parent
        split_type = source_log_path.parent.name
        log_name = source_log_path.name

        relative_feather_path = f"{split_type}/{log_name}/sensors/lidar/{lidar_timestamp_ns}.feather"
        lidar_feather_path = av2_sensor_data_root / relative_feather_path
        assert lidar_feather_path.exists(), f"Lidar feather file not found: {lidar_feather_path}"

        lidar_data = LidarData(
            lidar_name=LidarID.LIDAR_MERGED.serialize(),
            lidar_type=LidarID.LIDAR_MERGED,
            dataset_root=av2_sensor_data_root,
            relative_path=relative_feather_path,
        )
        lidars.append(lidar_data)

    return lidars


def _row_dict_to_pose_se3(row_dict: Dict[str, float]) -> PoseSE3:
    """Helper function to convert a row dictionary to a PoseSE3 object."""
    return PoseSE3(
        x=row_dict["tx_m"],
        y=row_dict["ty_m"],
        z=row_dict["tz_m"],
        qw=row_dict["qw"],
        qx=row_dict["qx"],
        qy=row_dict["qy"],
        qz=row_dict["qz"],
    )
