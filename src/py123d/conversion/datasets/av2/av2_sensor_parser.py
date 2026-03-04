from __future__ import annotations

import typing
from pathlib import Path
from typing import Dict, Iterator, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

from py123d.conversion.abstract_dataset_parser import (
    CameraData,
    DatasetParser,
    FrameData,
    LidarData,
    LogParser,
)
from py123d.conversion.datasets.av2.av2_map_parser import Av2MapParser, get_av2_map_metadata
from py123d.conversion.datasets.av2.utils.av2_constants import AV2_CAMERA_ID_MAPPING, AV2_SENSOR_SPLITS
from py123d.conversion.datasets.av2.utils.av2_helper import (
    build_sensor_dataframe,
    build_synchronization_dataframe,
    find_closest_target_fpath,
    get_slice_with_timestamp_ns,
)
from py123d.conversion.registry import AV2SensorBoxDetectionLabel
from py123d.datatypes import (
    BoxDetectionAttributes,
    BoxDetectionSE3,
    BoxDetectionsSE3,
    EgoStateSE3,
    LidarID,
    LidarMetadata,
    LogMetadata,
    PinholeCameraID,
    PinholeCameraMetadata,
    PinholeDistortion,
    PinholeIntrinsics,
    Timestamp,
)
from py123d.datatypes.detections.box_detection_label_metadata import BoxDetectionMetadata
from py123d.datatypes.sensors.fisheye_mei_camera import FisheyeMEICameraID, FisheyeMEICameraMetadata
from py123d.datatypes.vehicle_state.ego_metadata import EgoMetadata, get_av2_ford_fusion_hybrid_parameters
from py123d.geometry import BoundingBoxSE3, BoundingBoxSE3Index, PoseSE3, Vector3D, Vector3DIndex
from py123d.geometry.transform import reframe_se3_array, rel_to_abs_se3_array


class Av2SensorParser(DatasetParser):
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
        self._lidar_camera_matching = lidar_camera_matching
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

    def get_log_parsers(self) -> List[Av2SensorLogParser]:  # type: ignore
        """Inherited, see superclass."""
        return [
            Av2SensorLogParser(
                source_log_path=source_log_path,
                split=split,
                lidar_camera_matching=self._lidar_camera_matching,  # type: ignore
            )
            for source_log_path, split in self._log_paths_and_split
        ]

    def get_map_parsers(self) -> List[Av2MapParser]:  # type: ignore
        """Inherited, see superclass."""
        return [
            Av2MapParser(source_log_path=source_log_path, split=split, dataset="av2-sensor")
            for source_log_path, split in self._log_paths_and_split
        ]


class Av2SensorLogParser(LogParser):
    """Lightweight, picklable handle to one AV2 sensor log."""

    def __init__(
        self,
        source_log_path: Path,
        split: str,
        lidar_camera_matching: Literal["nearest", "sweep"],
    ) -> None:
        self._source_log_path = source_log_path
        self._split = split
        self._lidar_camera_matching = lidar_camera_matching

    def get_log_metadata(self) -> LogMetadata:
        """Inherited, see superclass."""
        map_metadata = get_av2_map_metadata(self._split, self._source_log_path, dataset="av2-sensor")
        return LogMetadata(
            dataset="av2-sensor",
            split=self._split,
            log_name=self._source_log_path.name,
            location=map_metadata.location,
            timestep_seconds=0.1,
            # box_detection_label_class=AV2SensorBoxDetectionLabel,
            # vehicle_parameters=get_av2_ford_fusion_hybrid_parameters(),
            # pinhole_camera_metadata=_get_av2_pinhole_camera_metadata(self._source_log_path),
            # lidar_metadata=_get_av2_lidar_metadata(self._source_log_path),
            # map_metadata=map_metadata,
        )

    @typing.override
    def get_ego_metadata(self) -> Optional[EgoMetadata]:
        """Inherited, see superclass."""
        # [1] https://en.wikipedia.org/wiki/Ford_Fusion_Hybrid#Second_generation
        # https://github.com/argoverse/av2-api/blob/6b22766247eda941cb1953d6a58e8d5631c561da/tests/unit/map/test_map_api.py#L375
        return EgoMetadata(
            vehicle_name="av2_ford_fusion_hybrid",
            width=1.852 + 0.275,  # 0.275 is the estimated width of the side mirrors
            length=4.869,
            height=1.476,
            wheel_base=2.850,
            center_to_imu_se3=PoseSE3(x=1.339, y=0.0, z=0.438, qw=1.0, qx=0.0, qy=0.0, qz=0.0),
            rear_axle_to_imu_se3=PoseSE3.identity(),
        )

    @typing.override
    def get_box_detection_metadata(self) -> BoxDetectionMetadata:
        """Inherited, see superclass."""
        return BoxDetectionMetadata(box_detection_label_class=AV2SensorBoxDetectionLabel)

    @typing.override
    def get_pinhole_camera_metadatas(self) -> Dict[PinholeCameraID, PinholeCameraMetadata]:
        """Inherited, see superclass."""
        return _get_av2_pinhole_camera_metadata(self._source_log_path)

    @typing.override
    def get_fisheye_mei_camera_metadatas(self) -> Dict[FisheyeMEICameraID, FisheyeMEICameraMetadata]:
        """Inherited, see superclass."""
        return {}

    @typing.override
    def get_lidar_metadatas(self) -> Dict[LidarID, LidarMetadata]:
        """Inherited, see superclass."""
        return _get_av2_lidar_metadata(self._source_log_path)

    def iter_frames(self) -> Iterator[FrameData]:
        """Inherited, see superclass."""
        sensor_df = build_sensor_dataframe(self._source_log_path)
        synchronization_df = build_synchronization_dataframe(
            sensor_df,
            camera_camera_matching="nearest",
            lidar_camera_matching=self._lidar_camera_matching,  # type: ignore
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
            ego_state = _extract_av2_sensor_ego_state(city_se3_egovehicle_df, lidar_timestamp_ns)
            yield FrameData(
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
                    self._source_log_path,
                ),
                lidars=[_l]
                if (_l := _extract_av2_sensor_lidar(self._source_log_path, lidar_timestamp_ns)) is not None
                else None,
            )


# ------------------------------------------------------------------------------------------------------------------
# Sensor extraction helpers
# ------------------------------------------------------------------------------------------------------------------


def _get_av2_pinhole_camera_metadata(source_log_path: Path) -> Dict[PinholeCameraID, PinholeCameraMetadata]:
    """Helper to get pinhole camera metadata for AV2 sensor dataset."""
    pinhole_camera_metadata: Dict[PinholeCameraID, PinholeCameraMetadata] = {}
    intrinsics_file = source_log_path / "calibration" / "intrinsics.feather"
    intrinsics_df = pd.read_feather(intrinsics_file)

    egovehicle_se3_sensor_file = source_log_path / "calibration" / "egovehicle_SE3_sensor.feather"
    egovehicle_se3_sensor_df = pd.read_feather(egovehicle_se3_sensor_file)

    for _, row_callib in egovehicle_se3_sensor_df.iterrows():
        row_callib = row_callib.to_dict()
        if row_callib["sensor_name"] in AV2_CAMERA_ID_MAPPING.keys():
            row_intrinsics = intrinsics_df[intrinsics_df["sensor_name"] == row_callib["sensor_name"]].iloc[0].to_dict()
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


def _get_av2_lidar_metadata(source_log_path: Path) -> Dict[LidarID, LidarMetadata]:
    """Helper to get Lidar metadata for AV2 sensor dataset."""
    metadata: Dict[LidarID, LidarMetadata] = {}
    calibration_file = source_log_path / "calibration" / "egovehicle_SE3_sensor.feather"
    calibration_df = pd.read_feather(calibration_file)

    metadata[LidarID.LIDAR_TOP] = LidarMetadata(
        lidar_name="up_lidar",
        lidar_id=LidarID.LIDAR_TOP,
        lidar_to_imu_se3=_row_dict_to_pose_se3(
            calibration_df[calibration_df["sensor_name"] == "up_lidar"].iloc[0].to_dict()
        ),
    )
    metadata[LidarID.LIDAR_DOWN] = LidarMetadata(
        lidar_name="down_lidar",
        lidar_id=LidarID.LIDAR_DOWN,
        lidar_to_imu_se3=_row_dict_to_pose_se3(
            calibration_df[calibration_df["sensor_name"] == "down_lidar"].iloc[0].to_dict()
        ),
    )
    return metadata


def _extract_av2_sensor_box_detections(
    annotations_df: Optional[pd.DataFrame],
    lidar_timestamp_ns: int,
    ego_state_se3: EgoStateSE3,
) -> BoxDetectionsSE3:
    """Extract box detections from AV2 sensor dataset annotations."""
    if annotations_df is None:
        return BoxDetectionsSE3(box_detections=[], timestamp=Timestamp.from_ns(int(lidar_timestamp_ns)))

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
                metadata=BoxDetectionAttributes(
                    label=detections_labels[detection_idx],
                    track_token=detections_token[detection_idx],
                    num_lidar_points=detections_num_lidar_points[detection_idx],
                ),
                bounding_box_se3=BoundingBoxSE3.from_array(detections_state[detection_idx]),
                velocity_3d=Vector3D.from_array(detections_velocity[detection_idx]),
            )
        )

    return BoxDetectionsSE3(box_detections=box_detections, timestamp=Timestamp.from_ns(int(lidar_timestamp_ns)))  # type: ignore


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
) -> List[CameraData]:
    """Extract pinhole camera data from AV2 sensor dataset."""
    camera_data_list: List[CameraData] = []
    split = source_log_path.parent.name
    log_id = source_log_path.name

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

            nearest_pose = get_slice_with_timestamp_ns(city_se3_egovehicle_df, int(timestamp_ns_str)).iloc[0].to_dict()
            nearest_pose_se3 = _row_dict_to_pose_se3(nearest_pose)  # type: ignore
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


def _extract_av2_sensor_lidar(source_log_path: Path, lidar_timestamp_ns: int) -> Optional[LidarData]:
    """Extract Lidar data from AV2 sensor dataset. Returns None if lidars not included."""

    av2_sensor_data_root = source_log_path.parent.parent
    split_type = source_log_path.parent.name
    log_name = source_log_path.name

    relative_feather_path = f"{split_type}/{log_name}/sensors/lidar/{lidar_timestamp_ns}.feather"
    lidar_feather_path = av2_sensor_data_root / relative_feather_path
    assert lidar_feather_path.exists(), f"Lidar feather file not found: {lidar_feather_path}"

    return LidarData(
        lidar_name=LidarID.LIDAR_MERGED.serialize(),
        lidar_type=LidarID.LIDAR_MERGED,
        start_timestamp=Timestamp.from_ns(int(lidar_timestamp_ns)),
        end_timestamp=Timestamp.from_ns(int(lidar_timestamp_ns)),
        dataset_root=av2_sensor_data_root,
        relative_path=relative_feather_path,
    )


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
