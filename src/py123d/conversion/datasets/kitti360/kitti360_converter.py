import datetime
import logging
import pickle
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple, Union

import numpy as np
import yaml

from py123d.api.map.abstract_map_writer import AbstractMapWriter
from py123d.api.scene.abstract_log_writer import AbstractLogWriter, CameraData, LidarData
from py123d.conversion.abstract_dataset_converter import AbstractDatasetConverter
from py123d.conversion.dataset_converter_config import DatasetConverterConfig
from py123d.conversion.datasets.kitti360.kitti360_map_conversion import convert_kitti360_map_with_writer
from py123d.conversion.datasets.kitti360.utils.kitti360_helper import (
    KITTI3602NUPLAN_IMU_CALIBRATION,
    KITTI360Bbox3D,
    get_kitti360_lidar_extrinsic,
)
from py123d.conversion.datasets.kitti360.utils.kitti360_labels import (
    BBOX_LABLES_TO_DETECTION_NAME_DICT,
    KITTI360_DETECTION_NAME_DICT,
    kittiId2label,
)
from py123d.conversion.datasets.kitti360.utils.preprocess_detection import process_detection
from py123d.conversion.registry import KITTI360BoxDetectionLabel
from py123d.datatypes import (
    BoxDetectionMetadata,
    BoxDetectionSE3,
    BoxDetectionsSE3,
    DynamicStateSE3,
    EgoStateSE3,
    FisheyeMEICameraID,
    FisheyeMEICameraMetadata,
    FisheyeMEIDistortion,
    FisheyeMEIProjection,
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
from py123d.datatypes.vehicle_state.vehicle_parameters import get_kitti360_vw_passat_parameters
from py123d.geometry import BoundingBoxSE3, PoseSE3, Quaternion, Vector3D

KITTI360_DT: Final[float] = 0.1
KITTI360_LIDAR_NAME: Final[str] = "velodyne_points"
KITTI360_PINHOLE_CAMERA_IDS = {
    PinholeCameraID.PCAM_STEREO_L: "image_00",
    PinholeCameraID.PCAM_STEREO_R: "image_01",
}
KITTI360_FISHEYE_MEI_CAMERA_IDS = {
    FisheyeMEICameraID.FCAM_L: "image_02",
    FisheyeMEICameraID.FCAM_R: "image_03",
}

KITTI360_SPLITS: List[str] = ["kitti360_train", "kitti360_val", "kitti360_test"]
KITTI360_ALL_SEQUENCES: Final[List[str]] = [
    "2013_05_28_drive_0000_sync",
    "2013_05_28_drive_0002_sync",
    "2013_05_28_drive_0003_sync",
    "2013_05_28_drive_0004_sync",
    "2013_05_28_drive_0005_sync",
    "2013_05_28_drive_0006_sync",
    "2013_05_28_drive_0007_sync",
    "2013_05_28_drive_0008_sync",
    "2013_05_28_drive_0009_sync",
    "2013_05_28_drive_0010_sync",
    "2013_05_28_drive_0018_sync",
]

DIR_ROOT = "root"
DIR_2D_RAW = "data_2d_raw"
DIR_2D_SMT = "data_2d_semantics"
DIR_3D_RAW = "data_3d_raw"
DIR_3D_SMT = "data_3d_semantics"
DIR_3D_BBOX = "data_3d_bboxes"
DIR_POSES = "data_poses"
DIR_CALIB = "calibration"


def _get_kitti360_paths_from_root(kitti_data_root: Path) -> Dict[str, Path]:
    return {
        DIR_ROOT: kitti_data_root,
        DIR_2D_RAW: kitti_data_root / DIR_2D_RAW,
        DIR_2D_SMT: kitti_data_root / DIR_2D_SMT,
        DIR_3D_RAW: kitti_data_root / DIR_3D_RAW,
        DIR_3D_SMT: kitti_data_root / DIR_3D_SMT,
        DIR_3D_BBOX: kitti_data_root / DIR_3D_BBOX,
        DIR_POSES: kitti_data_root / DIR_POSES,
        DIR_CALIB: kitti_data_root / DIR_CALIB,
    }


def _get_kitti360_required_modality_roots(kitti360_folders: Dict[str, Path]) -> Dict[str, Path]:
    return {
        DIR_2D_RAW: kitti360_folders[DIR_2D_RAW],
        DIR_3D_RAW: kitti360_folders[DIR_3D_RAW],
        DIR_POSES: kitti360_folders[DIR_POSES],
        DIR_3D_BBOX: kitti360_folders[DIR_3D_BBOX] / "train",
    }


class Kitti360Converter(AbstractDatasetConverter):
    """Converter class for KITTI-360 dataset."""

    def __init__(
        self,
        splits: List[str],
        kitti360_data_root: Union[Path, str],
        detection_cache_root: Union[Path, str],
        detection_radius: float,
        dataset_converter_config: DatasetConverterConfig,
        train_sequences: List[str],
        val_sequences: List[str],
        test_sequences: List[str],
    ) -> None:
        """Initializes the Kitti360Converter.

        :param splits: List of splits to include in the conversion, e.g. `kitti360_train`, `kitti360_val`, `kitti360_test`
        :param kitti360_data_root: Path to the KITTI-360 dataset root directory
        :param detection_cache_root: Path to the detection cache directory
        :param detection_radius: Radius for the box detections to include.
        :param dataset_converter_config: Dataset converter configuration
        :param train_sequences: List of sequences to include in the training split
        :param val_sequences: List of sequences to include in the validation split
        :param test_sequences: List of sequences to include in the test split
        """

        assert kitti360_data_root is not None, "The variable `kitti360_data_root` must be provided."
        super().__init__(dataset_converter_config)
        for split in splits:
            assert split in KITTI360_SPLITS, f"Split {split} is not available. Available splits: {KITTI360_SPLITS}"

        self._splits: List[str] = splits
        self._kitti360_data_root: Path = Path(kitti360_data_root)
        self._kitti360_folders: Dict[str, Path] = _get_kitti360_paths_from_root(self._kitti360_data_root)

        # NOTE: We preprocess detections into cache directory to speed up repeated conversions
        # The bounding boxes are preprocessed into a per-frame format based on the ego distance and
        # visibility based on the lidar point cloud.
        self._detection_cache_root: Path = Path(detection_cache_root)
        self._detection_radius: float = detection_radius

        self._train_sequences: List[str] = train_sequences
        self._val_sequences: List[str] = val_sequences
        self._test_sequences: List[str] = test_sequences

        self._log_names_and_split: List[Tuple[str, str]] = self._collect_valid_logs()
        self._total_maps = len(self._log_names_and_split)  # Each log has its own map
        self._total_logs = len(self._log_names_and_split)

        # NOTE: camera calibration is shared across all sequences, so we can load it once here and reuse for all logs
        self._camera_calibration = _load_kitti_360_calibration(self._kitti360_data_root)

    def _collect_valid_logs(self) -> List[Tuple[str, str]]:
        """Helper function to collect valid KITTI sequences ("logs") from the dataset root

        :raises FileNotFoundError: If required modality roots are missing
        :return: A list of tuples containing the log name and split name
        """

        def _has_modality(seq_name: str, modality_name: str, root: Path) -> bool:
            if modality_name == DIR_3D_BBOX:
                # expected: data_3d_bboxes/train/<seq_name>.xml
                xml_path = root / f"{seq_name}.xml"
                return xml_path.exists()
            else:
                return (root / seq_name).exists()

        required_modality_roots = _get_kitti360_required_modality_roots(self._kitti360_folders)
        missing_roots = [str(p) for p in required_modality_roots.values() if not p.exists()]
        if missing_roots:
            raise FileNotFoundError(f"KITTI-360 required roots missing: {missing_roots}")

        # Find all sequences in the 2D raw data directory, and add to split
        split_sequence_candidates: Dict[str, List[str]] = defaultdict(list)
        for sequence_path in required_modality_roots[DIR_2D_RAW].iterdir():
            if sequence_path.is_dir() and sequence_path.name.endswith("_sync"):
                seq_name = sequence_path.name
                if seq_name in self._train_sequences:
                    split_sequence_candidates["kitti360_train"].append(seq_name)
                elif seq_name in self._val_sequences:
                    split_sequence_candidates["kitti360_val"].append(seq_name)
                elif seq_name in self._test_sequences:
                    split_sequence_candidates["kitti360_test"].append(seq_name)

        # Iterate all candidates, check that modalities available, and add to flat list
        log_paths_and_split: List[Tuple[Path, str]] = []
        for split, sequence_names in split_sequence_candidates.items():
            if split not in self._splits:
                continue
            for sequence_name in sequence_names:
                missing_modalities = [
                    modality_name
                    for modality_name, root in required_modality_roots.items()
                    if not _has_modality(sequence_name, modality_name, root)
                ]
                if len(missing_modalities) == 0:
                    log_paths_and_split.append((sequence_name, split))
                else:
                    logging.info(
                        f"Sequence '{sequence_name}' skipped: missing modalities {missing_modalities}. "
                        f"Root: {self._kitti360_data_root}"
                    )

        logging.info(f"Valid sequences found: {len(log_paths_and_split)}")
        return log_paths_and_split

    def get_number_of_maps(self) -> int:
        """Inherited, see superclass."""
        return self._total_maps

    def get_number_of_logs(self) -> int:
        """Inherited, see superclass."""
        return self._total_logs

    def convert_map(self, map_index: int, map_writer: AbstractMapWriter) -> None:
        """Inherited, see superclass."""
        log_name, split = self._log_names_and_split[map_index]
        map_metadata = _get_kitti360_map_metadata(split, log_name)
        map_needs_writing = map_writer.reset(self.dataset_converter_config, map_metadata)
        if map_needs_writing:
            convert_kitti360_map_with_writer(log_name, map_writer)
        map_writer.close()

    def convert_log(self, log_index: int, log_writer: AbstractLogWriter) -> None:
        """Inherited, see superclass."""
        log_name, split = self._log_names_and_split[log_index]

        # Create log metadata
        log_metadata = LogMetadata(
            dataset="kitti360",
            split=split,
            log_name=log_name,
            location=log_name,
            timestep_seconds=KITTI360_DT,
            box_detection_label_class=KITTI360BoxDetectionLabel,
            vehicle_parameters=get_kitti360_vw_passat_parameters(),
            pinhole_camera_metadata=_get_kitti360_pinhole_camera_metadata(
                self._kitti360_folders,
                self._camera_calibration,
                self.dataset_converter_config,
            ),
            fisheye_mei_camera_metadata=_get_kitti360_fisheye_mei_camera_metadata(
                self._kitti360_folders,
                self._camera_calibration,
                self.dataset_converter_config,
            ),
            lidar_metadata=_get_kitti360_lidar_metadata(
                self._kitti360_folders,
                self.dataset_converter_config,
            ),
            map_metadata=_get_kitti360_map_metadata(split, log_name),
        )

        log_needs_writing = log_writer.reset(self.dataset_converter_config, log_metadata)

        if log_needs_writing:
            timestamps_dict: Dict[str, List[Timestamp]] = _read_timestamps(log_name, self._kitti360_folders)

            # NOTE: We use the Lidar timestamps as reference timestamps for the log
            assert KITTI360_LIDAR_NAME in timestamps_dict, "Lidar timestamps must be available, as main reference."
            reference_timestamps = timestamps_dict[KITTI360_LIDAR_NAME]

            ego_state_all, valid_timestamp = _extract_ego_state_all(log_name, self._kitti360_folders)
            ego_states_xyz = np.array(
                [ego_state.center_se3.point_3d.array[:3] for ego_state in ego_state_all], dtype=np.float64
            )
            box_detection_wrapper_all = _extract_kitti360_box_detections_all(
                log_name,
                len(reference_timestamps),
                ego_states_xyz,
                valid_timestamp,
                self._kitti360_folders,
                self._detection_cache_root,
                self._detection_radius,
            )

            logging.info(f"Number of valid timestamps with ego states: {len(valid_timestamp)}")

            for idx in range(len(valid_timestamp)):
                valid_idx = valid_timestamp[idx]

                pinhole_cameras = _extract_kitti360_pinhole_cameras(
                    log_name,
                    valid_idx,
                    self._camera_calibration,
                    timestamps_dict,
                    self._kitti360_folders,
                    self.dataset_converter_config,
                )
                fisheye_cameras = _extract_kitti360_fisheye_mei_cameras(
                    log_name,
                    valid_idx,
                    self._camera_calibration,
                    timestamps_dict,
                    self._kitti360_folders,
                    self.dataset_converter_config,
                )
                lidar = _extract_kitti360_lidar(
                    log_name,
                    valid_idx,
                    self._kitti360_folders,
                    self.dataset_converter_config,
                )

                log_writer.write(
                    timestamp=reference_timestamps[valid_idx],
                    ego_state=ego_state_all[idx],
                    box_detections=box_detection_wrapper_all[valid_idx],
                    pinhole_cameras=pinhole_cameras,
                    fisheye_mei_cameras=fisheye_cameras,
                    lidar=lidar,
                )

        log_writer.close()


def _get_kitti360_pinhole_camera_metadata(
    kitti360_folders: Dict[str, Path],
    camera_calibration: Dict[str, PoseSE3],
    dataset_converter_config: DatasetConverterConfig,
) -> Dict[PinholeCameraID, PinholeCameraMetadata]:
    """Gets the KITTI-360 pinhole camera metadata from calibration files."""

    pinhole_cam_metadatas: Dict[PinholeCameraID, PinholeCameraMetadata] = {}
    if dataset_converter_config.include_pinhole_cameras:
        persp = kitti360_folders[DIR_CALIB] / "perspective.txt"
        assert persp.exists()
        persp_result = {"image_00": {}, "image_01": {}}

        with open(persp, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
            for ln in lines:
                key, value = ln.split(" ", 1)
                cam_id = key.split("_")[-1][:2]
                if key.startswith("P_rect_"):
                    persp_result[f"image_{cam_id}"]["intrinsic"] = _read_projection_matrix(ln)
                elif key.startswith("S_rect_"):
                    persp_result[f"image_{cam_id}"]["wh"] = [int(round(float(x))) for x in value.split()]
                elif key.startswith("D_"):
                    persp_result[f"image_{cam_id}"]["distortion"] = [float(x) for x in value.split()]

        for pcam_type, pcam_name in KITTI360_PINHOLE_CAMERA_IDS.items():
            assert pcam_name in camera_calibration.keys(), f"Camera calibration missing for {pcam_name}"
            pinhole_cam_metadatas[pcam_type] = PinholeCameraMetadata(
                camera_name=pcam_name,
                camera_id=pcam_type,
                width=persp_result[pcam_name]["wh"][0],
                height=persp_result[pcam_name]["wh"][1],
                intrinsics=PinholeIntrinsics.from_camera_matrix(np.array(persp_result[pcam_name]["intrinsic"])),
                distortion=PinholeDistortion.from_array(np.array(persp_result[pcam_name]["distortion"])),
                camera_to_imu_se3=camera_calibration[pcam_name],
                is_undistorted=True,
            )

    return pinhole_cam_metadatas


def _get_kitti360_fisheye_mei_camera_metadata(
    kitti360_folders: Dict[str, Path],
    camera_calibration: Dict[str, PoseSE3],
    dataset_converter_config: DatasetConverterConfig,
) -> Dict[FisheyeMEICameraID, FisheyeMEICameraMetadata]:
    """Gets the KITTI-360 fisheye MEI camera metadata from calibration files."""

    fisheye_cam_metadatas: Dict[FisheyeMEICameraID, FisheyeMEICameraMetadata] = {}
    if dataset_converter_config.include_fisheye_mei_cameras:
        fisheye_camera02_path = kitti360_folders[DIR_CALIB] / "image_02.yaml"
        fisheye_camera03_path = kitti360_folders[DIR_CALIB] / "image_03.yaml"

        fisheye02 = _readYAMLFile(fisheye_camera02_path)
        fisheye03 = _readYAMLFile(fisheye_camera03_path)
        fisheye_result = {"image_02": fisheye02, "image_03": fisheye03}

        for fcam_id, fcam_name in KITTI360_FISHEYE_MEI_CAMERA_IDS.items():
            assert fcam_name in camera_calibration.keys(), f"Camera calibration missing for {fcam_name}"

            distortion_params = fisheye_result[fcam_name]["distortion_parameters"]
            distortion = FisheyeMEIDistortion(
                k1=distortion_params["k1"],
                k2=distortion_params["k2"],
                p1=distortion_params["p1"],
                p2=distortion_params["p2"],
            )

            projection_params = fisheye_result[fcam_name]["projection_parameters"]
            projection = FisheyeMEIProjection(
                gamma1=projection_params["gamma1"],
                gamma2=projection_params["gamma2"],
                u0=projection_params["u0"],
                v0=projection_params["v0"],
            )

            fisheye_cam_metadatas[fcam_id] = FisheyeMEICameraMetadata(
                camera_name=fcam_name,
                camera_id=fcam_id,
                width=fisheye_result[fcam_name]["image_width"],
                height=fisheye_result[fcam_name]["image_height"],
                mirror_parameter=float(fisheye_result[fcam_name]["mirror_parameters"]["xi"]),
                distortion=distortion,
                projection=projection,
                camera_to_imu_se3=camera_calibration[fcam_name],
            )

    return fisheye_cam_metadatas


def _get_kitti360_map_metadata(split: str, log_name: str) -> MapMetadata:
    """Gets the KITTI-360 map metadata."""
    return MapMetadata(
        dataset="kitti360",
        split=split,
        log_name=log_name,
        location=log_name,
        map_has_z=True,
        map_is_local=True,
    )


def _read_projection_matrix(p_line: str) -> np.ndarray:
    """Helper function to read projection matrix from calibration file line."""
    parts = p_line.split(" ", 1)
    if len(parts) != 2:
        raise ValueError(f"Bad projection line: {p_line}")
    vals = [float(x) for x in parts[1].strip().split()]
    P = np.array(vals, dtype=np.float64).reshape(3, 4)
    K = P[:, :3]
    return K


def _readYAMLFile(fileName: Path) -> Dict[str, Any]:
    """Make OpenCV YAML file compatible with python"""
    ret = {}
    skip_lines = 1  # Skip the first line which says "%YAML:1.0". Or replace it with "%YAML 1.0"
    with open(fileName) as fin:
        for i in range(skip_lines):
            fin.readline()
        yamlFileOut = fin.read()
        myRe = re.compile(r":([^ ])")  # Add space after ":", if it doesn't exist. Python yaml requirement
        yamlFileOut = myRe.sub(r": \1", yamlFileOut)
        ret = yaml.safe_load(yamlFileOut)
    return ret


def _get_kitti360_lidar_metadata(
    kitti360_folders: Dict[str, Path], dataset_converter_config: DatasetConverterConfig
) -> Dict[LidarID, LidarMetadata]:
    """Gets the KITTI-360 Lidar metadata from calibration files."""
    metadata: Dict[LidarID, LidarMetadata] = {}
    if dataset_converter_config.include_lidars:
        extrinsic = get_kitti360_lidar_extrinsic(kitti360_folders[DIR_CALIB])
        extrinsic_pose_se3 = PoseSE3.from_transformation_matrix(extrinsic)
        # TODO: remove when stable.
        # extrinsic_pose_se3 = _extrinsic_from_imu_to_rear_axle(extrinsic_pose_se3)
        metadata[LidarID.LIDAR_TOP] = LidarMetadata(
            lidar_name=KITTI360_LIDAR_NAME,
            lidar_id=LidarID.LIDAR_TOP,
            lidar_to_imu_se3=extrinsic_pose_se3,
        )
    return metadata


def _read_timestamps(log_name: str, kitti360_folders: Dict[str, Path]) -> Dict[str, List[Timestamp]]:
    """Read KITTI-360 timestamps for the given sequence and return Unix epoch timestamps."""

    ts_files = {
        "oxts": kitti360_folders[DIR_POSES] / log_name / "oxts" / "timestamps.txt",
        "velodyne_points": kitti360_folders[DIR_3D_RAW] / log_name / "velodyne_points" / "timestamps.txt",
        "image_00": kitti360_folders[DIR_2D_RAW] / log_name / "image_00" / "timestamps.txt",
        "image_01": kitti360_folders[DIR_2D_RAW] / log_name / "image_01" / "timestamps.txt",
        "image_02": kitti360_folders[DIR_2D_RAW] / log_name / "image_02" / "timestamps.txt",
        "image_03": kitti360_folders[DIR_2D_RAW] / log_name / "image_03" / "timestamps.txt",
    }

    # if log_name == "2013_05_28_drive_0002_sync":
    #     ts_files = ts_files[1:]  # FIXME: Why is this needed?

    timestamps: Dict[str, List[Timestamp]] = {}

    for modality, ts_file in ts_files.items():
        if ts_file.exists():
            tps: List[Timestamp] = []
            with open(ts_file, "r") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    dt_str, ns_str = s.split(".")
                    dt_obj = datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                    dt_obj = dt_obj.replace(tzinfo=datetime.timezone.utc)
                    unix_epoch = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
                    total_seconds = (dt_obj - unix_epoch).total_seconds()
                    ns_value = int(ns_str)
                    us_from_ns = ns_value // 1000
                    total_us = int(total_seconds * 1_000_000) + us_from_ns
                    tps.append(Timestamp.from_us(total_us))
            # return tps
            timestamps[modality] = tps
    return timestamps


def _extract_ego_state_all(log_name: str, kitti360_folders: Dict[str, Path]) -> Tuple[List[EgoStateSE3], List[int]]:
    """Extracts all ego states for the given sequence."""

    ego_state_all: List[EgoStateSE3] = []
    pose_file = kitti360_folders[DIR_POSES] / log_name / "poses.txt"
    if not pose_file.exists():
        raise FileNotFoundError(f"Pose file not found: {pose_file}")
    poses = np.loadtxt(pose_file)
    poses_time = poses[:, 0].astype(np.int32)
    valid_timestamp: List[int] = list(poses_time)
    oxts_path = kitti360_folders[DIR_POSES] / log_name / "oxts" / "data"

    for idx in range(len(valid_timestamp)):
        vehicle_parameters = get_kitti360_vw_passat_parameters()

        pos = idx
        if log_name == "2013_05_28_drive_0004_sync" and pos == 0:
            pos = 1

        # NOTE you can use oxts_data[3:6] as roll, pitch, yaw for simplicity
        # roll, pitch, yaw = oxts_data[3:6]
        r00, r01, r02 = poses[pos, 1:4]
        r10, r11, r12 = poses[pos, 5:8]
        r20, r21, r22 = poses[pos, 9:12]
        R_mat = np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]], dtype=np.float64)
        R_mat_cali = R_mat @ KITTI3602NUPLAN_IMU_CALIBRATION[:3, :3]

        ego_quaternion = Quaternion.from_rotation_matrix(R_mat_cali)
        imu_pose_se3 = PoseSE3(
            x=poses[pos, 4],
            y=poses[pos, 8],
            z=poses[pos, 12],
            qw=ego_quaternion.qw,
            qx=ego_quaternion.qx,
            qy=ego_quaternion.qy,
            qz=ego_quaternion.qz,
        )

        oxts_path_file = oxts_path / f"{int(valid_timestamp[idx]):010d}.txt"
        if oxts_path_file.exists():
            # NOTE: "2013_05_28_drive_0009_sync" is missing oxts files
            oxts_data = np.loadtxt(oxts_path_file)
            dynamic_state_se3 = DynamicStateSE3(
                velocity=Vector3D(x=oxts_data[8], y=oxts_data[9], z=oxts_data[10]),
                acceleration=Vector3D(x=oxts_data[14], y=oxts_data[15], z=oxts_data[16]),
                angular_velocity=Vector3D(x=oxts_data[20], y=oxts_data[21], z=oxts_data[22]),
            )
        else:
            dynamic_state_se3 = None
        ego_state_all.append(
            EgoStateSE3.from_imu(
                imu_se3=imu_pose_se3,
                vehicle_parameters=vehicle_parameters,
                dynamic_state_se3=dynamic_state_se3,
            )
        )
    return ego_state_all, valid_timestamp


def _extract_kitti360_box_detections_all(
    log_name: str,
    ts_len: int,
    ego_states_xyz: np.ndarray,
    valid_timestamp: List[int],
    kitti360_folders: Dict[str, Path],
    detection_cache_root: Path,
    detection_radius: float,
) -> List[BoxDetectionsSE3]:
    """Extracts all KITTI-360 box detections for the given sequence."""

    detections_states: List[List[List[float]]] = [[] for _ in range(ts_len)]
    detections_velocity: List[List[List[float]]] = [[] for _ in range(ts_len)]
    detections_tokens: List[List[str]] = [[] for _ in range(ts_len)]
    detections_labels: List[List[int]] = [[] for _ in range(ts_len)]

    if log_name == "2013_05_28_drive_0004_sync":
        bbox_3d_path = kitti360_folders[DIR_3D_BBOX] / "train_full" / f"{log_name}.xml"
    else:
        bbox_3d_path = kitti360_folders[DIR_3D_BBOX] / "train" / f"{log_name}.xml"
    if not bbox_3d_path.exists():
        raise FileNotFoundError(f"BBox 3D file not found: {bbox_3d_path}")

    tree = ET.parse(bbox_3d_path)
    root = tree.getroot()

    detection_preprocess_path = detection_cache_root / f"{log_name}_detection_preprocessed.pkl"
    if not detection_preprocess_path.exists():
        process_detection(
            kitti360_data_root=kitti360_folders[DIR_ROOT],
            log_name=log_name,
            radius_m=detection_radius,
            output_dir=detection_cache_root,
        )
    with open(detection_preprocess_path, "rb") as f:
        detection_preprocess_result = pickle.load(f)
        static_records_dict = {
            record_item["global_id"]: record_item for record_item in detection_preprocess_result["static"]
        }
        logging.info(f"Loaded detection preprocess data from {detection_preprocess_path}")

    dynamic_objs: Dict[int, List[KITTI360Bbox3D]] = defaultdict(list)

    for child in root:
        if child.find("semanticId") is not None:
            semanticIdKITTI = int(child.find("semanticId").text)
            name = kittiId2label[semanticIdKITTI].name
        else:
            label = child.find("label").text
            name = BBOX_LABLES_TO_DETECTION_NAME_DICT.get(label, "unknown")
        if child.find("transform") is None or name not in KITTI360_DETECTION_NAME_DICT.keys():
            continue
        obj = KITTI360Bbox3D()
        obj.parseBbox(child)

        # static object
        if obj.timestamp == -1:
            if detection_preprocess_result is None:
                obj.filter_by_radius(ego_states_xyz, valid_timestamp, radius=50.0)
            else:
                obj.load_detection_preprocess(static_records_dict)
            for record in obj.valid_frames["records"]:
                frame = record["timestamp"]
                detections_states[frame].append(obj.get_state_array())
                detections_velocity[frame].append(np.array([0.0, 0.0, 0.0]))
                detections_tokens[frame].append(str(obj.globalID))
                detections_labels[frame].append(KITTI360_DETECTION_NAME_DICT[obj.name])
        else:
            global_ID = obj.globalID
            dynamic_objs[global_ID].append(obj)

    # dynamic object
    for global_id, obj_list in dynamic_objs.items():
        obj_list.sort(key=lambda obj: obj.timestamp)
        num_frames = len(obj_list)

        positions = [obj.get_state_array()[:3] for obj in obj_list]
        timestamps = [int(obj.timestamp) for obj in obj_list]

        velocities = []

        for i in range(1, num_frames - 1):
            dt_frames = timestamps[i + 1] - timestamps[i - 1]
            if dt_frames > 0:
                dt = dt_frames * KITTI360_DT
                vel = (positions[i + 1] - positions[i - 1]) / dt
                vel = KITTI3602NUPLAN_IMU_CALIBRATION[:3, :3] @ obj_list[i].Rm.T @ vel
            else:
                vel = np.zeros(3)
            velocities.append(vel)

        if num_frames > 1:
            # first and last frame
            velocities.insert(0, velocities[0])
            velocities.append(velocities[-1])
        elif num_frames == 1:
            velocities.append(np.zeros(3))

        for obj, vel in zip(obj_list, velocities):
            frame = obj.timestamp
            detections_states[frame].append(obj.get_state_array())
            detections_velocity[frame].append(vel)
            detections_tokens[frame].append(str(obj.globalID))
            detections_labels[frame].append(KITTI360_DETECTION_NAME_DICT[obj.name])

    box_detection_wrapper_all: List[BoxDetectionsSE3] = []
    for frame in range(ts_len):
        box_detections: List[BoxDetectionSE3] = []
        for state, velocity, token, detection_label in zip(
            detections_states[frame],
            detections_velocity[frame],
            detections_tokens[frame],
            detections_labels[frame],
        ):
            if state is None:
                break
            detection_metadata = BoxDetectionMetadata(
                label=detection_label,
                timestamp=None,
                track_token=token,
            )
            bounding_box_se3 = BoundingBoxSE3.from_array(state)
            velocity_vector = Vector3D.from_array(velocity)
            box_detection = BoxDetectionSE3(
                metadata=detection_metadata,
                bounding_box_se3=bounding_box_se3,
                velocity_3d=velocity_vector,
            )
            box_detections.append(box_detection)
        box_detection_wrapper_all.append(BoxDetectionsSE3(box_detections=box_detections))
    return box_detection_wrapper_all


def _extract_kitti360_lidar(
    log_name: str,
    idx: int,
    kitti360_folders: Dict[str, Path],
    data_converter_config: DatasetConverterConfig,
) -> Optional[LidarData]:
    """Extracts KITTI-360 Lidar data for the given sequence and index."""

    lidar: Optional[LidarData] = None
    if data_converter_config.include_lidars:
        # NOTE special case for sequence 2013_05_28_drive_0002_sync which has no lidar data before frame 4391
        if not (log_name == "2013_05_28_drive_0002_sync" and idx <= 4390):
            lidar_full_path = kitti360_folders[DIR_3D_RAW] / log_name / "velodyne_points" / "data" / f"{idx:010d}.bin"
            if lidar_full_path.exists():
                lidar = LidarData(
                    lidar_name=KITTI360_LIDAR_NAME,
                    lidar_type=LidarID.LIDAR_TOP,
                    timestamp=None,
                    iteration=idx,
                    dataset_root=kitti360_folders[DIR_ROOT],
                    relative_path=lidar_full_path.relative_to(kitti360_folders[DIR_ROOT]),
                )
            else:
                raise FileNotFoundError(f"Lidar file not found: {lidar_full_path}")

    return lidar


def _extract_kitti360_pinhole_cameras(
    log_name: str,
    idx: int,
    camera_calibration: Dict[str, PoseSE3],
    timestamps_dict: Dict[str, List[Timestamp]],
    kitti360_folders: Dict[str, Path],
    data_converter_config: DatasetConverterConfig,
) -> List[CameraData]:
    """Extracts KITTI-360 pinhole camera data for the given sequence and index."""

    pinhole_camera_data_list: List[CameraData] = []
    if data_converter_config.include_pinhole_cameras:
        for camera_type, camera_name in KITTI360_PINHOLE_CAMERA_IDS.items():
            img_path_png = kitti360_folders[DIR_2D_RAW] / log_name / camera_name / "data_rect" / f"{idx:010d}.png"
            camera_extrinsic = camera_calibration[camera_name]
            camera_timestamp = timestamps_dict[camera_name][idx]
            if img_path_png.exists():
                pinhole_camera_data_list.append(
                    CameraData(
                        camera_name=camera_name,
                        camera_id=camera_type,
                        extrinsic=camera_extrinsic,
                        timestamp=camera_timestamp,
                        dataset_root=kitti360_folders[DIR_ROOT],
                        relative_path=img_path_png.relative_to(kitti360_folders[DIR_ROOT]),
                    )
                )

    return pinhole_camera_data_list


def _extract_kitti360_fisheye_mei_cameras(
    log_name: str,
    idx: int,
    camera_calibration: Dict[str, PoseSE3],
    timestamps_dict: Dict[str, List[Timestamp]],
    kitti360_folders: Dict[str, Path],
    data_converter_config: DatasetConverterConfig,
) -> List[CameraData]:
    """Extracts KITTI-360 fisheye MEI camera data for the given sequence and index."""
    fisheye_camera_data_list: List[CameraData] = []
    if data_converter_config.include_fisheye_mei_cameras:
        for camera_type, cam_dir_name in KITTI360_FISHEYE_MEI_CAMERA_IDS.items():
            img_path_png = kitti360_folders[DIR_2D_RAW] / log_name / cam_dir_name / "data_rgb" / f"{idx:010d}.png"
            camera_extrinsic = camera_calibration[cam_dir_name]
            camera_timestamp = timestamps_dict[cam_dir_name][idx]
            if img_path_png.exists():
                fisheye_camera_data_list.append(
                    CameraData(
                        camera_name=cam_dir_name,
                        camera_id=camera_type,
                        extrinsic=camera_extrinsic,
                        timestamp=camera_timestamp,
                        dataset_root=kitti360_folders[DIR_ROOT],
                        relative_path=img_path_png.relative_to(kitti360_folders[DIR_ROOT]),
                    )
                )
    return fisheye_camera_data_list


def _load_kitti_360_calibration(kitti_360_data_root: Path) -> Dict[str, PoseSE3]:
    """Helper function to load KITTI-360 camera to IMU calibration."""
    calib_file = kitti_360_data_root / DIR_CALIB / "calib_cam_to_pose.txt"
    if not calib_file.exists():
        raise FileNotFoundError(f"Calibration file not found: {calib_file}")

    lastrow = np.array([0, 0, 0, 1]).reshape(1, 4)
    calib_dict: Dict[str, PoseSE3] = {}
    with open(calib_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            key = parts[0][:-1]
            values = list(map(float, parts[1:]))
            matrix = np.array(values).reshape(3, 4)
            cam2pose = np.concatenate((matrix, lastrow))
            cam2pose = KITTI3602NUPLAN_IMU_CALIBRATION @ cam2pose
            camera_extrinsic = PoseSE3.from_transformation_matrix(cam2pose)
            calib_dict[key] = camera_extrinsic
    return calib_dict
