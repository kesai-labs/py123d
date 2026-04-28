from __future__ import annotations

import contextlib
import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from py123d.parser.pandaset.pandaset_download import PandasetDownloader

logger = logging.getLogger(__name__)

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
    compute_global_main_lidar_from_camera,
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
        pandaset_data_root: Optional[Union[Path, str]] = None,
        train_log_names: Optional[List[str]] = None,
        val_log_names: Optional[List[str]] = None,
        test_log_names: Optional[List[str]] = None,
        downloader: Optional["PandasetDownloader"] = None,
    ) -> None:
        """Initializes the :class:`PandasetParser`.

        :param splits: List of splits to include in the conversion. \
            Available splits: 'pandaset_train', 'pandaset_val', 'pandaset_test'.
        :param pandaset_data_root: Path to the root directory of the Pandaset dataset.
            Required when ``downloader`` is ``None``; ignored otherwise.
        :param train_log_names: List of log names to include in the training split.
        :param val_log_names: List of log names to include in the validation split.
        :param test_log_names: List of log names to include in the test split.
        :param downloader: Optional :class:`~py123d.parser.pandaset.pandaset_download.PandasetDownloader`
            used for streaming mode. When provided, each log parser extracts its assigned
            log from the cached ``pandaset.zip`` into a per-log
            :class:`tempfile.TemporaryDirectory`, converts it, and deletes the temp dir
            before moving on. Log selection comes from
            :meth:`PandasetDownloader.resolve_log_names`, intersected with the
            per-split lists so each log is routed to the right split. No local
            ``pandaset_data_root`` is required in this mode.
        """
        for split in splits:
            assert split in PANDASET_SPLITS, f"Split {split} is not available. Available splits: {PANDASET_SPLITS}"

        self._splits: List[str] = splits
        self._train_log_names: List[str] = list(train_log_names) if train_log_names else []
        self._val_log_names: List[str] = list(val_log_names) if val_log_names else []
        self._test_log_names: List[str] = list(test_log_names) if test_log_names else []
        self._downloader: Optional["PandasetDownloader"] = downloader

        if downloader is not None:
            self._pandaset_data_root: Optional[Path] = None
            self._log_entries: List[Tuple[Optional[Path], str, str]] = self._collect_log_entries_streaming()
        else:
            assert pandaset_data_root is not None, "`pandaset_data_root` must be provided when `downloader` is None."
            self._pandaset_data_root = Path(pandaset_data_root)
            self._log_entries = self._collect_log_entries_local()

    def _split_for_log(self, log_name: str) -> Optional[str]:
        """Return the active split a ``log_name`` belongs to, or ``None`` if none matches.

        Streaming-mode convenience: when a downloader is configured AND none of the
        per-split log-name lists are populated, every valid log is routed to
        ``splits[0]``. This lets ``dataset=pandaset-stream`` keep its config short —
        the caller only specifies which logs to stream; split routing defaults to the
        first active split. Local mode requires explicit per-split lists as before.
        """
        streaming_fallback = (
            self._downloader is not None
            and not (self._train_log_names or self._val_log_names or self._test_log_names)
            and log_name in PANDASET_LOG_NAMES
        )
        if streaming_fallback:
            return self._splits[0] if self._splits else None

        result: Optional[str] = None
        if (log_name in self._train_log_names) and ("pandaset_train" in self._splits):
            result = "pandaset_train"
        elif (log_name in self._val_log_names) and ("pandaset_val" in self._splits):
            result = "pandaset_val"
        elif (log_name in self._test_log_names) and ("pandaset_test" in self._splits):
            result = "pandaset_test"
        return result

    def _collect_log_entries_local(self) -> List[Tuple[Optional[Path], str, str]]:
        """Discover logs under ``{pandaset_data_root}/{log_name}/`` and route them to splits."""
        assert self._pandaset_data_root is not None
        entries: List[Tuple[Optional[Path], str, str]] = []
        for log_folder in self._pandaset_data_root.iterdir():
            if not log_folder.is_dir():
                continue
            log_name = log_folder.name
            assert log_name in PANDASET_LOG_NAMES, f"Log name {log_name} is not recognized."
            split = self._split_for_log(log_name)
            if split is not None:
                entries.append((log_folder, log_name, split))
        return entries

    def _collect_log_entries_streaming(self) -> List[Tuple[Optional[Path], str, str]]:
        """Enumerate log names from the downloader and route them to splits.

        Logs the downloader selects that are not present in any active split's log-name
        list are skipped with a warning — the per-split lists remain authoritative for
        split routing even in streaming mode.
        """
        assert self._downloader is not None
        selected = self._downloader.resolve_log_names()
        logger.info("PandaSet streaming: %d logs selected by downloader", len(selected))

        entries: List[Tuple[Optional[Path], str, str]] = []
        for log_name in selected:
            split = self._split_for_log(log_name)
            if split is None:
                logger.warning(
                    "PandaSet streaming: log %s is not assigned to any active split "
                    "(train/val/test name lists); skipping.",
                    log_name,
                )
                continue
            entries.append((None, log_name, split))
        return entries

    def get_log_parsers(self) -> List[PandasetLogParser]:  # type: ignore
        """Inherited, see superclass."""
        return [
            PandasetLogParser(
                source_log_path=source_log_path,
                log_name=log_name,
                split=split,
                downloader=self._downloader,
            )
            for source_log_path, log_name, split in self._log_entries
        ]

    def get_map_parsers(self) -> List[BaseMapParser]:
        """Inherited, see superclass."""
        return []  # NOTE @DanielDauner: Pandaset does not have maps.


class PandasetLogParser(BaseLogParser):
    """Lightweight, picklable handle to one Pandaset log."""

    def __init__(
        self,
        source_log_path: Optional[Path],
        log_name: str,
        split: str,
        downloader: Optional["PandasetDownloader"] = None,
    ) -> None:
        self._source_log_path = source_log_path
        self._log_name = log_name
        self._split = split
        self._downloader = downloader

    def get_log_metadata(self) -> LogMetadata:
        """Inherited, see superclass."""
        return LogMetadata(
            dataset="pandaset",
            split=self._split,
            log_name=self._log_name,
            location=None,  # TODO: Add location information.
        )

    @contextlib.contextmanager
    def _resolved_log(self) -> Iterator[Path]:
        """Yield the on-disk log path for the duration of one iterator pass.

        In local mode this is the pre-set ``source_log_path``. In streaming mode
        the log is extracted from the cached ``pandaset.zip`` into a fresh
        :class:`tempfile.TemporaryDirectory` which is deleted when the context
        manager exits (i.e. after ``iter_modalities_sync`` is exhausted).
        """
        if self._downloader is None:
            assert self._source_log_path is not None
            yield self._source_log_path
            return

        with tempfile.TemporaryDirectory(prefix=f"py123d-pandaset-{self._log_name}-") as tmp:
            tmp_root = Path(tmp)
            logger.info("Streaming PandaSet log %s to %s", self._log_name, tmp_root)
            log_path = self._downloader.download_single_log(
                log_name=self._log_name,
                output_dir=tmp_root,
            )
            yield log_path

    def iter_modalities_sync(self) -> Iterator[ModalitiesSync]:
        """Inherited, see superclass."""
        with self._resolved_log() as source_log_path:
            yield from self._iter_modalities_sync_from_path(source_log_path)

    def _iter_modalities_sync_from_path(self, source_log_path: Path) -> Iterator[ModalitiesSync]:
        """Emit synchronized modalities from an on-disk log directory."""
        ego_state_se3_metadata = PANDASET_EGO_STATE_SE3_METADATA
        box_detections_se3_metadata = PANDASET_BOX_DETECTIONS_SE3_METADATA
        pinhole_cameras_metadata = _get_pandaset_camera_metadata(source_log_path)
        lidar_merged_metadata = PANDASET_LIDAR_MERGED_METADATA

        # Read files from pandaset
        lidar_timestamps_s = read_json(source_log_path / "meta" / "timestamps.json")

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
                front_camera_pose=camera_poses["front_camera"][iteration],
                ego_metadata=ego_state_se3_metadata,
                timestamp=Timestamp.from_s(camera_timestamps_s["front_camera"][iteration]),
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
            is_undistorted=True,
        )

    return camera_metadata if camera_metadata else None


def _extract_pandaset_sensor_ego_state(
    front_camera_pose: Dict[str, Dict[str, float]],
    ego_metadata: EgoStateSE3Metadata,
    timestamp: Timestamp,
) -> EgoStateSE3:
    """Extracts the ego state from PandaSet front camera pose data.

    NOTE @DanielDauner: The lidar poses were not reliable in general and are inconsistant across logs.
    We use the same strategy as Neurad studio and use the front camera as reference ego pose.
    https://github.com/georghess/neurad-studio/blob/main/nerfstudio/data/dataparsers/pandaset_dataparser.py#L217-L218

    PandaSet lidar poses are unreliable, so the lidar-to-world transform is derived
    from the front camera pose and its static extrinsic calibration.
    """
    global_lidar = compute_global_main_lidar_from_camera(
        camera_pose=pandaset_pose_dict_to_pose_se3(front_camera_pose),
        camera_extrinsic=PANDASET_CAMERA_EXTRINSICS["front_camera"],
    )
    imu_se3 = global_main_lidar_to_global_imu(global_lidar)

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
