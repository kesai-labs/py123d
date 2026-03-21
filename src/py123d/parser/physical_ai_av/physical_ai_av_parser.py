from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd

from py123d.datatypes import (
    BoxDetectionAttributes,
    BoxDetectionSE3,
    BoxDetectionsSE3,
    CameraID,
    EgoStateSE3,
    LidarID,
    LidarMetadata,
    LogMetadata,
    Timestamp,
)
from py123d.datatypes.detections.box_detections_metadata import BoxDetectionsSE3Metadata
from py123d.datatypes.modalities.base_modality import BaseModality
from py123d.datatypes.sensors.ftheta_camera import FThetaCameraMetadata, FThetaIntrinsics
from py123d.datatypes.sensors.lidar import LidarMergedMetadata
from py123d.datatypes.vehicle_state.dynamic_state import DynamicStateSE3
from py123d.datatypes.vehicle_state.ego_state_metadata import EgoStateSE3Metadata
from py123d.geometry import BoundingBoxSE3, BoundingBoxSE3Index, Vector3D, Vector3DIndex
from py123d.geometry.transform import rel_to_abs_se3_array
from py123d.geometry.transform.transform_se3 import rel_to_abs_se3
from py123d.parser.base_dataset_parser import (
    BaseDatasetParser,
    BaseLogParser,
    BaseMapParser,
    ModalitiesSync,
    ParsedCamera,
    ParsedLidar,
)
from py123d.parser.physical_ai_av.utils.physical_ai_av_constants import (
    PHYSICAL_AI_AV_BOX_DETECTIONS_SE3_METADATA,
    PHYSICAL_AI_AV_CAMERA_ID_MAPPING,
    PHYSICAL_AI_AV_EGO_STATE_SE3_METADATA,
    PHYSICAL_AI_AV_LABEL_CLASS_MAPPING,
    PHYSICAL_AI_AV_SPLITS,
)
from py123d.parser.physical_ai_av.utils.physical_ai_av_helper import (
    find_closest_index,
    quat_scalar_last_to_pose_se3,
)
from py123d.parser.registry import PhysicalAIAVBoxDetectionLabel


class PhysicalAIAVParser(BaseDatasetParser):
    """Dataset parser for the NVIDIA Physical AI Autonomous Vehicles dataset."""

    def __init__(
        self,
        splits: List[str],
        physical_ai_av_data_root: Union[Path, str],
        max_clips: Optional[int] = None,
    ) -> None:
        """Initializes the PhysicalAIAVParser.

        :param splits: List of dataset splits, e.g. ["physical-ai-av_train"].
        :param physical_ai_av_data_root: Root directory of the Physical AI AV dataset.
        :param max_clips: Maximum number of clips to convert, or None for all available.
        """
        assert physical_ai_av_data_root is not None, "The variable `physical_ai_av_data_root` must be provided."
        assert Path(physical_ai_av_data_root).exists(), (
            f"The provided `physical_ai_av_data_root` path {physical_ai_av_data_root} does not exist."
        )
        for split in splits:
            assert split in PHYSICAL_AI_AV_SPLITS, (
                f"Split {split} is not available. Available splits: {PHYSICAL_AI_AV_SPLITS}"
            )

        self._splits = splits
        self._data_root = Path(physical_ai_av_data_root)
        self._max_clips = max_clips
        self._clip_entries: List[Tuple[str, int, str]] = self._collect_clips()

    def _collect_clips(self) -> List[Tuple[str, int, str]]:
        """Collects (clip_id, chunk, split_name) tuples for the requested splits.

        Only includes clips whose chunk data is actually present on disk
        (i.e. the calibration parquet for that chunk exists).
        """
        clip_index = pd.read_parquet(self._data_root / "clip_index.parquet")
        available_chunks = self._discover_available_chunks()
        entries: List[Tuple[str, int, str]] = []
        for split in self._splits:
            split_type = split.split("_")[-1]  # "train", "val", or "test"
            split_clips = clip_index[clip_index["split"] == split_type]
            for clip_id, row in split_clips.iterrows():
                chunk = int(row["chunk"])
                if chunk in available_chunks:
                    entries.append((str(clip_id), chunk, split))
                    if self._max_clips is not None and len(entries) >= self._max_clips:
                        return entries
        return entries

    def _discover_available_chunks(self) -> set:
        """Discovers which chunks have calibration data on disk."""
        cal_dir = self._data_root / "calibration" / "sensor_extrinsics"
        available = set()
        if cal_dir.exists():
            for f in cal_dir.iterdir():
                if f.name.startswith("sensor_extrinsics.chunk_") and f.suffix == ".parquet":
                    chunk_str = f.stem.split("chunk_")[-1]
                    available.add(int(chunk_str))
        return available

    def get_log_parsers(self) -> List[PhysicalAIAVLogParser]:  # type: ignore[override]
        """Inherited, see superclass."""
        return [
            PhysicalAIAVLogParser(
                data_root=self._data_root,
                clip_id=clip_id,
                chunk=chunk,
                split=split,
            )
            for clip_id, chunk, split in self._clip_entries
        ]

    def get_map_parsers(self) -> List[BaseMapParser]:  # type: ignore[override]
        """Inherited, see superclass. No map data available for this dataset."""
        return []


class PhysicalAIAVLogParser(BaseLogParser):
    """Lightweight, picklable handle to one Physical AI AV clip."""

    def __init__(self, data_root: Path, clip_id: str, chunk: int, split: str) -> None:
        self._data_root = data_root
        self._clip_id = clip_id
        self._chunk = chunk
        self._split = split

    def get_log_metadata(self) -> LogMetadata:
        """Inherited, see superclass."""
        return LogMetadata(
            dataset="physical-ai-av",
            split=self._split,
            log_name=self._clip_id,
            location=None,
            timestep_seconds=0.1,
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Synchronized iteration (lidar-rate)
    # ------------------------------------------------------------------------------------------------------------------

    def _ego_offline_path(self) -> Path:
        """Path to the offline (smoothed) egomotion file — matches obstacle.offline reference frame."""
        return self._data_root / "labels" / "egomotion.offline" / f"{self._clip_id}.egomotion.offline.parquet"

    def _ego_regular_path(self) -> Path:
        """Path to the regular (raw) egomotion file — higher rate, includes velocity/acceleration."""
        return self._data_root / "labels" / "egomotion" / f"{self._clip_id}.egomotion.parquet"

    def iter_modalities_sync(self) -> Iterator[ModalitiesSync]:
        """Inherited, see superclass."""
        ego_metadata = PHYSICAL_AI_AV_EGO_STATE_SE3_METADATA
        det_metadata = PHYSICAL_AI_AV_BOX_DETECTIONS_SE3_METADATA

        # 1. Load egomotion
        # Regular ego for ego states and camera poses (higher rate, has velocity/acceleration).
        # Offline ego only for box detection transforms (its anchor frame matches obstacle.offline labels).
        ego_df = pd.read_parquet(self._ego_regular_path())
        ego_timestamps = ego_df["timestamp"].to_numpy(dtype=np.int64)
        if self._ego_offline_path().exists():
            ego_offline_df = pd.read_parquet(self._ego_offline_path())
            ego_offline_ts = ego_offline_df["timestamp"].to_numpy(dtype=np.int64)
        else:
            ego_offline_df = ego_df
            ego_offline_ts = ego_timestamps

        # 2. Load LiDAR timestamps
        lidar_path = self._data_root / "lidar" / "lidar_top_360fov" / f"{self._clip_id}.lidar_top_360fov.parquet"
        lidar_df = pd.read_parquet(lidar_path, columns=["reference_timestamp"])
        lidar_timestamps = lidar_df["reference_timestamp"].to_numpy(dtype=np.int64)

        # 3. Load camera timestamps
        cam_timestamps: Dict[str, np.ndarray] = {}
        for cam_name in PHYSICAL_AI_AV_CAMERA_ID_MAPPING.keys():
            ts_path = self._data_root / "camera" / cam_name / f"{self._clip_id}.{cam_name}.timestamps.parquet"
            if ts_path.exists():
                ts_df = pd.read_parquet(ts_path)
                cam_timestamps[cam_name] = ts_df["timestamp"].to_numpy(dtype=np.int64)

        # 4. Load calibration
        ftheta_metadatas = _get_ftheta_camera_metadatas(self._data_root, self._clip_id, self._chunk)
        lidar_metadata = _get_lidar_metadata(self._data_root, self._clip_id, self._chunk)

        # 5. Load obstacle labels (if available)
        obstacle_path = self._data_root / "labels" / "obstacle.offline" / f"{self._clip_id}.obstacle.offline.parquet"
        obstacle_df = pd.read_parquet(obstacle_path) if obstacle_path.exists() else None
        obs_timestamps = obstacle_df["timestamp_us"].to_numpy(dtype=np.int64) if obstacle_df is not None else None

        # 6. Open camera video captures
        captures: Dict[str, cv2.VideoCapture] = {}
        for cam_name in PHYSICAL_AI_AV_CAMERA_ID_MAPPING.keys():
            video_path = self._data_root / "camera" / cam_name / f"{self._clip_id}.{cam_name}.mp4"
            if video_path.exists():
                captures[cam_name] = cv2.VideoCapture(str(video_path))

        try:
            for spin_idx, lidar_ts in enumerate(lidar_timestamps):
                timestamp = Timestamp.from_us(int(lidar_ts))

                # Ego state (from regular ego — higher rate, has velocity/acceleration)
                ego_state = _extract_ego_state(ego_df, ego_timestamps, lidar_ts, ego_metadata)

                # Box detections (transformed using offline ego to match obstacle.offline frame)
                box_detections = _extract_box_detections(
                    obstacle_df, obs_timestamps, lidar_ts, ego_offline_df, ego_offline_ts, ego_metadata, det_metadata
                )

                # LiDAR
                parsed_lidar = ParsedLidar(
                    metadata=lidar_metadata,
                    start_timestamp=timestamp,
                    end_timestamp=Timestamp.from_us(int(lidar_ts) + 100_000),  # ~100ms sweep
                    dataset_root=self._data_root,
                    relative_path=f"lidar/lidar_top_360fov/{self._clip_id}.lidar_top_360fov.parquet",
                    iteration=spin_idx,
                )

                # Cameras
                parsed_cameras = _extract_cameras(
                    lidar_ts, ego_df, ego_timestamps, captures, cam_timestamps, ftheta_metadatas
                )

                yield ModalitiesSync(
                    timestamp=timestamp,
                    modalities=[ego_state, box_detections, parsed_lidar, *parsed_cameras],
                )
        finally:
            for cap in captures.values():
                cap.release()

    # ------------------------------------------------------------------------------------------------------------------
    # Asynchronous iteration (native-rate)
    # ------------------------------------------------------------------------------------------------------------------

    def iter_modalities_async(self) -> Iterator[BaseModality]:
        """Inherited, see superclass."""
        ego_metadata = PHYSICAL_AI_AV_EGO_STATE_SE3_METADATA
        det_metadata = PHYSICAL_AI_AV_BOX_DETECTIONS_SE3_METADATA
        ftheta_metadatas = _get_ftheta_camera_metadatas(self._data_root, self._clip_id, self._chunk)
        lidar_metadata = _get_lidar_metadata(self._data_root, self._clip_id, self._chunk)

        yield from self._iter_ego_states_se3(ego_metadata)
        yield from self._iter_box_detections_se3(det_metadata)
        yield from self._iter_lidar(lidar_metadata)
        for cam_name, cam_id in PHYSICAL_AI_AV_CAMERA_ID_MAPPING.items():
            if cam_id in ftheta_metadatas:
                yield from self._iter_camera(cam_name, ftheta_metadatas[cam_id], ego_metadata)

    def _iter_ego_states_se3(self, metadata: EgoStateSE3Metadata) -> Iterator[EgoStateSE3]:
        """Yields all ego state observations at native rate (~67-100Hz)."""
        ego_df = pd.read_parquet(self._ego_regular_path())

        for _, row in ego_df.iterrows():
            ego_pose = quat_scalar_last_to_pose_se3(
                qx=row["qx"], qy=row["qy"], qz=row["qz"], qw=row["qw"], x=row["x"], y=row["y"], z=row["z"]
            )
            dynamic_state = DynamicStateSE3(
                velocity=Vector3D(x=row["vx"], y=row["vy"], z=row["vz"]),
                acceleration=Vector3D(x=row["ax"], y=row["ay"], z=row["az"]),
                angular_velocity=Vector3D(x=0.0, y=0.0, z=0.0),
            )
            yield EgoStateSE3.from_imu(
                imu_se3=ego_pose,
                metadata=metadata,
                dynamic_state_se3=dynamic_state,
                timestamp=Timestamp.from_us(int(row["timestamp"])),
            )

    def _iter_box_detections_se3(self, metadata: BoxDetectionsSE3Metadata) -> Iterator[BoxDetectionsSE3]:
        """Yields obstacle detections grouped into lidar-rate windows.

        Each detection in Physical AI AV has its own unique timestamp, so we accumulate
        all detections within each lidar sweep interval (±50ms) and yield one
        BoxDetectionsSE3 per lidar timestamp.
        """
        obstacle_path = self._data_root / "labels" / "obstacle.offline" / f"{self._clip_id}.obstacle.offline.parquet"
        if not obstacle_path.exists():
            return

        ego_metadata = PHYSICAL_AI_AV_EGO_STATE_SE3_METADATA
        obstacle_df = pd.read_parquet(obstacle_path)
        ego_df = (
            pd.read_parquet(self._ego_offline_path())
            if self._ego_offline_path().exists()
            else pd.read_parquet(self._ego_regular_path())
        )
        ego_timestamps = ego_df["timestamp"].to_numpy(dtype=np.int64)

        lidar_path = self._data_root / "lidar" / "lidar_top_360fov" / f"{self._clip_id}.lidar_top_360fov.parquet"
        lidar_df = pd.read_parquet(lidar_path, columns=["reference_timestamp"])
        lidar_timestamps = lidar_df["reference_timestamp"].to_numpy(dtype=np.int64)
        obs_timestamps = obstacle_df["timestamp_us"].to_numpy()

        for lidar_ts in lidar_timestamps:
            mask = (obs_timestamps >= lidar_ts - 50_000) & (obs_timestamps < lidar_ts + 50_000)
            group_df = obstacle_df[mask]
            yield _build_box_detections(group_df, int(lidar_ts), ego_df, ego_timestamps, ego_metadata, metadata)

    def _iter_lidar(self, metadata: LidarMergedMetadata) -> Iterator[ParsedLidar]:
        """Yields all LiDAR spins at native rate (~10Hz)."""
        lidar_path = self._data_root / "lidar" / "lidar_top_360fov" / f"{self._clip_id}.lidar_top_360fov.parquet"
        lidar_df = pd.read_parquet(lidar_path, columns=["reference_timestamp"])

        for spin_idx, row in lidar_df.iterrows():
            ts = int(row["reference_timestamp"])
            yield ParsedLidar(
                metadata=metadata,
                start_timestamp=Timestamp.from_us(ts),
                end_timestamp=Timestamp.from_us(ts + 100_000),
                dataset_root=self._data_root,
                relative_path=f"lidar/lidar_top_360fov/{self._clip_id}.lidar_top_360fov.parquet",
                iteration=int(spin_idx),  # type: ignore[arg-type]
            )

    def _iter_camera(
        self, cam_name: str, cam_metadata: FThetaCameraMetadata, ego_metadata: EgoStateSE3Metadata
    ) -> Iterator[ParsedCamera]:
        """Yields camera frames at native rate (~30fps)."""
        ts_path = self._data_root / "camera" / cam_name / f"{self._clip_id}.{cam_name}.timestamps.parquet"
        video_path = self._data_root / "camera" / cam_name / f"{self._clip_id}.{cam_name}.mp4"
        if not ts_path.exists() or not video_path.exists():
            return

        ts_df = pd.read_parquet(ts_path)
        ego_df = pd.read_parquet(self._ego_regular_path())
        ego_timestamps = ego_df["timestamp"].to_numpy(dtype=np.int64)

        cap = cv2.VideoCapture(str(video_path))
        try:
            for frame_idx, row in ts_df.iterrows():
                cam_ts = int(row["timestamp"])

                # Find closest ego pose
                ego_idx = find_closest_index(ego_timestamps, cam_ts)
                ego_row = ego_df.iloc[ego_idx]
                ego_pose = quat_scalar_last_to_pose_se3(
                    qx=ego_row["qx"],
                    qy=ego_row["qy"],
                    qz=ego_row["qz"],
                    qw=ego_row["qw"],
                    x=ego_row["x"],
                    y=ego_row["y"],
                    z=ego_row["z"],
                )
                camera_to_global = rel_to_abs_se3(origin=ego_pose, pose_se3=cam_metadata.camera_to_imu_se3)

                # Extract frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))  # type: ignore[arg-type]
                ret, frame = cap.read()
                if not ret:
                    continue
                _, jpeg_bytes = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

                yield ParsedCamera(
                    metadata=cam_metadata,
                    timestamp=Timestamp.from_us(cam_ts),
                    camera_to_global_se3=camera_to_global,
                    byte_string=bytes(jpeg_bytes),
                )
        finally:
            cap.release()


# ------------------------------------------------------------------------------------------------------------------
# Metadata helpers
# ------------------------------------------------------------------------------------------------------------------


def _get_ftheta_camera_metadatas(data_root: Path, clip_id: str, chunk: int) -> Dict[CameraID, FThetaCameraMetadata]:
    """Returns f-theta camera metadata for all cameras in a clip."""
    chunk_str = f"{chunk:04d}"
    intrinsics_path = data_root / "calibration" / "camera_intrinsics" / f"camera_intrinsics.chunk_{chunk_str}.parquet"
    extrinsics_path = data_root / "calibration" / "sensor_extrinsics" / f"sensor_extrinsics.chunk_{chunk_str}.parquet"

    intrinsics_df = pd.read_parquet(intrinsics_path)
    extrinsics_df = pd.read_parquet(extrinsics_path)

    metadatas: Dict[CameraID, FThetaCameraMetadata] = {}

    for cam_name, cam_id in PHYSICAL_AI_AV_CAMERA_ID_MAPPING.items():
        # Get intrinsics for this clip + camera
        try:
            cam_intr = intrinsics_df.loc[(clip_id, cam_name)]
        except KeyError:
            continue

        # Get extrinsics for this clip + camera
        try:
            cam_ext = extrinsics_df.loc[(clip_id, cam_name)]
        except KeyError:
            continue

        fw_poly = np.array(
            [
                cam_intr["fw_poly_0"],
                cam_intr["fw_poly_1"],
                cam_intr["fw_poly_2"],
                cam_intr["fw_poly_3"],
                cam_intr["fw_poly_4"],
            ],
            dtype=np.float64,
        )
        bw_poly = np.array(
            [
                cam_intr["bw_poly_0"],
                cam_intr["bw_poly_1"],
                cam_intr["bw_poly_2"],
                cam_intr["bw_poly_3"],
                cam_intr["bw_poly_4"],
            ],
            dtype=np.float64,
        )

        intrinsics = FThetaIntrinsics(
            cx=cam_intr["cx"],
            cy=cam_intr["cy"],
            fw_poly=fw_poly,
            bw_poly=bw_poly,
        )

        camera_to_imu = quat_scalar_last_to_pose_se3(
            qx=cam_ext["qx"],
            qy=cam_ext["qy"],
            qz=cam_ext["qz"],
            qw=cam_ext["qw"],
            x=cam_ext["x"],
            y=cam_ext["y"],
            z=cam_ext["z"],
        )

        metadatas[cam_id] = FThetaCameraMetadata(
            camera_name=cam_name,
            camera_id=cam_id,
            intrinsics=intrinsics,
            width=int(cam_intr["width"]),
            height=int(cam_intr["height"]),
            camera_to_imu_se3=camera_to_imu,
        )

    return metadatas


def _get_lidar_metadata(data_root: Path, clip_id: str, chunk: int) -> LidarMergedMetadata:
    """Returns LiDAR metadata for the clip."""
    chunk_str = f"{chunk:04d}"
    extrinsics_path = data_root / "calibration" / "sensor_extrinsics" / f"sensor_extrinsics.chunk_{chunk_str}.parquet"
    extrinsics_df = pd.read_parquet(extrinsics_path)

    try:
        lidar_ext = extrinsics_df.loc[(clip_id, "lidar_top_360fov")]
    except KeyError:
        # Fallback: identity pose
        lidar_ext = {"qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}

    lidar_to_imu = quat_scalar_last_to_pose_se3(
        qx=lidar_ext["qx"],
        qy=lidar_ext["qy"],
        qz=lidar_ext["qz"],
        qw=lidar_ext["qw"],
        x=lidar_ext["x"],
        y=lidar_ext["y"],
        z=lidar_ext["z"],
    )

    metadata_dict = {
        LidarID.LIDAR_TOP: LidarMetadata(
            lidar_name="lidar_top_360fov",
            lidar_id=LidarID.LIDAR_TOP,
            lidar_to_imu_se3=lidar_to_imu,
        ),
    }
    return LidarMergedMetadata(metadata_dict)


# ------------------------------------------------------------------------------------------------------------------
# Extraction helpers
# ------------------------------------------------------------------------------------------------------------------


def _extract_ego_state(
    ego_df: pd.DataFrame,
    ego_timestamps: np.ndarray,
    target_ts: int,
    metadata: EgoStateSE3Metadata,
) -> EgoStateSE3:
    """Extract the closest ego state to a target timestamp."""
    idx = find_closest_index(ego_timestamps, target_ts)
    row = ego_df.iloc[idx]

    ego_pose = quat_scalar_last_to_pose_se3(
        qx=row["qx"],
        qy=row["qy"],
        qz=row["qz"],
        qw=row["qw"],
        x=row["x"],
        y=row["y"],
        z=row["z"],
    )
    dynamic_state = DynamicStateSE3(
        velocity=Vector3D(x=row["vx"], y=row["vy"], z=row["vz"]),
        acceleration=Vector3D(x=row["ax"], y=row["ay"], z=row["az"]),
        angular_velocity=Vector3D(x=0.0, y=0.0, z=0.0),
    )
    return EgoStateSE3.from_imu(
        imu_se3=ego_pose,
        metadata=metadata,
        dynamic_state_se3=dynamic_state,
        timestamp=Timestamp.from_us(int(row["timestamp"])),
    )


def _extract_box_detections(
    obstacle_df: Optional[pd.DataFrame],
    obs_timestamps: Optional[np.ndarray],
    lidar_ts: int,
    ego_df: pd.DataFrame,
    ego_timestamps: np.ndarray,
    ego_metadata: EgoStateSE3Metadata,
    metadata: BoxDetectionsSE3Metadata,
    window_us: int = 50_000,
) -> BoxDetectionsSE3:
    """Extract obstacle detections within a time window around a lidar timestamp.

    Each detection in Physical AI AV has its own unique timestamp (not grouped into
    synchronous frames), so we gather all detections within ±window_us of the lidar timestamp.
    Each detection is transformed from ego frame to global using the ego pose at the
    detection's own timestamp, not the lidar timestamp.

    :param obstacle_df: DataFrame with obstacle detections, or None.
    :param obs_timestamps: Pre-computed obstacle timestamps array, or None.
    :param lidar_ts: Reference lidar timestamp in microseconds.
    :param ego_df: Egomotion DataFrame.
    :param ego_timestamps: Sorted egomotion timestamps array.
    :param ego_metadata: Ego state metadata.
    :param metadata: Box detections metadata.
    :param window_us: Half-window size in microseconds (default 50ms = half a lidar sweep).
    """
    timestamp = Timestamp.from_us(int(lidar_ts))

    if obstacle_df is None or obs_timestamps is None or len(obstacle_df) == 0:
        return BoxDetectionsSE3(box_detections=[], timestamp=timestamp, metadata=metadata)

    # Gather all detections within ±window_us of the lidar timestamp
    mask = (obs_timestamps >= lidar_ts - window_us) & (obs_timestamps < lidar_ts + window_us)
    group_df = obstacle_df[mask]

    return _build_box_detections(group_df, int(lidar_ts), ego_df, ego_timestamps, ego_metadata, metadata)


def _build_box_detections(
    group_df: pd.DataFrame,
    timestamp_us: int,
    ego_df: pd.DataFrame,
    ego_timestamps: np.ndarray,
    ego_metadata: EgoStateSE3Metadata,
    metadata: BoxDetectionsSE3Metadata,
) -> BoxDetectionsSE3:
    """Build BoxDetectionsSE3 from a group of obstacle detections at one timestamp.

    Each detection is transformed from the ego frame at the detection's own timestamp
    to the global frame, using the closest ego pose to that detection's timestamp.
    """
    timestamp = Timestamp.from_us(timestamp_us)
    num_dets = len(group_df)

    if num_dets == 0:
        return BoxDetectionsSE3(box_detections=[], timestamp=timestamp, metadata=metadata)

    det_state = np.zeros((num_dets, len(BoundingBoxSE3Index)), dtype=np.float64)
    det_velocity = np.zeros((num_dets, len(Vector3DIndex)), dtype=np.float64)
    det_labels: List[PhysicalAIAVBoxDetectionLabel] = []
    det_tokens: List[str] = []
    det_timestamps_us = group_df["timestamp_us"].to_numpy(dtype=np.int64)

    for det_idx, (_, det_row) in enumerate(group_df.iterrows()):
        # Position (in rig/ego frame at the detection's own timestamp)
        det_state[det_idx, BoundingBoxSE3Index.XYZ] = [
            det_row["center_x"],
            det_row["center_y"],
            det_row["center_z"],
        ]
        # Orientation: scalar-last (orientation_x/y/z/w) → scalar-first (qw, qx, qy, qz)
        det_state[det_idx, BoundingBoxSE3Index.QUATERNION] = [
            det_row["orientation_w"],
            det_row["orientation_x"],
            det_row["orientation_y"],
            det_row["orientation_z"],
        ]
        # Extent (size_x = length, size_y = width, size_z = height)
        det_state[det_idx, BoundingBoxSE3Index.EXTENT] = [
            det_row["size_x"],
            det_row["size_y"],
            det_row["size_z"],
        ]

        label_str = det_row["label_class"]
        label = PHYSICAL_AI_AV_LABEL_CLASS_MAPPING.get(label_str, PhysicalAIAVBoxDetectionLabel.OTHER_VEHICLE)
        det_labels.append(label)
        det_tokens.append(str(det_row["track_id"]))

    # Transform each detection from ego frame (at detection's own timestamp) to global
    for det_idx in range(num_dets):
        ego_idx = find_closest_index(ego_timestamps, int(det_timestamps_us[det_idx]))
        ego_row = ego_df.iloc[ego_idx]
        ego_pose = quat_scalar_last_to_pose_se3(
            qx=ego_row["qx"],
            qy=ego_row["qy"],
            qz=ego_row["qz"],
            qw=ego_row["qw"],
            x=ego_row["x"],
            y=ego_row["y"],
            z=ego_row["z"],
        )
        det_state[det_idx, BoundingBoxSE3Index.SE3] = rel_to_abs_se3_array(
            origin=ego_pose,
            pose_se3_array=det_state[det_idx : det_idx + 1, BoundingBoxSE3Index.SE3],
        )

    box_detections: List[BoxDetectionSE3] = []
    for det_idx in range(num_dets):
        box_detections.append(
            BoxDetectionSE3(
                attributes=BoxDetectionAttributes(
                    label=det_labels[det_idx],
                    track_token=det_tokens[det_idx],
                ),
                bounding_box_se3=BoundingBoxSE3.from_array(det_state[det_idx]),
                velocity_3d=Vector3D.from_array(det_velocity[det_idx]),
            )
        )

    return BoxDetectionsSE3(box_detections=box_detections, timestamp=timestamp, metadata=metadata)


def _extract_cameras(
    lidar_ts: int,
    ego_df: pd.DataFrame,
    ego_timestamps: np.ndarray,
    captures: Dict[str, cv2.VideoCapture],
    cam_timestamps: Dict[str, np.ndarray],
    ftheta_metadatas: Dict[CameraID, FThetaCameraMetadata],
) -> List[ParsedCamera]:
    """Extract camera frames closest to a lidar timestamp."""
    cameras: List[ParsedCamera] = []

    for cam_name, cam_ts_array in cam_timestamps.items():
        cam_id = PHYSICAL_AI_AV_CAMERA_ID_MAPPING.get(cam_name)
        if cam_id is None or cam_id not in ftheta_metadatas or cam_name not in captures:
            continue

        cam_metadata = ftheta_metadatas[cam_id]
        cap = captures[cam_name]

        # Find closest camera frame
        frame_idx = find_closest_index(cam_ts_array, lidar_ts)
        cam_ts = int(cam_ts_array[frame_idx])

        # Look up ego pose at the camera's own timestamp (not the lidar timestamp)
        ego_idx = find_closest_index(ego_timestamps, cam_ts)
        ego_row = ego_df.iloc[ego_idx]
        ego_pose = quat_scalar_last_to_pose_se3(
            qx=ego_row["qx"],
            qy=ego_row["qy"],
            qz=ego_row["qz"],
            qw=ego_row["qw"],
            x=ego_row["x"],
            y=ego_row["y"],
            z=ego_row["z"],
        )
        camera_to_global = rel_to_abs_se3(origin=ego_pose, pose_se3=cam_metadata.camera_to_imu_se3)

        # Extract frame from video
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Encode as JPEG
        _, jpeg_bytes = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        cameras.append(
            ParsedCamera(
                metadata=cam_metadata,
                timestamp=Timestamp.from_us(cam_ts),
                camera_to_global_se3=camera_to_global,
                byte_string=bytes(jpeg_bytes),
            )
        )

    return cameras
