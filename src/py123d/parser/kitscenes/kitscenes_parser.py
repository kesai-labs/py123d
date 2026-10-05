"""KITScenes Multimodal dataset parser."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt

from py123d.datatypes import EgoStateSE3, LogMetadata, Timestamp
from py123d.datatypes.sensors.pinhole_camera import PinholeCameraMetadata, PinholeIntrinsics
from py123d.geometry import PoseSE3
from py123d.geometry.transform.transform_se3 import rel_to_abs_se3
from py123d.parser.base_dataset_parser import (
    BaseDatasetParser,
    BaseLogParser,
    BaseMapParser,
    ModalitiesSync,
    ParsedCamera,
)
from py123d.parser.kitscenes.kitscenes_constants import (
    CALIBRATION_FILE,
    CAMERA_ID_MAPPING,
    CITY_CENTERS,
    DATA_SUBDIR,
    DATASET_NAME,
    FRAME_INDEX_WIDTH,
    KITSCENES_EGO_STATE_SE3_METADATA,
    KITSCENES_SPLITS,
    MAP_ORIGIN_FILE,
    POSES_FILE,
)

logger = logging.getLogger(__name__)


class KITScenesDatasetParser(BaseDatasetParser):
    """Top-level parser for the KITScenes Multimodal dataset."""

    def __init__(
        self,
        kitscenes_data_root: Union[Path, str],
        splits: Optional[Sequence[str]] = None,
        scene_ids: Optional[Sequence[str]] = None,
    ) -> None:
        """Initialize the KITScenes dataset parser.

        :param kitscenes_data_root: Root of the HuggingFace download, containing ``data/<split>/<scene_uuid>/``.
        :param splits: KITScenes splits to convert (e.g. ``["train", "val"]``). When ``None``, all splits on disk.
        :param scene_ids: Optional subset of scene UUIDs. When ``None``, all scenes in the selected splits.
        """
        self._data_root = Path(kitscenes_data_root)
        assert self._data_root.is_dir(), f"`kitscenes_data_root` path {self._data_root} does not exist."

        if splits is not None:
            unknown_splits = set(splits) - set(KITSCENES_SPLITS)
            assert not unknown_splits, f"Unknown KITScenes splits {unknown_splits}, expected {KITSCENES_SPLITS}."
        self._splits = list(splits) if splits is not None else list(KITSCENES_SPLITS)
        self._scene_ids = set(scene_ids) if scene_ids else None

    def _collect_scenes(self) -> List[Tuple[str, str]]:
        """Collect ``(split, scene_id)`` pairs of all extracted scenes on disk."""
        scenes: List[Tuple[str, str]] = []
        for split in self._splits:
            split_dir = self._data_root / DATA_SUBDIR / split
            if not split_dir.is_dir():
                continue
            for scene_dir in sorted(split_dir.iterdir()):
                if not scene_dir.is_dir():
                    continue  # skips the downloaded ``.tar`` archives
                if self._scene_ids is not None and scene_dir.name not in self._scene_ids:
                    continue
                scenes.append((split, scene_dir.name))
        return scenes

    def get_log_parsers(self) -> List[BaseLogParser]:
        """Inherited, see superclass."""
        return [
            KITScenesLogParser(data_root=self._data_root, split=split, scene_id=scene_id)
            for split, scene_id in self._collect_scenes()
        ]

    def get_map_parsers(self) -> List[BaseMapParser]:
        """Inherited, see superclass."""
        # TODO: Convert the per-scene Lanelet2 maps (``maps/map.osm``).
        return []


class KITScenesLogParser(BaseLogParser):
    """Lightweight handle to one KITScenes scene. Heavy loading is deferred to the worker process."""

    def __init__(self, data_root: Path, split: str, scene_id: str) -> None:
        self._data_root = Path(data_root)
        self._split = split
        self._scene_id = scene_id

    @property
    def _scene_relative_dir(self) -> Path:
        return Path(DATA_SUBDIR) / self._split / self._scene_id

    @property
    def _scene_dir(self) -> Path:
        return self._data_root / self._scene_relative_dir

    def get_log_metadata(self) -> LogMetadata:
        """Inherited, see superclass."""
        return LogMetadata(
            dataset=DATASET_NAME,
            split=f"{DATASET_NAME}_{self._split}",
            log_name=self._scene_id,
            location=_infer_location(self._scene_dir / MAP_ORIGIN_FILE),
        )

    def iter_modalities_sync(self) -> Iterator[ModalitiesSync]:
        """Inherited, see superclass."""
        timestamps_ns, imu_to_global_poses = _load_poses(self._scene_dir / POSES_FILE)
        camera_metadatas = _load_camera_metadatas(self._scene_dir / CALIBRATION_FILE)

        # Every sensor stores exactly one file per reference frame, so frame ``i`` is the ``i``-th pose.
        for frame_index, (timestamp_ns, imu_to_global) in enumerate(zip(timestamps_ns, imu_to_global_poses)):
            timestamp = Timestamp.from_ns(int(timestamp_ns))
            ego_state = EgoStateSE3.from_imu(
                imu_se3=imu_to_global,
                metadata=KITSCENES_EGO_STATE_SE3_METADATA,
                timestamp=timestamp,
            )
            modalities = [ego_state]

            for camera_name, camera_metadata in camera_metadatas.items():
                relative_path = self._scene_relative_dir / camera_name / f"{frame_index:0{FRAME_INDEX_WIDTH}d}.jpg"
                if not (self._data_root / relative_path).is_file():
                    continue
                modalities.append(
                    ParsedCamera(
                        metadata=camera_metadata,
                        timestamp=timestamp,
                        camera_to_global_se3=rel_to_abs_se3(
                            origin=imu_to_global, pose_se3=camera_metadata.camera_to_imu_se3
                        ),
                        dataset_root=self._data_root,
                        relative_path=relative_path,
                    )
                )

            yield ModalitiesSync(timestamp=timestamp, modalities=modalities)


def _load_poses(poses_path: Path) -> Tuple[npt.NDArray[np.int64], List[PoseSE3]]:
    """Load the TUM-format ego trajectory (``timestamp tx ty tz qx qy qz qw``, timestamp in seconds).

    The poses describe the ego reference frame (= ``lidar_top``) in a local metric frame centered at the map origin.
    """
    poses = np.loadtxt(poses_path, dtype=np.float64, ndmin=2)
    timestamps_ns = np.round(poses[:, 0] * 1e9).astype(np.int64)
    imu_to_global_poses = [
        PoseSE3(x=x, y=y, z=z, qw=qw, qx=qx, qy=qy, qz=qz) for _, x, y, z, qx, qy, qz, qw in poses.tolist()
    ]
    return timestamps_ns, imu_to_global_poses


def _load_camera_metadatas(calibration_path: Path) -> Dict[str, PinholeCameraMetadata]:
    """Load the rectified pinhole camera calibrations from ``calib.json``."""
    with calibration_path.open("r", encoding="utf-8") as file:
        calibration = json.load(file)

    camera_metadatas: Dict[str, PinholeCameraMetadata] = {}
    for camera_name, camera_id in CAMERA_ID_MAPPING.items():
        entry = calibration.get(camera_name)
        if entry is None:
            logger.warning("Camera %s missing in %s.", camera_name, calibration_path)
            continue
        assert entry["camera_model"] == "pinhole", f"Unexpected camera model {entry['camera_model']}."
        intrinsics = entry["intrinsics"]
        camera_metadatas[camera_name] = PinholeCameraMetadata(
            camera_name=camera_name,
            camera_id=camera_id,
            intrinsics=PinholeIntrinsics(
                fx=intrinsics["focal_length"],
                fy=intrinsics["focal_length"],
                cx=intrinsics["principal_point_u"],
                cy=intrinsics["principal_point_v"],
            ),
            distortion=None,
            width=entry["resolution"]["width"],
            height=entry["resolution"]["height"],
            camera_to_imu_se3=PoseSE3.from_transformation_matrix(np.array(entry["T_to_reference"], dtype=np.float64)),
            is_undistorted=True,
        )
    return camera_metadatas


def _infer_location(map_origin_path: Path) -> Optional[str]:
    """Name the city closest to the scene's map origin, since KITScenes does not store it explicitly."""
    if not map_origin_path.is_file():
        return None
    with map_origin_path.open("r", encoding="utf-8") as file:
        origin = json.load(file)
    return min(
        CITY_CENTERS,
        key=lambda city: math.dist((origin["latitude"], origin["longitude"]), CITY_CENTERS[city]),
    )
