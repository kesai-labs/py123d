from __future__ import annotations

import bisect
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

from py123d.datatypes import (
    BoxDetectionAttributes,
    BoxDetectionSE3,
    BoxDetectionsSE3,
    EgoStateSE3,
    LogMetadata,
    PinholeCameraMetadata,
    PinholeDistortion,
    PinholeIntrinsics,
    Timestamp,
)
from py123d.geometry import BoundingBoxSE3, BoundingBoxSE3Index, EulerAnglesIndex, PoseSE3
from py123d.geometry.transform import rel_to_abs_se3, rel_to_abs_se3_array
from py123d.geometry.utils.constants import DEFAULT_PITCH, DEFAULT_ROLL
from py123d.geometry.utils.rotation_utils import get_quaternion_array_from_euler_array, slerp_quaternion_arrays
from py123d.parser.base_dataset_parser import (
    BaseDatasetParser,
    BaseLogParser,
    BaseMapParser,
    ModalitiesSync,
    ParsedCamera,
    ParsedLidar,
)
from py123d.parser.once.utils.once_constants import (
    ONCE_BOX_DETECTION_FROM_STR,
    ONCE_BOX_DETECTIONS_SE3_METADATA,
    ONCE_CAMERA_MAPPING,
    ONCE_EGO_STATE_SE3_METADATA,
    ONCE_LIDAR_MERGED_METADATA,
    ONCE_LIDAR_NAME,
    ONCE_LIDAR_SWEEP_DURATION_US,
    ONCE_MAX_INTERPOLATION_GAP_US,
    ONCE_SEQUENCES_BY_SPLIT,
    ONCE_SPLITS,
    ONCE_TO_ISO_ROTATION,
    ONCE_TO_ISO_TRANSFORM,
    ONCE_TRACK_MAX_SPEED_MPS,
)
from py123d.parser.registry import OnceBoxDetectionLabel

if TYPE_CHECKING:
    from py123d.parser.once.once_download import OnceDownloader

logger = logging.getLogger(__name__)


class OnceParser(BaseDatasetParser):
    """Dataset parser for the ONCE dataset (labelled splits)."""

    def __init__(
        self,
        splits: List[str],
        once_data_root: Optional[Union[Path, str]] = None,
        log_names: Optional[List[str]] = None,
        interpolate_labels: bool = True,
        downloader: Optional["OnceDownloader"] = None,
    ) -> None:
        """Initializes the :class:`OnceParser`.

        :param splits: List of splits to include in the conversion. \
            Available splits: 'once_train', 'once_val', 'once_test'.
        :param once_data_root: Path to the root directory of the ONCE dataset, which contains ``data/<sequence>/``.
            Required when ``downloader`` is ``None``; ignored otherwise.
        :param log_names: Optional list of sequence ids to convert. ``None`` converts all sequences of the splits.
        :param interpolate_labels: ONCE annotates every second released frame (1 Hz, sensors at 2 Hz). When true,
            boxes are tracked across annotated frames and interpolated at the unannotated frames in between.
            When false, unannotated frames carry no box detections.
        :param downloader: Optional :class:`~py123d.parser.once.once_download.OnceDownloader` for streaming mode.
            The archives of ``splits`` are downloaded and extracted into ``downloader.output_dir`` before the
            conversion, and that directory is used as ``once_data_root``.
        """
        for split in splits:
            assert split in ONCE_SPLITS, f"Split {split} is not available. Available splits: {ONCE_SPLITS}"

        self._splits = list(splits)
        self._log_names = list(log_names) if log_names is not None else None
        self._interpolate_labels = interpolate_labels

        if downloader is not None:
            downloader.splits = list(self._splits)
            downloader.download()
            assert downloader.output_dir is not None, "`downloader.output_dir` must be set for streaming."
            self._once_data_root = Path(downloader.output_dir)
        else:
            assert once_data_root is not None, "`once_data_root` must be provided when `downloader` is None."
            self._once_data_root = Path(once_data_root)

        self._log_entries: List[Tuple[str, str]] = self._collect_log_entries()

    def _collect_log_entries(self) -> List[Tuple[str, str]]:
        """Collects the (split, sequence id) pairs that exist under ``once_data_root/data``."""
        entries: List[Tuple[str, str]] = []
        for split in self._splits:
            for log_name in ONCE_SEQUENCES_BY_SPLIT[split]:
                if self._log_names is not None and log_name not in self._log_names:
                    continue
                if not (self._once_data_root / "data" / log_name / f"{log_name}.json").exists():
                    logger.warning(
                        "ONCE sequence %s (%s) not found under %s, skipping.", log_name, split, self._once_data_root
                    )
                    continue
                entries.append((split, log_name))
        return entries

    def get_log_parsers(self) -> List[OnceLogParser]:  # type: ignore
        """Inherited, see superclass."""
        return [
            OnceLogParser(
                once_data_root=self._once_data_root,
                log_name=log_name,
                split=split,
                interpolate_labels=self._interpolate_labels,
            )
            for split, log_name in self._log_entries
        ]

    def get_map_parsers(self) -> List[BaseMapParser]:
        """Inherited, see superclass."""
        return []  # ONCE does not have maps.


class OnceLogParser(BaseLogParser):
    """Lightweight, picklable handle to one ONCE sequence."""

    def __init__(self, once_data_root: Path, log_name: str, split: str, interpolate_labels: bool = True) -> None:
        self._once_data_root = Path(once_data_root)
        self._log_name = log_name
        self._split = split
        self._interpolate_labels = interpolate_labels

    def get_log_metadata(self) -> LogMetadata:
        """Inherited, see superclass."""
        return LogMetadata(dataset="once", split=self._split, log_name=self._log_name, location=None)

    def iter_modalities_sync(self) -> Iterator[ModalitiesSync]:
        """Inherited, see superclass."""
        log_dir = self._once_data_root / "data" / self._log_name
        with open(log_dir / f"{self._log_name}.json") as f:
            sequence = json.load(f)

        frames = sorted(sequence["frames"], key=lambda frame: int(frame["frame_id"]))
        timestamps_us = [int(frame["frame_id"]) * 1000 for frame in frames]
        ego_poses = [_once_pose_to_global_iso(frame["pose"]) for frame in frames]
        cameras_metadata = _get_once_camera_metadata(sequence, log_dir)

        # Boxes of the annotated frames in the global frame, with track tokens from frame-to-frame association.
        annotated_indices = [idx for idx, frame in enumerate(frames) if "annos" in frame]
        annotated_boxes = [
            _once_annos_to_global_boxes(frames[idx]["annos"], ego_poses[idx]) for idx in annotated_indices
        ]
        annotated_tokens = _assign_track_tokens(
            self._log_name, annotated_boxes, [timestamps_us[idx] for idx in annotated_indices]
        )

        for idx, frame in enumerate(frames):
            timestamp = Timestamp.from_us(timestamps_us[idx])
            frame_id = frame["frame_id"]
            ego_state = EgoStateSE3.from_imu(
                imu_se3=ego_poses[idx], metadata=ONCE_EGO_STATE_SE3_METADATA, timestamp=timestamp
            )

            lidar_relative_path = Path("data") / self._log_name / ONCE_LIDAR_NAME / f"{frame_id}.bin"
            assert (self._once_data_root / lidar_relative_path).exists(), (
                f"ONCE lidar file {self._once_data_root / lidar_relative_path} does not exist."
            )
            parsed_lidar = ParsedLidar(
                metadata=ONCE_LIDAR_MERGED_METADATA,
                start_timestamp=timestamp,
                end_timestamp=Timestamp.from_us(timestamps_us[idx] + ONCE_LIDAR_SWEEP_DURATION_US),
                iteration=idx,
                dataset_root=self._once_data_root,
                relative_path=str(lidar_relative_path),
            )

            parsed_cameras: List[ParsedCamera] = []
            for camera_name, camera_metadata in cameras_metadata.items():
                image_relative_path = Path("data") / self._log_name / camera_name / f"{frame_id}.jpg"
                if not (self._once_data_root / image_relative_path).exists():
                    continue
                parsed_cameras.append(
                    ParsedCamera(
                        metadata=camera_metadata,
                        timestamp=timestamp,
                        camera_to_global_se3=rel_to_abs_se3(ego_poses[idx], camera_metadata.camera_to_imu_se3),
                        dataset_root=self._once_data_root,
                        relative_path=image_relative_path,
                    )
                )

            modalities = [ego_state, parsed_lidar, *parsed_cameras]
            box_detections = self._get_box_detections(
                idx, timestamp, timestamps_us, annotated_indices, annotated_boxes, annotated_tokens
            )
            if box_detections is not None:
                modalities.append(box_detections)

            yield ModalitiesSync(timestamp=timestamp, modalities=modalities)

    def _get_box_detections(
        self,
        idx: int,
        timestamp: Timestamp,
        timestamps_us: List[int],
        annotated_indices: List[int],
        annotated_boxes: List[Tuple[np.ndarray, List[OnceBoxDetectionLabel]]],
        annotated_tokens: List[List[str]],
    ) -> Optional[BoxDetectionsSE3]:
        """Returns the (annotated or interpolated) boxes of frame ``idx``, or ``None`` if the frame has none."""
        position = bisect.bisect_left(annotated_indices, idx)
        if position < len(annotated_indices) and annotated_indices[position] == idx:
            boxes, labels = annotated_boxes[position]
            return _build_box_detections(boxes, labels, annotated_tokens[position], timestamp)

        if not self._interpolate_labels or position == 0 or position == len(annotated_indices):
            return None

        prev_idx, next_idx = annotated_indices[position - 1], annotated_indices[position]
        gap_us = timestamps_us[next_idx] - timestamps_us[prev_idx]
        if gap_us > ONCE_MAX_INTERPOLATION_GAP_US:
            return None

        t = (timestamps_us[idx] - timestamps_us[prev_idx]) / gap_us
        boxes, labels, tokens = _interpolate_boxes(
            *annotated_boxes[position - 1],
            annotated_tokens[position - 1],
            *annotated_boxes[position],
            annotated_tokens[position],
            t,
        )
        return _build_box_detections(boxes, labels, tokens, timestamp)


def _once_pose_to_global_iso(pose: List[float]) -> PoseSE3:
    """Converts an ONCE frame pose (qx, qy, qz, qw, tx, ty, tz) of the lidar relative to the first frame of the
    sequence into an ISO 8855 pose relative to the ISO frame of the first frame."""
    transformation = np.eye(4, dtype=np.float64)
    transformation[:3, :3] = Rotation.from_quat(pose[:4]).as_matrix()
    transformation[:3, 3] = pose[4:]
    # Change both the global and the body frame from the ONCE lidar convention to ISO 8855.
    return PoseSE3.from_transformation_matrix(ONCE_TO_ISO_TRANSFORM @ transformation @ ONCE_TO_ISO_TRANSFORM.T)


def _get_once_camera_metadata(sequence: Dict, log_dir: Path) -> Dict[str, PinholeCameraMetadata]:
    """Builds the camera metadata from the sequence calibration, for the cameras that exist on disk."""
    width, height = sequence["meta_info"]["image_size"]
    cameras_metadata: Dict[str, PinholeCameraMetadata] = {}
    for camera_name, camera_id in ONCE_CAMERA_MAPPING.items():
        if camera_name not in sequence["calib"] or not (log_dir / camera_name).is_dir():
            continue
        calib = sequence["calib"][camera_name]
        intrinsic = np.array(calib["cam_intrinsic"], dtype=np.float64)
        k1, k2, p1, p2, k3 = (list(calib["distortion"]) + [0.0] * 5)[:5]
        cameras_metadata[camera_name] = PinholeCameraMetadata(
            camera_name=camera_name,
            camera_id=camera_id,
            width=int(width),
            height=int(height),
            intrinsics=PinholeIntrinsics(
                fx=intrinsic[0, 0], fy=intrinsic[1, 1], cx=intrinsic[0, 2], cy=intrinsic[1, 2]
            ),
            distortion=PinholeDistortion(k1=k1, k2=k2, p1=p1, p2=p2, k3=k3),
            camera_to_imu_se3=PoseSE3.from_transformation_matrix(
                ONCE_TO_ISO_TRANSFORM @ np.array(calib["cam_to_velo"], dtype=np.float64)
            ),
            is_undistorted=True,
        )
    return cameras_metadata


def _once_annos_to_global_boxes(annos: Dict, ego_pose: PoseSE3) -> Tuple[np.ndarray, List[OnceBoxDetectionLabel]]:
    """Converts the ``annos`` of a frame to a ``(N, 10)`` :class:`BoundingBoxSE3Index` array in the global frame."""
    labels = [ONCE_BOX_DETECTION_FROM_STR[name] for name in annos["names"]]
    boxes_3d = np.asarray(annos["boxes_3d"], dtype=np.float64).reshape(-1, 7)  # cx, cy, cz, l, w, h, yaw
    num_boxes = len(boxes_3d)

    # The box length is along the box x-axis in both conventions, so rotating the lidar frame into ISO 8855
    # moves the center and adds 90° to the yaw.
    euler_angles = np.zeros((num_boxes, len(EulerAnglesIndex)), dtype=np.float64)
    euler_angles[:, EulerAnglesIndex.ROLL] = DEFAULT_ROLL
    euler_angles[:, EulerAnglesIndex.PITCH] = DEFAULT_PITCH
    euler_angles[:, EulerAnglesIndex.YAW] = boxes_3d[:, 6] + np.pi / 2

    boxes = np.zeros((num_boxes, len(BoundingBoxSE3Index)), dtype=np.float64)
    boxes[:, BoundingBoxSE3Index.XYZ] = boxes_3d[:, :3] @ ONCE_TO_ISO_ROTATION.T
    boxes[:, BoundingBoxSE3Index.QUATERNION] = get_quaternion_array_from_euler_array(euler_angles)
    boxes[:, BoundingBoxSE3Index.EXTENT] = boxes_3d[:, 3:6]
    boxes[:, BoundingBoxSE3Index.SE3] = rel_to_abs_se3_array(ego_pose, boxes[:, BoundingBoxSE3Index.SE3])
    return boxes, labels


def _assign_track_tokens(
    log_name: str,
    annotated_boxes: List[Tuple[np.ndarray, List[OnceBoxDetectionLabel]]],
    annotated_timestamps_us: List[int],
) -> List[List[str]]:
    """Associates the boxes of consecutive annotated frames into tracks and returns a token per box.

    ONCE has no track ids. Boxes of the same class are matched between consecutive annotated frames (Hungarian
    assignment on the distance between the constant-velocity prediction and the box center, with a class-specific
    speed gate and a penalty for differing extents). Unmatched boxes start new tracks.
    """
    tracks: Dict[str, Dict] = {}  # token -> {"label", "center", "velocity", "extent", "timestamp_us"}
    tokens_per_frame: List[List[str]] = []
    num_tracks = 0

    for (boxes, labels), timestamp_us in zip(annotated_boxes, annotated_timestamps_us):
        tokens: List[Optional[str]] = [None] * len(boxes)
        centers = boxes[:, BoundingBoxSE3Index.XYZ]
        extents = boxes[:, BoundingBoxSE3Index.EXTENT]

        active = [
            token
            for token, track in tracks.items()
            if timestamp_us - track["timestamp_us"] <= ONCE_MAX_INTERPOLATION_GAP_US
        ]
        if active and len(boxes) > 0:
            cost = np.full((len(active), len(boxes)), np.inf, dtype=np.float64)
            for row, token in enumerate(active):
                track = tracks[token]
                dt_s = (timestamp_us - track["timestamp_us"]) / 1e6
                predicted = track["center"] + track["velocity"] * dt_s
                displacement = np.linalg.norm(centers - track["center"], axis=-1)
                prediction_error = np.linalg.norm(centers - predicted, axis=-1)
                extent_error = np.abs(extents - track["extent"]).sum(axis=-1)
                same_label = np.array([label == track["label"] for label in labels])
                gated = same_label & (displacement <= ONCE_TRACK_MAX_SPEED_MPS[track["label"]] * dt_s)
                cost[row, gated] = prediction_error[gated] + extent_error[gated]

            finite = np.where(np.isfinite(cost), cost, 1e9)
            for row, col in zip(*linear_sum_assignment(finite)):
                if not np.isfinite(cost[row, col]):
                    continue
                token = active[row]
                track = tracks[token]
                dt_s = (timestamp_us - track["timestamp_us"]) / 1e6
                track["velocity"] = (centers[col] - track["center"]) / dt_s
                track["center"], track["extent"], track["timestamp_us"] = centers[col], extents[col], timestamp_us
                tokens[col] = token

        for col in range(len(boxes)):
            if tokens[col] is None:
                token = f"{log_name}_{num_tracks:06d}"
                num_tracks += 1
                tracks[token] = {
                    "label": labels[col],
                    "center": centers[col],
                    "velocity": np.zeros(3, dtype=np.float64),
                    "extent": extents[col],
                    "timestamp_us": timestamp_us,
                }
                tokens[col] = token
        tokens_per_frame.append(tokens)  # type: ignore
    return tokens_per_frame


def _interpolate_boxes(
    prev_boxes: np.ndarray,
    prev_labels: List[OnceBoxDetectionLabel],
    prev_tokens: List[str],
    next_boxes: np.ndarray,
    next_labels: List[OnceBoxDetectionLabel],
    next_tokens: List[str],
    t: float,
) -> Tuple[np.ndarray, List[OnceBoxDetectionLabel], List[str]]:
    """Interpolates the boxes of tracks present in both frames (linear center and extent, SLERP rotation)."""
    next_index = {token: col for col, token in enumerate(next_tokens)}
    pairs = [(row, next_index[token]) for row, token in enumerate(prev_tokens) if token in next_index]
    if not pairs:
        return np.zeros((0, len(BoundingBoxSE3Index)), dtype=np.float64), [], []

    prev_rows, next_rows = (np.array(rows) for rows in zip(*pairs))
    prev, nxt = prev_boxes[prev_rows], next_boxes[next_rows]
    boxes = np.zeros((len(pairs), len(BoundingBoxSE3Index)), dtype=np.float64)
    boxes[:, BoundingBoxSE3Index.XYZ] = (1 - t) * prev[:, BoundingBoxSE3Index.XYZ] + t * nxt[:, BoundingBoxSE3Index.XYZ]
    boxes[:, BoundingBoxSE3Index.QUATERNION] = slerp_quaternion_arrays(
        prev[:, BoundingBoxSE3Index.QUATERNION], nxt[:, BoundingBoxSE3Index.QUATERNION], np.full(len(pairs), t)
    )
    boxes[:, BoundingBoxSE3Index.EXTENT] = (1 - t) * prev[:, BoundingBoxSE3Index.EXTENT] + t * nxt[
        :, BoundingBoxSE3Index.EXTENT
    ]
    return boxes, [prev_labels[row] for row in prev_rows], [prev_tokens[row] for row in prev_rows]


def _build_box_detections(
    boxes: np.ndarray, labels: List[OnceBoxDetectionLabel], tokens: List[str], timestamp: Timestamp
) -> BoxDetectionsSE3:
    """Wraps a ``(N, 10)`` box array into :class:`BoxDetectionsSE3`."""
    box_detections = [
        BoxDetectionSE3(
            attributes=BoxDetectionAttributes(label=label, track_token=token),
            bounding_box_se3=BoundingBoxSE3.from_array(box),
            velocity_3d=None,
        )
        for box, label, token in zip(boxes, labels, tokens)
    ]
    return BoxDetectionsSE3(
        box_detections=box_detections, timestamp=timestamp, metadata=ONCE_BOX_DETECTIONS_SE3_METADATA
    )
