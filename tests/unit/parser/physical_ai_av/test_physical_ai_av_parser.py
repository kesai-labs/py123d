"""The Physical AI AV parser on a small synthetic clip."""

import logging
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pytest

from py123d.datatypes import EgoStateSE3
from py123d.datatypes.modalities.base_modality import BaseModality
from py123d.parser.base_dataset_parser import ParsedCamera
from py123d.parser.physical_ai_av.physical_ai_av_parser import (
    PhysicalAIAVLogParser,
    PhysicalAIAVParser,
    _SequentialFrameReader,
)

from .conftest import (
    ACCELERATION,
    CAMERA_NAMES,
    CAMERA_TIMESTAMPS_US,
    CLIP_ID,
    NUM_CAMERA_FRAMES,
    SPEED,
    SPIN_STARTS_US,
    SPLIT,
    write_physical_ai_av_root,
)


def _log_parser(root: Path) -> PhysicalAIAVLogParser:
    (log_parser,) = PhysicalAIAVParser(splits=[SPLIT], physical_ai_av_data_root=root).get_log_parsers()
    return log_parser


def _modalities(root: Path, async_conversion: bool) -> List[BaseModality]:
    """Everything the log parser yields, in order, for either conversion mode."""
    log_parser = _log_parser(root)
    if async_conversion:
        return list(log_parser.iter_modalities_async())
    return [modality for frame in log_parser.iter_modalities_sync() for modality in frame.modalities]


@pytest.mark.parametrize("async_conversion", [False, True])
def test_ego_dynamics_are_in_the_ego_frame(physical_ai_av_root: Path, async_conversion: bool):
    """Driving straight, velocity and acceleration point along the ego's x axis, whatever its heading.

    The dataset gives both in the odometry frame, where they point along the heading.
    """
    ego_states = [m for m in _modalities(physical_ai_av_root, async_conversion) if isinstance(m, EgoStateSE3)]

    assert len(ego_states) > 0
    for ego_state in ego_states:
        speed = SPEED + ACCELERATION * ego_state.timestamp.time_us / 1e6
        dynamic_state = ego_state.dynamic_state_se3
        np.testing.assert_allclose(dynamic_state.velocity_3d.array, [speed, 0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(dynamic_state.acceleration_3d.array, [ACCELERATION, 0.0, 0.0], atol=1e-9)


def _video_path(root: Path, camera_name: str) -> Path:
    return root / "camera" / camera_name / f"{CLIP_ID}.{camera_name}.mp4"


def _frames_by_seeking(video_path: Path) -> List[np.ndarray]:
    """Every frame of the video, each read after a seek to its index."""
    capture = cv2.VideoCapture(str(video_path))
    frames = []
    for frame_index in range(NUM_CAMERA_FRAMES):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = capture.read()
        assert success
        frames.append(frame)
    capture.release()
    return frames


def _frame_index_of(image: np.ndarray, reference_frames: List[np.ndarray]) -> int:
    """Index of the video frame an image shows; every frame of the test videos has its own gray level."""
    return int(np.argmin([abs(float(image.mean()) - float(frame.mean())) for frame in reference_frames]))


@pytest.mark.parametrize(
    "requested",
    [
        list(range(NUM_CAMERA_FRAMES)),  # every frame, as in the asynchronous conversion
        [0, 3, 6, 9, 12, 15],  # one frame per lidar spin, as in the synchronous conversion
        [4, 4, 5, 5, 17],  # the same frame twice
        [10, 2, 3, 16, 0],  # backwards
    ],
)
def test_sequential_reader_returns_the_requested_frames(physical_ai_av_root: Path, requested: List[int]):
    video_path = _video_path(physical_ai_av_root, CAMERA_NAMES[0])
    expected_frames = _frames_by_seeking(video_path)

    reader = _SequentialFrameReader(video_path)
    for frame_index in requested:
        np.testing.assert_array_equal(reader.read_at(frame_index), expected_frames[frame_index])
    reader.release()


def test_sequential_reader_past_the_end(physical_ai_av_root: Path):
    reader = _SequentialFrameReader(_video_path(physical_ai_av_root, CAMERA_NAMES[0]))

    assert reader.read_at(NUM_CAMERA_FRAMES - 1) is not None
    assert reader.read_at(NUM_CAMERA_FRAMES) is None
    reader.release()


@pytest.mark.parametrize("async_conversion", [False, True])
def test_cameras_show_the_frame_of_their_timestamp(physical_ai_av_root: Path, async_conversion: bool):
    """Every converted image is the video frame its timestamp belongs to."""
    reference_frames = _frames_by_seeking(_video_path(physical_ai_av_root, CAMERA_NAMES[0]))
    cameras = [m for m in _modalities(physical_ai_av_root, async_conversion) if isinstance(m, ParsedCamera)]

    if async_conversion:  # every frame of every camera
        expected_indices = list(range(NUM_CAMERA_FRAMES)) * len(CAMERA_NAMES)
    else:  # per lidar spin, the frame nearest to the spin start
        nearest = [int(np.argmin(np.abs(CAMERA_TIMESTAMPS_US - spin_start))) for spin_start in SPIN_STARTS_US]
        expected_indices = [index for index in nearest for _ in CAMERA_NAMES]

    assert len(cameras) == len(expected_indices)
    for camera, expected_index in zip(cameras, expected_indices):
        image = cv2.imdecode(np.frombuffer(camera._byte_string, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert _frame_index_of(image, reference_frames) == expected_index
        assert camera.timestamp.time_us == CAMERA_TIMESTAMPS_US[expected_index]


@pytest.mark.parametrize(
    "missing_file",
    [
        "lidar/lidar_top_360fov/{clip_id}.lidar_top_360fov.parquet",
        "labels/egomotion/{clip_id}.egomotion.parquet",
    ],
)
class TestSkipIncompleteClips:
    """Some clips of the dataset come without the lidar or the egomotion file."""

    @staticmethod
    def _root_with_incomplete_clip(tmp_path: Path, missing_file: str) -> Path:
        root = write_physical_ai_av_root(tmp_path / "physical_ai_av", ["clip-a", "clip-b", "clip-c"])
        (root / missing_file.format(clip_id="clip-b")).unlink()
        return root

    def test_incomplete_clips_are_kept_by_default(self, tmp_path: Path, missing_file: str):
        root = self._root_with_incomplete_clip(tmp_path, missing_file)

        log_parsers = PhysicalAIAVParser(splits=[SPLIT], physical_ai_av_data_root=root).get_log_parsers()

        assert [log_parser.get_log_metadata().log_name for log_parser in log_parsers] == ["clip-a", "clip-b", "clip-c"]

    def test_incomplete_clips_are_left_out_on_request(self, tmp_path: Path, missing_file: str, caplog):
        root = self._root_with_incomplete_clip(tmp_path, missing_file)

        with caplog.at_level(logging.WARNING):
            parser = PhysicalAIAVParser(splits=[SPLIT], physical_ai_av_data_root=root, skip_incomplete_clips=True)

        assert [log_parser.get_log_metadata().log_name for log_parser in parser.get_log_parsers()] == [
            "clip-a",
            "clip-c",
        ]
        assert "Skipping 1 clips without lidar or egomotion data." in caplog.text

    def test_skipped_clips_do_not_count_towards_max_clips(self, tmp_path: Path, missing_file: str):
        root = self._root_with_incomplete_clip(tmp_path, missing_file)

        parser = PhysicalAIAVParser(
            splits=[SPLIT], physical_ai_av_data_root=root, max_clips=2, skip_incomplete_clips=True
        )

        assert [log_parser.get_log_metadata().log_name for log_parser in parser.get_log_parsers()] == [
            "clip-a",
            "clip-c",
        ]
