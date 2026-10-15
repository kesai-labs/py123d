"""NuRec scenes joined with the sensors of their Physical AI AV clip."""

import json
import zipfile
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pytest

from py123d.datatypes import BoxDetectionsSE3, EgoStateSE3
from py123d.datatypes.sensors.lidar import Lidar
from py123d.parser.base_dataset_parser import ParsedCamera
from py123d.parser.nurec.nurec_pai_parser import NuRecPAILogParser, NuRecPAIParser, _clock_offset_us
from py123d.parser.nurec.nurec_parser import NuRecLogParser

from ..physical_ai_av.conftest import (
    CAMERA_NAMES,
    CAMERA_TIMESTAMPS_US,
    CLIP_ID,
    SPIN_STARTS_US,
    camera_frame_level,
    write_physical_ai_av_root,
)

RELEASE = "26.01"
SPLIT = "nurec-test_train"
SCENE_LISTS = {SPLIT: [f"{RELEASE}/{CLIP_ID}"]}
# The scene's clock runs ahead of the clip's clock.
CLOCK_OFFSET_US = 5_000_000_000
# The rig drives along x at 10 m/s, sampled every 100 ms over the clip's sensor window.
RIG_TIMESTAMPS_US = CLOCK_OFFSET_US + np.arange(0, 700_000, 100_000)
RIG_SPEED = 10.0


def rig_x_at(scene_time_us: int) -> float:
    """Rig position at a scene time; the motion is linear, so interpolation is exact."""
    return RIG_SPEED * (scene_time_us - CLOCK_OFFSET_US) / 1e6


def write_scene(root: Path, uuid: str, spin_starts_us: np.ndarray, clipgt: bool = False) -> Path:
    """Writes a minimal USDZ of one scene in the release layout, with one tracked cuboid alive throughout."""
    rig_poses = []
    for timestamp_us in RIG_TIMESTAMPS_US:
        pose = np.eye(4)
        pose[0, 3] = rig_x_at(int(timestamp_us))
        rig_poses.append(pose.tolist())
    rig = {
        "sequence_id": uuid,
        "rig_bbox": {"dim": [4.7, 2.0, 1.5], "centroid": [1.4, 0.0, 0.75]},
        "T_rig_worlds": rig_poses,
        "T_rig_world_timestamps_us": [int(t) for t in RIG_TIMESTAMPS_US],
        "lidars_frame_timestamps_us": {
            f"lidar_gt_top@{uuid}": [
                [int(t) + CLOCK_OFFSET_US, int(t) + CLOCK_OFFSET_US + 99_994] for t in spin_starts_us
            ]
        },
    }
    track_timestamps_us = [int(t) for t in RIG_TIMESTAMPS_US]
    tracks = {
        uuid: {
            "tracks_data": {
                "tracks_id": ["car-1"],
                "tracks_poses": [[[20.0, 3.0, 0.8, 0.0, 0.0, 0.0, 1.0] for _ in track_timestamps_us]],
                "tracks_timestamps_us": [track_timestamps_us],
                "tracks_label_class": ["automobile"],
            },
            "cuboidtracks_data": {"cuboids_dims": [[4.5, 1.9, 1.6]]},
        }
    }
    usdz_path = root / "sample_set" / f"{RELEASE}_release" / uuid / f"{uuid}.usdz"
    usdz_path.parent.mkdir(parents=True)
    with zipfile.ZipFile(usdz_path, "w") as archive:
        archive.writestr("rig_trajectories.json", json.dumps({"rig_trajectories": [rig]}))
        archive.writestr("sequence_tracks.json", json.dumps(tracks))
    return usdz_path


@pytest.fixture
def physical_ai_av_root(tmp_path: Path) -> Path:
    return write_physical_ai_av_root(tmp_path / "physical_ai_av", [CLIP_ID])


@pytest.fixture
def nurec_root(tmp_path: Path) -> Path:
    root = tmp_path / "nurec"
    write_scene(root, CLIP_ID, SPIN_STARTS_US)
    return root


def _parser(nurec_root: Path, physical_ai_av_root: Path, **kwargs) -> NuRecPAIParser:
    return NuRecPAIParser(
        physical_ai_av_data_root=physical_ai_av_root,
        nurec_root=nurec_root,
        splits=[SPLIT],
        scene_lists=SCENE_LISTS,
        min_traffic_duration_us=0,
        smooth_track_positions=False,
        **kwargs,
    )


def _frames(log_parser: NuRecPAILogParser) -> List[Dict[type, List]]:
    frames = []
    for frame in log_parser.iter_modalities_sync():
        by_type: Dict[type, List] = {}
        for modality in frame.modalities:
            by_type.setdefault(type(modality), []).append(modality)
        frames.append({"timestamp_us": frame.timestamp.time_us, **by_type})
    return frames


# ----------------------------------------------------------------------------------------------------------------------
# Scene selection
# ----------------------------------------------------------------------------------------------------------------------


def test_scene_with_clip_files_is_selected(nurec_root: Path, physical_ai_av_root: Path):
    parser = _parser(nurec_root, physical_ai_av_root)

    log_parsers = parser.get_log_parsers()
    assert len(log_parsers) == 1
    metadata = log_parsers[0].get_log_metadata()
    assert metadata.dataset == "nurec"
    assert metadata.split == SPLIT
    assert metadata.log_name == CLIP_ID
    assert metadata.map_metadata is None  # the scene carries no map source
    assert parser.get_map_parsers() == []


def test_scene_without_clip_is_converted_without_sensors(tmp_path: Path, nurec_root: Path, physical_ai_av_root: Path):
    other_uuid = "00000000-0000-4000-8000-000000000002"
    write_scene(nurec_root, other_uuid, SPIN_STARTS_US)
    scene_lists = {SPLIT: [f"{RELEASE}/{CLIP_ID}", f"{RELEASE}/{other_uuid}"]}

    parser = NuRecPAIParser(
        physical_ai_av_data_root=physical_ai_av_root, nurec_root=nurec_root, splits=[SPLIT], scene_lists=scene_lists
    )

    with_sensors, without_sensors = parser.get_log_parsers()
    assert isinstance(with_sensors, NuRecPAILogParser)
    assert with_sensors.get_log_metadata().log_name == CLIP_ID
    assert type(without_sensors) is NuRecLogParser
    assert without_sensors.get_log_metadata().log_name == other_uuid
    assert without_sensors.get_log_metadata().split == SPLIT
    frames = list(without_sensors.iter_modalities_sync())
    assert frames and all(len(frame.modalities) == 2 for frame in frames)  # ego state and boxes only


def test_no_scene_with_clip_files_raises(tmp_path: Path, nurec_root: Path, physical_ai_av_root: Path):
    (physical_ai_av_root / "lidar" / "lidar_top_360fov" / f"{CLIP_ID}.lidar_top_360fov.parquet").unlink()

    with pytest.raises(FileNotFoundError):
        _parser(nurec_root, physical_ai_av_root)


def test_unknown_camera_is_rejected(nurec_root: Path, physical_ai_av_root: Path):
    with pytest.raises(AssertionError):
        _parser(nurec_root, physical_ai_av_root, cameras=["camera_front_wide_120fov", "camera_top"])


# ----------------------------------------------------------------------------------------------------------------------
# Frames
# ----------------------------------------------------------------------------------------------------------------------


def test_frames_are_the_clip_spins_on_the_scene_clock(nurec_root: Path, physical_ai_av_root: Path):
    (log_parser,) = _parser(nurec_root, physical_ai_av_root).get_log_parsers()

    frames = _frames(log_parser)

    assert [frame["timestamp_us"] for frame in frames] == [int(t) + CLOCK_OFFSET_US for t in SPIN_STARTS_US]
    for spin_index, frame in enumerate(frames):
        (ego_state,) = frame[EgoStateSE3]
        assert ego_state.timestamp.time_us == frame["timestamp_us"]
        assert ego_state.imu_se3.x == pytest.approx(rig_x_at(frame["timestamp_us"]))
        assert ego_state.metadata.vehicle_name == "nurec_rig"

        (box_detections,) = frame[BoxDetectionsSE3]
        assert [box.attributes.track_token for box in box_detections.box_detections] == ["car-1"]

        (lidar,) = frame[Lidar]
        assert lidar.timestamp.time_us == frame["timestamp_us"]
        assert lidar.timestamp_end.time_us == frame["timestamp_us"] + 100_000
        assert lidar.point_cloud_3d.shape == (30, 3)  # the fixture encodes 30 points per spin


def test_cameras_are_posed_with_the_rig_at_their_own_time(nurec_root: Path, physical_ai_av_root: Path):
    (log_parser,) = _parser(nurec_root, physical_ai_av_root, jpeg_quality=100).get_log_parsers()

    frames = _frames(log_parser)

    for spin_start_us, frame in zip(SPIN_STARTS_US, frames):
        cameras = frame[ParsedCamera]
        assert {camera.metadata.camera_name for camera in cameras} == set(CAMERA_NAMES)
        frame_index = int(np.argmin(np.abs(CAMERA_TIMESTAMPS_US - spin_start_us)))
        for camera in cameras:
            scene_time_us = int(CAMERA_TIMESTAMPS_US[frame_index]) + CLOCK_OFFSET_US
            assert camera.timestamp.time_us == scene_time_us
            # The camera sits 1.5 m in front of the rig (see the calibration of the fixture).
            assert camera.camera_to_global_se3.x == pytest.approx(rig_x_at(scene_time_us) + 1.5)
            image = cv2.imdecode(np.frombuffer(camera._byte_string, dtype=np.uint8), cv2.IMREAD_COLOR)
            # The frames of the video differ by 12 gray levels; the codec shifts a level by a few.
            assert abs(float(image.mean()) - camera_frame_level(frame_index)) < 6


def test_camera_subset(nurec_root: Path, physical_ai_av_root: Path):
    (log_parser,) = _parser(nurec_root, physical_ai_av_root, cameras=[CAMERA_NAMES[0]]).get_log_parsers()

    frames = _frames(log_parser)

    assert all([camera.metadata.camera_name for camera in frame[ParsedCamera]] == [CAMERA_NAMES[0]] for frame in frames)


def test_async_conversion_is_not_supported(nurec_root: Path, physical_ai_av_root: Path):
    (log_parser,) = _parser(nurec_root, physical_ai_av_root).get_log_parsers()

    with pytest.raises(NotImplementedError):
        next(log_parser.iter_modalities_async())


# ----------------------------------------------------------------------------------------------------------------------
# Clock alignment
# ----------------------------------------------------------------------------------------------------------------------


def test_clock_offset_is_the_rig_start_confirmed_by_the_spins():
    rig_timestamps_us = [CLOCK_OFFSET_US, CLOCK_OFFSET_US + 100_000]
    clip_spins = np.arange(66_059, 20_000_000, 99_997, dtype=np.int64)

    assert _clock_offset_us(rig_timestamps_us, clip_spins + CLOCK_OFFSET_US, clip_spins) == CLOCK_OFFSET_US
    # The scene may list a subset of the clip's spins, and a few may be off.
    scene_spins = clip_spins[3:] + CLOCK_OFFSET_US
    scene_spins[10] += 50_000
    assert _clock_offset_us(rig_timestamps_us, scene_spins, clip_spins) == CLOCK_OFFSET_US


def test_spins_that_do_not_align_raise():
    rig_timestamps_us = [CLOCK_OFFSET_US, CLOCK_OFFSET_US + 100_000]
    clip_spins = np.arange(0, 2_000_000, 100_000, dtype=np.int64)

    with pytest.raises(ValueError):
        _clock_offset_us(rig_timestamps_us, clip_spins + CLOCK_OFFSET_US + 40_000, clip_spins)


def test_scene_with_shifted_spins_still_converts(tmp_path: Path, physical_ai_av_root: Path):
    # The scene lists the spins of the clip from the third one on, so the offset cannot come from the first spins.
    nurec_root = tmp_path / "nurec"
    write_scene(nurec_root, CLIP_ID, SPIN_STARTS_US[2:])
    (log_parser,) = _parser(nurec_root, physical_ai_av_root).get_log_parsers()

    frames = _frames(log_parser)

    assert [frame["timestamp_us"] for frame in frames] == [int(t) + CLOCK_OFFSET_US for t in SPIN_STARTS_US]
