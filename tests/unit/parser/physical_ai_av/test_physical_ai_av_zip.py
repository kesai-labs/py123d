"""Reading Physical AI AV from the chunk archives it is published in, instead of from extracted files."""

import zipfile
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd
import pytest

from py123d.common.io.lidar.path_lidar_io import load_point_cloud_data_from_path
from py123d.common.io.radar.path_radar_io import load_radar_point_cloud_data_from_path
from py123d.common.runtime import dataset_paths
from py123d.common.runtime.dataset_paths import DatasetPaths
from py123d.datatypes import BoxDetectionsSE3, EgoStateSE3
from py123d.datatypes.sensors.lidar import Lidar
from py123d.datatypes.sensors.radar import Radar
from py123d.parser.base_dataset_parser import ParsedCamera, ParsedLidar, ParsedRadar
from py123d.parser.physical_ai_av.physical_ai_av_parser import PhysicalAIAVParser
from py123d.parser.physical_ai_av.utils.physical_ai_av_zip import (
    extract_file,
    is_archive_member,
    open_file,
    path_exists,
    read_parquet,
    remove_extracted_files,
    resolve_clip_file,
)

from .conftest import CAMERA_NAMES, CHUNK, CLIP_ID, LIDAR_NAME, SPLIT, write_physical_ai_av_root


@pytest.fixture
def download_root(tmp_path: Path, monkeypatch) -> Path:
    """Points the download root, where videos of the archives are copied to, into the test directory."""
    data_root = tmp_path / "py123d"
    monkeypatch.setattr(dataset_paths, "_global_dataset_paths", DatasetPaths(py123d_data_root=data_root))
    return data_root / ".download"


# ----------------------------------------------------------------------------------------------------------------------
# Files
# ----------------------------------------------------------------------------------------------------------------------


def _clip_file(root: Path, group: str, sensor: str, extension: str) -> Path:
    return resolve_clip_file(root, group, sensor, CLIP_ID, extension, CHUNK)


def test_extracted_file_is_preferred(physical_ai_av_root: Path):
    path = _clip_file(physical_ai_av_root, "labels", "egomotion", "parquet")

    assert path == physical_ai_av_root / "labels" / "egomotion" / f"{CLIP_ID}.egomotion.parquet"
    assert not is_archive_member(path)
    assert path_exists(path)


def test_archive_member_is_found(physical_ai_av_zip_root: Path):
    path = _clip_file(physical_ai_av_zip_root, "labels", "egomotion", "parquet")

    assert path.parent == physical_ai_av_zip_root / "labels" / "egomotion" / "egomotion.chunk_0000.zip"
    assert is_archive_member(path)
    assert path_exists(path)


def test_missing_file(physical_ai_av_zip_root: Path):
    in_archive = resolve_clip_file(physical_ai_av_zip_root, "labels", "egomotion", "other-clip", "parquet", CHUNK)
    without_archive = resolve_clip_file(physical_ai_av_zip_root, "labels", "egomotion", CLIP_ID, "parquet", 7)

    for path in (in_archive, without_archive):
        assert not path_exists(path)
        with pytest.raises(FileNotFoundError):
            open_file(path)


@pytest.mark.parametrize("columns", [None, ["spin_start_timestamp"]])
def test_parquet_from_archive_equals_extracted(physical_ai_av_root: Path, physical_ai_av_zip_root: Path, columns):
    extracted = read_parquet(_clip_file(physical_ai_av_root, "lidar", LIDAR_NAME, "parquet"), columns=columns)
    from_archive = read_parquet(_clip_file(physical_ai_av_zip_root, "lidar", LIDAR_NAME, "parquet"), columns=columns)

    pd.testing.assert_frame_equal(from_archive, extracted)


def test_parquet_index_survives(physical_ai_av_zip_root: Path):
    """The camera timestamps are indexed by frame, which the parser relies on."""
    timestamps = read_parquet(_clip_file(physical_ai_av_zip_root, "camera", CAMERA_NAMES[0], "timestamps.parquet"))

    assert timestamps.index.name == "frame_index"


def test_archive_member_is_a_seekable_file(physical_ai_av_root: Path, physical_ai_av_zip_root: Path):
    expected = _clip_file(physical_ai_av_root, "camera", CAMERA_NAMES[0], "mp4").read_bytes()

    with open_file(_clip_file(physical_ai_av_zip_root, "camera", CAMERA_NAMES[0], "mp4")) as file:
        assert file.read() == expected
        file.seek(-10, 2)
        assert file.read() == expected[-10:]
        file.seek(5)
        assert file.read(7) == expected[5:12]
        assert file.tell() == 12


def test_compressed_archive_member_is_readable(tmp_path: Path):
    """The dataset is published without compression, an archive that was packed again may have it."""
    archive_path = tmp_path / "labels" / "egomotion" / "egomotion.chunk_0000.zip"
    archive_path.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{CLIP_ID}.egomotion.parquet", b"content" * 100)

    with open_file(_clip_file(tmp_path, "labels", "egomotion", "parquet")) as file:
        assert file.read() == b"content" * 100


def test_extract_and_remove(physical_ai_av_root: Path, physical_ai_av_zip_root: Path, tmp_path: Path):
    destination_dir = tmp_path / "download" / "physical-ai-av" / CLIP_ID
    other_clip_dir = tmp_path / "download" / "physical-ai-av" / "other-clip"

    copy_path = extract_file(_clip_file(physical_ai_av_zip_root, "camera", CAMERA_NAMES[0], "mp4"), destination_dir)
    assert copy_path.read_bytes() == _clip_file(physical_ai_av_root, "camera", CAMERA_NAMES[0], "mp4").read_bytes()

    other_clip_dir.mkdir()
    remove_extracted_files(destination_dir)
    assert not destination_dir.exists() and other_clip_dir.exists()

    remove_extracted_files(other_clip_dir)
    assert not (tmp_path / "download").exists()


# ----------------------------------------------------------------------------------------------------------------------
# Conversion
# ----------------------------------------------------------------------------------------------------------------------


def _content(modality: Any) -> List[Any]:
    """What a modality holds, with lidar and radar loaded if the parser left that to the log writer."""
    if isinstance(modality, EgoStateSE3):
        return ["ego", modality.timestamp.time_us, modality.imu_se3.array, modality.dynamic_state_se3.array]
    if isinstance(modality, BoxDetectionsSE3):
        boxes = [(box.attributes.track_token, box.bounding_box_se3.array) for box in modality.box_detections]
        return ["boxes", modality.timestamp.time_us, boxes]
    if isinstance(modality, ParsedCamera):
        pose = modality.camera_to_global_se3
        return ["camera", modality.metadata.camera_name, modality.timestamp.time_us, modality._byte_string, pose.array]
    if isinstance(modality, ParsedLidar):
        points, features = load_point_cloud_data_from_path(
            modality._relative_path,
            "physical-ai-av",
            modality._iteration,
            modality._dataset_root,
            lidar_metadatas=dict(modality.metadata),
        )
        return ["lidar", modality.timestamp.time_us, modality.end_timestamp.time_us, points, features]
    if isinstance(modality, Lidar):
        features = modality.point_cloud_features
        return ["lidar", modality.timestamp.time_us, modality.timestamp_end.time_us, modality.point_cloud_3d, features]
    if isinstance(modality, ParsedRadar):
        points, features = load_radar_point_cloud_data_from_path(
            modality._relative_path,
            "physical-ai-av",
            modality._iteration,
            modality._dataset_root,
            radar_metadatas=dict(modality.metadata),
            load_kwargs=modality.load_kwargs,
        )
        return ["radar", modality.timestamp.time_us, points, features]
    if isinstance(modality, Radar):
        return ["radar", modality.timestamp.time_us, modality.point_cloud_3d, modality.point_cloud_features]
    raise TypeError(type(modality))


def _assert_same(actual: Any, expected: Any) -> None:
    if isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_same(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_same(actual_item, expected_item)
    else:
        assert actual == expected


def _modalities(root: Path, async_conversion: bool) -> List[Any]:
    (log_parser,) = PhysicalAIAVParser(splits=[SPLIT], physical_ai_av_data_root=root).get_log_parsers()
    if async_conversion:
        return list(log_parser.iter_modalities_async())
    return [modality for frame in log_parser.iter_modalities_sync() for modality in frame.modalities]


@pytest.mark.parametrize("async_conversion", [False, True])
def test_archives_convert_like_extracted_files(
    physical_ai_av_root: Path, physical_ai_av_zip_root: Path, download_root: Path, async_conversion: bool
):
    """Every modality of a clip has the same content, whichever way the clip's files are stored."""
    from_files = _modalities(physical_ai_av_root, async_conversion)
    from_archives = _modalities(physical_ai_av_zip_root, async_conversion)

    assert {type(modality) for modality in from_files} >= {EgoStateSE3, ParsedCamera, ParsedLidar, ParsedRadar}
    assert {type(modality) for modality in from_archives} >= {EgoStateSE3, ParsedCamera, Lidar, Radar}
    assert len(from_archives) == len(from_files)
    for modality_from_archive, modality_from_file in zip(from_archives, from_files):
        _assert_same(_content(modality_from_archive), _content(modality_from_file))


@pytest.mark.parametrize("async_conversion", [False, True])
def test_videos_copied_out_of_the_archives_are_removed(
    physical_ai_av_zip_root: Path, download_root: Path, async_conversion: bool
):
    (log_parser,) = PhysicalAIAVParser(
        splits=[SPLIT], physical_ai_av_data_root=physical_ai_av_zip_root
    ).get_log_parsers()
    iterator = log_parser.iter_modalities_async() if async_conversion else log_parser.iter_modalities_sync()

    staged_videos = set()
    for _ in iterator:
        staged_videos.update(path.name for path in download_root.rglob("*.mp4"))

    assert staged_videos == {f"{CLIP_ID}.{camera_name}.mp4" for camera_name in CAMERA_NAMES}
    assert not download_root.exists()


def test_extracted_files_leave_the_download_root_alone(physical_ai_av_root: Path, download_root: Path):
    download_root.mkdir(parents=True)

    _modalities(physical_ai_av_root, async_conversion=False)

    assert download_root.exists() and not list(download_root.iterdir())


def test_incomplete_clips_are_found_in_the_archives(tmp_path: Path):
    """Clips without lidar are left out on request, also when the files are in chunk archives."""
    from .conftest import write_chunk_archives

    flat_root = write_physical_ai_av_root(tmp_path / "flat", ["clip-a", "clip-b"])
    (flat_root / "lidar" / LIDAR_NAME / f"clip-b.{LIDAR_NAME}.parquet").unlink()
    zip_root = write_chunk_archives(flat_root, tmp_path / "zip")

    parser = PhysicalAIAVParser(splits=[SPLIT], physical_ai_av_data_root=zip_root, skip_incomplete_clips=True)

    assert [log_parser.get_log_metadata().log_name for log_parser in parser.get_log_parsers()] == ["clip-a"]
