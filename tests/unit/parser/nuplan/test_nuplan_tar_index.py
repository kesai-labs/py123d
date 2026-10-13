"""The index of the nuPlan sensor archives, which lets their files be read without extracting them."""

import io
import tarfile
from pathlib import Path
from typing import Dict

import pytest

from py123d.parser.nuplan import nuplan_tar_index
from py123d.parser.nuplan.nuplan_tar_index import NuplanTarIndex, TarMemberReader, build_tar_index, load_log_members

LOG_A = "2021.05.25.14.24.08_veh-25_00934_01067"
LOG_B = "2021.05.25.14.16.10_veh-35_01100_01664"

CAMERA_FILES = {
    f"{LOG_A}/CAM_F0/aa.jpg": b"jpeg-bytes-aa",
    f"{LOG_A}/CAM_L0/bb.jpg": b"jpeg-bytes-bb-longer",
    f"{LOG_B}/CAM_F0/cc.jpg": b"jpeg-bytes-cc",
}
# With its wrapper folder, this path is longer than the 100 characters a plain tar header holds.
LIDAR_FILES = {f"{LOG_A}/MergedPointCloud/9e56754b1c205e17.pcd": b"pcd-bytes"}


def write_sensor_archive(path: Path, files: Dict[str, bytes]) -> Path:
    """Writes an uncompressed tar archive in the layout of a nuPlan sensor shard.

    :param path: Path of the archive, named like the shard, e.g. ``nuplan-v1.1_test_camera_0.zip``.
    :param files: Content of the files, by their path below the wrapper folder.
    :return: The path of the archive.
    """
    wrapper_folder = path.name[: -len(".zip")]
    with tarfile.open(path, "w") as archive:
        for name, content in files.items():
            member = tarfile.TarInfo(f"{wrapper_folder}/{name}")
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    return path


@pytest.fixture
def camera_archive(tmp_path: Path) -> Path:
    return write_sensor_archive(tmp_path / "nuplan-v1.1_test_camera_0.zip", CAMERA_FILES)


@pytest.fixture
def lidar_archive(tmp_path: Path) -> Path:
    return write_sensor_archive(tmp_path / "nuplan-v1.1_test_lidar_0.zip", LIDAR_FILES)


def test_indexed_files_are_read_back(camera_archive: Path):
    index_path = build_tar_index(camera_archive)

    assert index_path == camera_archive.with_name(camera_archive.name + ".index.parquet")
    assert camera_archive.with_name(camera_archive.name + ".logs.txt").read_text().splitlines() == [LOG_B, LOG_A]

    members = load_log_members(LOG_A, [camera_archive])
    assert set(members) == {f"{LOG_A}/CAM_F0/aa.jpg", f"{LOG_A}/CAM_L0/bb.jpg"}
    reader = TarMemberReader()
    for relative_path, member in members.items():
        assert reader.read(*member) == CAMERA_FILES[relative_path]
    reader.close()


def test_long_file_names_are_indexed(lidar_archive: Path):
    build_tar_index(lidar_archive)

    members = load_log_members(LOG_A, [lidar_archive])

    (relative_path,) = LIDAR_FILES
    assert TarMemberReader().read(*members[relative_path]) == LIDAR_FILES[relative_path]


def test_archives_of_a_log(camera_archive: Path, lidar_archive: Path, tmp_path: Path):
    build_tar_index(camera_archive)
    build_tar_index(lidar_archive)

    index = NuplanTarIndex(tmp_path)

    assert index.log_names == {LOG_A, LOG_B}
    assert index.archives_for_log(LOG_A) == [camera_archive, lidar_archive]
    assert index.archives_for_log(LOG_B) == [camera_archive]
    assert index.archives_for_log("some-other-log") == []
    # The camera and lidar files of a log are found together.
    assert set(load_log_members(LOG_A, index.archives_for_log(LOG_A))) == {
        f"{LOG_A}/CAM_F0/aa.jpg",
        f"{LOG_A}/CAM_L0/bb.jpg",
        f"{LOG_A}/MergedPointCloud/9e56754b1c205e17.pcd",
    }


def test_archive_without_index_is_reported(camera_archive: Path, lidar_archive: Path, tmp_path: Path, caplog):
    build_tar_index(camera_archive)

    index = NuplanTarIndex(tmp_path)

    assert index.archives_for_log(LOG_A) == [camera_archive]
    assert "1 sensor archives" in caplog.text and lidar_archive.name in caplog.text


def test_up_to_date_index_is_kept(camera_archive: Path):
    index_path = build_tar_index(camera_archive)
    modification_time = index_path.stat().st_mtime_ns

    build_tar_index(camera_archive)
    assert index_path.stat().st_mtime_ns == modification_time

    build_tar_index(camera_archive, force=True)
    assert index_path.stat().st_mtime_ns > modification_time


def test_command_line(camera_archive: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["nuplan_tar_index", str(camera_archive), str(tmp_path / "missing.zip")])

    assert nuplan_tar_index.main() == 1  # one archive could not be indexed
    assert camera_archive.with_name(camera_archive.name + ".index.parquet").exists()
