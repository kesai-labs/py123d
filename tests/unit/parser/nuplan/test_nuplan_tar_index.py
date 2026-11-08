import io
import tarfile
from pathlib import Path

import pytest

from py123d.parser.nuplan.nuplan_tar_index import (
    NuplanTarIndex,
    TarMemberReader,
    build_tar_index,
    load_log_members,
)

LOG_A = "2021.05.25.14.24.08_veh-25_00934_01067"
LOG_B = "2021.05.25.14.16.10_veh-35_01100_01664"


def _make_archive(path: Path, wrapper: str, files: dict) -> None:
    """Write an uncompressed tar mirroring a nuPlan sensor shard's layout."""
    with tarfile.open(path, "w") as tf:
        for name, payload in files.items():
            info = tarfile.TarInfo(f"{wrapper}/{name}")
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))


@pytest.fixture
def camera_archive(tmp_path: Path) -> tuple:
    files = {
        f"{LOG_A}/CAM_F0/aa.jpg": b"jpeg-bytes-aa",
        f"{LOG_A}/CAM_L0/bb.jpg": b"jpeg-bytes-bb-longer",
        f"{LOG_B}/CAM_F0/cc.jpg": b"jpeg-bytes-cc",
    }
    tar_path = tmp_path / "nuplan-v1.1_test_camera_0.zip"
    _make_archive(tar_path, "nuplan-v1.1_test_camera_0", files)
    return tar_path, files


def test_build_and_read_roundtrip(camera_archive) -> None:
    tar_path, files = camera_archive
    index_path = build_tar_index(tar_path)
    assert index_path.exists()
    assert tar_path.with_name(tar_path.name + ".logs.txt").exists()

    members = load_log_members(LOG_A, [tar_path])
    assert set(members) == {f"{LOG_A}/CAM_F0/aa.jpg", f"{LOG_A}/CAM_L0/bb.jpg"}

    reader = TarMemberReader()
    for relative_path, member in members.items():
        assert reader.read(*member) == files[relative_path]
    reader.close()


def test_catalog_maps_logs_to_archives(camera_archive, tmp_path: Path) -> None:
    tar_path, _ = camera_archive
    lidar_path = tmp_path / "nuplan-v1.1_test_lidar_0.zip"
    _make_archive(
        lidar_path,
        "nuplan-v1.1_test_lidar_0",
        {f"{LOG_A}/MergedPointCloud/dd.pcd": b"pcd-bytes"},
    )
    build_tar_index(tar_path)
    build_tar_index(lidar_path)

    index = NuplanTarIndex(tmp_path)
    assert index.log_names == {LOG_A, LOG_B}
    assert index.archives_for_log(LOG_A) == [tar_path, lidar_path]
    assert index.archives_for_log(LOG_B) == [tar_path]
    assert index.archives_for_log("missing-log") == []

    # Camera and lidar members merge into one lookup for the log.
    members = load_log_members(LOG_A, index.archives_for_log(LOG_A))
    assert f"{LOG_A}/MergedPointCloud/dd.pcd" in members
    assert f"{LOG_A}/CAM_F0/aa.jpg" in members


def test_index_is_idempotent_and_forceable(camera_archive) -> None:
    tar_path, _ = camera_archive
    index_path = build_tar_index(tar_path)
    mtime = index_path.stat().st_mtime_ns
    assert build_tar_index(tar_path) == index_path
    assert index_path.stat().st_mtime_ns == mtime  # skipped, not rebuilt
    build_tar_index(tar_path, force=True)


def test_catalog_over_several_folders(camera_archive, tmp_path: Path) -> None:
    """Camera indexes in one folder (symbolic links), lidar archives in another; an archive indexed in both folders
    appears once."""
    tar_path, files = camera_archive
    build_tar_index(tar_path)
    lidar_dir = tmp_path / "lidar"
    lidar_dir.mkdir()
    lidar_path = lidar_dir / "nuplan-v1.1_test_lidar_0.zip"
    _make_archive(lidar_path, "nuplan-v1.1_test_lidar_0", {f"{LOG_B}/MergedPointCloud/ee.pcd": b"pcd"})
    build_tar_index(lidar_path)
    (lidar_dir / "nuplan-v1.1_test_lidar_1.zip.part").write_bytes(b"")  # a download in progress: not reported
    link_dir = tmp_path / "links"
    link_dir.mkdir()
    (link_dir / tar_path.name).symlink_to(tar_path)
    build_tar_index(link_dir / tar_path.name)

    index = NuplanTarIndex([tmp_path, lidar_dir, link_dir])
    assert index.archives_for_log(LOG_B) == [tar_path, lidar_path]
    members = load_log_members(LOG_B, index.archives_for_log(LOG_B))
    reader = TarMemberReader()
    assert reader.read(*members[f"{LOG_B}/CAM_F0/cc.jpg"]) == files[f"{LOG_B}/CAM_F0/cc.jpg"]
    assert reader.read(*members[f"{LOG_B}/MergedPointCloud/ee.pcd"]) == b"pcd"
    reader.close()


def test_prefetch_then_read(camera_archive) -> None:
    tar_path, files = camera_archive
    build_tar_index(tar_path)
    members = load_log_members(LOG_A, [tar_path])
    reader = TarMemberReader()
    for relative_path, member in members.items():
        reader.prefetch(*member)
        assert reader.read(*member) == files[relative_path]
    reader.close()
