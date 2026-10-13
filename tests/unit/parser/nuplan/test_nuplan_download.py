"""The nuPlan sensor archives land in the layout the parser reads, whether they are zip or tar files."""

import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from py123d.parser.nuplan.nuplan_download import NUPLAN_ARCHIVE_CATALOG, _extract_nuplan_archive

LOG_NAME = "2021.05.25.14.24.08_veh-25_00934_01067"


def _write_zip(path: Path, members: dict) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)


def _write_tar(path: Path, members: dict) -> None:
    with tarfile.open(path, "w") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


def _extract(tmp_path: Path, filename: str) -> Path:
    """Extracts the archive ``tmp_path / filename`` the way the downloader does, returns the output directory."""
    spec = next(spec for spec in NUPLAN_ARCHIVE_CATALOG if spec.filename == filename)
    output_dir = tmp_path / "output"
    _extract_nuplan_archive(
        archive_path=tmp_path / filename,
        extract_dir=output_dir,
        target_subdir=spec.target_subdir,
        skip_levels=spec.skip_levels,
        staging_root=tmp_path / "_unpack",
    )
    return output_dir


@pytest.mark.parametrize("filename", ["nuplan-v1.1_test_camera_0.zip", "nuplan-v1.1_val_lidar_3.zip"])
def test_full_split_sensor_shard_is_a_tar_archive(tmp_path: Path, filename: str):
    """The train, val and test sensor shards are tar files with a ``.zip`` name and one wrapper folder."""
    wrapper = filename[: -len(".zip")]
    _write_tar(tmp_path / filename, {f"{wrapper}/{LOG_NAME}/CAM_L0/267b107b56f6521a.jpg": b"jpg"})

    output_dir = _extract(tmp_path, filename)

    sensor_root = output_dir / "nuplan-v1.1" / "sensor_blobs"
    assert (sensor_root / LOG_NAME / "CAM_L0" / "267b107b56f6521a.jpg").read_bytes() == b"jpg"
    assert not (sensor_root / wrapper).exists()


def test_mini_sensor_shard_is_a_zip_archive(tmp_path: Path):
    """The mini sensor shards are zip files, with the same wrapper folder."""
    filename = "nuplan-v1.1_mini_camera_0.zip"
    wrapper = filename[: -len(".zip")]
    _write_zip(tmp_path / filename, {f"{wrapper}/{LOG_NAME}/CAM_R2/37773bdda1775cd4.jpg": b"jpg"})

    output_dir = _extract(tmp_path, filename)

    sensor_root = output_dir / "nuplan-v1.1" / "sensor_blobs"
    assert (sensor_root / LOG_NAME / "CAM_R2" / "37773bdda1775cd4.jpg").read_bytes() == b"jpg"
    assert not (sensor_root / wrapper).exists()


def test_log_archive_keeps_its_layout(tmp_path: Path):
    """The log archives are zip files with three wrapper folders."""
    filename = "nuplan-v1.1_test.zip"
    _write_zip(tmp_path / filename, {f"data/cache/test/{LOG_NAME}.db": b"db"})

    output_dir = _extract(tmp_path, filename)

    assert (output_dir / "nuplan-v1.1" / "splits" / "test" / f"{LOG_NAME}.db").read_bytes() == b"db"


def test_shards_of_one_log_are_merged(tmp_path: Path):
    """A log's cameras and lidar come in different shards, and one sensor can span two shards."""
    members_per_shard = {
        "nuplan-v1.1_test_camera_0.zip": f"{LOG_NAME}/CAM_L0/a.jpg",
        "nuplan-v1.1_test_camera_1.zip": f"{LOG_NAME}/CAM_L0/b.jpg",
        "nuplan-v1.1_test_lidar_0.zip": f"{LOG_NAME}/MergedPointCloud/c.pcd",
    }
    for filename, member in members_per_shard.items():
        _write_tar(tmp_path / filename, {f"{filename[: -len('.zip')]}/{member}": b"data"})
        output_dir = _extract(tmp_path, filename)

    log_dir = output_dir / "nuplan-v1.1" / "sensor_blobs" / LOG_NAME
    assert sorted(path.relative_to(log_dir).as_posix() for path in log_dir.rglob("*") if path.is_file()) == [
        "CAM_L0/a.jpg",
        "CAM_L0/b.jpg",
        "MergedPointCloud/c.pcd",
    ]


def test_unsafe_tar_member_is_refused(tmp_path: Path):
    """A tar member that points outside the extraction directory is not extracted."""
    filename = "nuplan-v1.1_test_camera_0.zip"
    _write_tar(tmp_path / filename, {"../escaped.jpg": b"jpg"})

    with pytest.raises(ValueError, match="unsafe member"):
        _extract(tmp_path, filename)

    assert not list(tmp_path.rglob("escaped.jpg"))


def test_tar_link_member_is_refused(tmp_path: Path):
    """Links are not part of the nuPlan archives and are not extracted."""
    filename = "nuplan-v1.1_test_camera_0.zip"
    with tarfile.open(tmp_path / filename, "w") as archive:
        link = tarfile.TarInfo("nuplan-v1.1_test_camera_0/link.jpg")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/hostname"
        archive.addfile(link)

    with pytest.raises(ValueError, match="unsafe member"):
        _extract(tmp_path, filename)
