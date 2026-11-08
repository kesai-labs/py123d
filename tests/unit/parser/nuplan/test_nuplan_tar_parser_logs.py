"""Log selection of the NuplanTarParser: allow-list and logs with or without sensor data."""

import io
import tarfile
from pathlib import Path

import pytest

pytest.importorskip("nuplan")  # the nuPlan devkit (extra "nuplan")

from py123d.parser.nuplan.nuplan_parser import _create_splits_logs  # noqa: E402
from py123d.parser.nuplan.nuplan_tar_index import build_tar_index  # noqa: E402
from py123d.parser.nuplan.nuplan_tar_parser import NuplanTarParser  # noqa: E402


@pytest.fixture
def roots(tmp_path: Path):
    """Three train logs (empty database files); the first has a camera file in an indexed archive."""
    logs = _create_splits_logs()["train"][:3]
    split_dir = tmp_path / "data" / "nuplan-v1.1" / "splits" / "trainval"
    split_dir.mkdir(parents=True)
    for log in logs:
        (split_dir / f"{log}.db").write_bytes(b"")
    archives = tmp_path / "archives"
    archives.mkdir()
    tar_path = archives / "nuplan-v1.1_train_camera_0.zip"
    with tarfile.open(tar_path, "w") as tf:
        info = tarfile.TarInfo(f"nuplan-v1.1_train_camera_0/{logs[0]}/CAM_F0/a.jpg")
        info.size = 3
        tf.addfile(info, io.BytesIO(b"jpg"))
    build_tar_index(tar_path)
    return tmp_path / "data", archives, logs


def _names(parser):
    return sorted(log_parser._source_log_path.stem for log_parser in parser.get_log_parsers())


@pytest.mark.parametrize("sensor_logs, expected", [(None, [0, 1, 2]), ("with", [0]), ("without", [1, 2])])
def test_sensor_logs(roots, sensor_logs, expected):
    data_root, archives, logs = roots
    parser = NuplanTarParser(
        splits=["nuplan_train"],
        nuplan_data_root=data_root,
        nuplan_maps_root=None,
        nuplan_archive_roots=str(archives),
        sensor_logs=sensor_logs,
    )
    assert _names(parser) == sorted(logs[i] for i in expected)


def test_allow_list_and_filter(roots):
    data_root, archives, logs = roots
    parser = NuplanTarParser(
        splits=["nuplan_train"],
        nuplan_data_root=data_root,
        nuplan_maps_root=None,
        nuplan_archive_roots=[archives],
        log_names=logs[:2],
        sensor_logs="without",
    )
    assert _names(parser) == [logs[1]]


def test_invalid_filter(roots):
    data_root, archives, _ = roots
    with pytest.raises(AssertionError):
        NuplanTarParser(["nuplan_train"], data_root, None, archives, sensor_logs="some")
