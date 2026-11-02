"""Transcoding of converted 123D logs into other camera and lidar codecs (for example JPEG and IPC into kedec).

A log is transcoded without its raw dataset: every file except the camera and lidar files is copied byte for byte;
the cameras are written again from their stored rows (time stamps, poses as stored, exposure, arrival times and the
encoded images), optionally only a selection of their images; the lidars are decoded and encoded again row by row.
``sync.arrow`` is copied with new camera columns: each sync row references the kept image nearest to its reference
time, strictly within half the period of the sync table (the sync direction ``"nearest"``). The UUIDs, the time
stamps and all other columns of the sync table stay identical.

The new log is written into ``<target logs root>/.incomplete/<split>/<log>`` and moved into place when it is
complete, ``sync.arrow`` last, as :class:`~py123d.api.scene.arrow.arrow_log_writer.ArrowLogWriter` does.
"""

from __future__ import annotations

import abc
import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from py123d.api.scene.arrow.arrow_log_writer import nearest_sync_rows
from py123d.api.scene.arrow.modalities.arrow_camera import ArrowCameraWriter
from py123d.api.scene.arrow.modalities.arrow_lidar import ArrowLidarWriter
from py123d.api.scene.arrow.utils.log_writer_config import LogWriterConfig
from py123d.api.utils.arrow_metadata_utils import get_metadata_from_arrow_schema, resolve_metadata_class
from py123d.common.frame_selection import FrameSelectionConfig, select_nearest_frames, uniform_reference_grid
from py123d.common.io.camera.av1_camera_io import av1_file_name, read_av1_frame
from py123d.common.io.lidar.point_cloud_codec_registry import decode_point_cloud
from py123d.datatypes.metadata.log_metadata import LogMetadata
from py123d.datatypes.sensors.base_camera import BaseCameraMetadata
from py123d.datatypes.sensors.lidar import Lidar, LidarMergedMetadata, LidarMetadata
from py123d.datatypes.time.timestamp import Timestamp
from py123d.geometry.pose import PoseSE3
from py123d.parser.base_dataset_parser import ParsedCamera

logger = logging.getLogger(__name__)

_CAMERA_PREFIX = "camera."
_LIDAR_PREFIX = "lidar."
_SYNC_FILE = "sync.arrow"
_CAMERA_COLUMNS = ("timestamp_us", "data", "camera_to_global_se3", "exposure_factor", "arrival_us")
_LIDAR_COLUMNS = ("timestamp_us", "end_timestamp_us", "data", "arrival_us")


# ------------------------------------------------------------------------------------------------------------------
# Camera selection
# ------------------------------------------------------------------------------------------------------------------


class CameraSelection(abc.ABC):
    """Chooses which stored images of each camera a transcoded log keeps."""

    @abc.abstractmethod
    def select(self, log_dir: Path, camera_timestamps: Dict[str, npt.NDArray[np.int64]]) -> Dict[str, npt.NDArray]:
        """The kept row indices of each camera.

        :param log_dir: The source log directory (for rules that read other modalities, e.g. the lidar).
        :param camera_timestamps: Camera modality key -> the time stamps of its stored rows.
        :return: Camera modality key -> sorted, unique row indices to keep.
        """


class KeepAllImages(CameraSelection):
    """Keeps every image."""

    def select(self, log_dir: Path, camera_timestamps: Dict[str, npt.NDArray[np.int64]]) -> Dict[str, npt.NDArray]:
        return {key: np.arange(len(ts)) for key, ts in camera_timestamps.items()}


@dataclass
class GridSelection(CameraSelection):
    """Keeps, per camera, the image nearest to each time of a uniform grid, with hysteresis.

    :param config: The period, phase (``grid_phase_us``), hysteresis and limit of the selection.
    """

    config: FrameSelectionConfig = field(default_factory=lambda: FrameSelectionConfig(mode="grid"))

    def select(self, log_dir: Path, camera_timestamps: Dict[str, npt.NDArray[np.int64]]) -> Dict[str, npt.NDArray]:
        kept: Dict[str, npt.NDArray] = {}
        for key, ts in camera_timestamps.items():
            if len(ts) == 0:
                kept[key] = np.zeros(0, dtype=np.int64)
                continue
            grid = uniform_reference_grid(int(ts[0]), int(ts[-1]) + 1, self.config.period_us, self.config.grid_phase_us)
            rows = select_nearest_frames(ts, grid, self.config.hysteresis_us, limit_us=self.config.limit_us)
            kept[key] = np.unique(rows[rows >= 0])
        return kept


# ------------------------------------------------------------------------------------------------------------------
# Configuration and log discovery
# ------------------------------------------------------------------------------------------------------------------


@dataclass
class TranscodeConfig:
    """Settings of a transcoding run.

    :param log_writer_config: The camera and lidar codecs of the new log (``camera_store_option``,
        ``video_codec_config``, ``lidar_codec``, ``lidar_codec_config``, ``encode_threads``); its other fields are
        not used.
    :param camera_threads: Cameras of a log transcoded at the same time (each with its own video encoder).
    :param force: Transcode logs whose target already exists.
    :param ipc_compression: Compression of the new camera and lidar Arrow files (None, "lz4" or "zstd").
    """

    log_writer_config: LogWriterConfig = field(
        default_factory=lambda: LogWriterConfig(
            camera_store_option="av1", lidar_store_option="binary", lidar_codec="kedec"
        )
    )
    camera_threads: int = 4
    force: bool = False
    ipc_compression: Optional[str] = None


def find_logs(
    logs_root: Path, splits: Optional[Sequence[str]] = None, log_names: Optional[Sequence[str]] = None
) -> List[Tuple[str, str]]:
    """The (split, log name) of every complete log (with ``sync.arrow``) under a logs root, sorted.

    :param logs_root: The logs root (``<split>/<log>/sync.arrow``); the ``.incomplete`` folder is skipped.
    :param splits: Optional splits to keep.
    :param log_names: Optional log names to keep.
    """
    found: List[Tuple[str, str]] = []
    for sync_path in sorted(Path(logs_root).glob(f"*/*/{_SYNC_FILE}")):
        split, log_name = sync_path.parent.parent.name, sync_path.parent.name
        if split.startswith("."):
            continue
        if splits is not None and split not in splits:
            continue
        if log_names is not None and log_name not in log_names:
            continue
        found.append((split, log_name))
    return found


# ------------------------------------------------------------------------------------------------------------------
# Transcoding
# ------------------------------------------------------------------------------------------------------------------


def _open(path: Path) -> pa.ipc.RecordBatchFileReader:
    return pa.ipc.open_file(pa.memory_map(str(path), "r"))


def _metadata(path: Path):
    return get_metadata_from_arrow_schema(_open(path).schema, resolve_metadata_class(path.stem))


def _iter_rows(path: Path, columns: Sequence[str]) -> Iterator[Tuple[int, Dict[str, object]]]:
    """(row index, {column: value}) of the given columns, batch by batch (missing columns give None)."""
    reader = _open(path)
    key = path.stem
    row = 0
    for b in range(reader.num_record_batches):
        batch = reader.get_batch(b)
        values = {
            name: batch.column(f"{key}.{name}").to_pylist() if f"{key}.{name}" in batch.schema.names else None
            for name in columns
        }
        for i in range(batch.num_rows):
            yield row, {name: (column[i] if column is not None else None) for name, column in values.items()}
            row += 1


def _check_columns(path: Path, known: Sequence[str]) -> None:
    key = path.stem
    unknown = [name for name in _open(path).schema.names if name[len(key) + 1 :] not in known]
    if unknown:
        raise ValueError(f"{path}: columns {unknown} would be lost by transcoding.")


def _timestamps(path: Path) -> npt.NDArray[np.int64]:
    table = _open(path).read_all()
    return table.column(f"{path.stem}.timestamp_us").to_numpy().astype(np.int64)


def _camera_image(source_dir: Path, key: str, data: object) -> Tuple[Optional[bytes], Optional[np.ndarray]]:
    """(encoded bytes, None) for bytes, or (None, RGB image) for an image number of a video."""
    if isinstance(data, bytes):
        return data, None
    if isinstance(data, int):
        av1_path = source_dir / av1_file_name(key)
        if av1_path.exists():
            return None, read_av1_frame(av1_path, data)
        from py123d.common.io.camera.mp4_camera_io import get_mp4_reader_from_path

        return None, get_mp4_reader_from_path(str(source_dir / f"{key}.mp4")).get_frame(data)
    raise ValueError(f"Camera {key}: rows with {type(data).__name__} data (e.g. file paths) cannot be transcoded.")


def transcode_camera(
    source_path: Path, target_dir: Path, kept_rows: npt.NDArray, config: TranscodeConfig
) -> npt.NDArray[np.int64]:
    """Write the kept rows of one camera file into ``target_dir`` with the configured camera codec.

    :return: The time stamps of the written rows.
    """
    _check_columns(source_path, _CAMERA_COLUMNS)
    metadata = _metadata(source_path)
    assert isinstance(metadata, BaseCameraMetadata)
    writer = ArrowCameraWriter(
        log_dir=target_dir,
        metadata=metadata,
        camera_codec=config.log_writer_config.camera_store_option,
        ipc_compression=config.ipc_compression,  # type: ignore[arg-type]
        video_codec_config=config.log_writer_config.video_codec_config,
    )
    kept = set(int(r) for r in kept_rows)
    written: List[int] = []
    try:
        for row, values in _iter_rows(source_path, _CAMERA_COLUMNS):
            if row not in kept:
                continue
            byte_string, frame = _camera_image(source_path.parent, source_path.stem, values["data"])
            pose = values["camera_to_global_se3"]
            arrival = values["arrival_us"]
            writer.write_modality(
                ParsedCamera(
                    metadata=metadata,  # type: ignore[arg-type]
                    timestamp=Timestamp.from_us(int(values["timestamp_us"])),  # type: ignore[arg-type]
                    camera_to_global_se3=None if pose is None else PoseSE3.from_list(pose),  # type: ignore[arg-type]
                    byte_string=byte_string,
                    frame=frame,
                    exposure_factor=values["exposure_factor"],  # type: ignore[arg-type]
                    arrival_timestamp=None if arrival is None else Timestamp.from_us(int(arrival)),  # type: ignore[arg-type]
                )
            )
            written.append(int(values["timestamp_us"]))  # type: ignore[arg-type]
    finally:
        writer.close()
    return np.asarray(written, dtype=np.int64)


def transcode_lidar(source_path: Path, target_dir: Path, log_metadata: LogMetadata, config: TranscodeConfig) -> int:
    """Decode every row of one lidar file and write it again with the configured lidar codec.

    :return: The number of rows.
    """
    _check_columns(source_path, _LIDAR_COLUMNS)
    metadata = _metadata(source_path)
    assert isinstance(metadata, (LidarMetadata, LidarMergedMetadata))
    writer_config = config.log_writer_config
    writer = ArrowLidarWriter(
        log_dir=target_dir,
        metadata=metadata,
        log_metadata=log_metadata,
        lidar_store_option="binary",
        lidar_codec=writer_config.lidar_codec,
        ipc_compression=config.ipc_compression,  # type: ignore[arg-type]
        lidar_codec_config=writer_config.lidar_codec_config,
        encode_threads=writer_config.encode_threads,
    )
    rows = 0
    try:
        for _, values in _iter_rows(source_path, _LIDAR_COLUMNS):
            data = values["data"]
            if not isinstance(data, bytes):
                raise ValueError(f"{source_path}: lidar rows must store binary blobs to be transcoded.")
            xyz, features = decode_point_cloud(data)
            arrival = values["arrival_us"]
            writer.write_modality(
                Lidar(
                    timestamp=Timestamp.from_us(int(values["timestamp_us"])),  # type: ignore[arg-type]
                    timestamp_end=Timestamp.from_us(int(values["end_timestamp_us"])),  # type: ignore[arg-type]
                    metadata=metadata,
                    point_cloud_3d=xyz,
                    point_cloud_features=features,
                    arrival_timestamp=None if arrival is None else Timestamp.from_us(int(arrival)),  # type: ignore[arg-type]
                )
            )
            rows += 1
    finally:
        writer.close()
    return rows


def rewrite_sync_cameras(
    source_sync: Path, target_sync: Path, camera_timestamps: Dict[str, npt.NDArray[np.int64]]
) -> None:
    """Copy a sync table with new camera columns: the kept image nearest to each row's time, strictly within half
    the period of the table; all other columns and the schema metadata unchanged."""
    reader = _open(source_sync)
    table = reader.read_all()
    sync_ts = table.column("sync.timestamp_us").to_numpy().astype(np.int64)
    for key, timestamps in camera_timestamps.items():
        if key not in table.column_names:
            continue
        rows = nearest_sync_rows(timestamps, sync_ts)
        column = pa.array([int(r) if r >= 0 else None for r in rows], type=table.schema.field(key).type)
        table = table.set_column(table.schema.get_field_index(key), table.schema.field(key), column)
    table = table.replace_schema_metadata(reader.schema.metadata)
    with pa.OSFile(str(target_sync), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table, max_chunksize=max(1, table.num_rows))


def transcode_log(
    source_logs_root: Path,
    target_logs_root: Path,
    split: str,
    log_name: str,
    config: TranscodeConfig,
    selection: Optional[CameraSelection] = None,
) -> bool:
    """Transcode one log.

    :param source_logs_root: The logs root of the source log.
    :param target_logs_root: The logs root of the new log (not the source root).
    :param split: The split of the log.
    :param log_name: The name of the log.
    :param config: The settings.
    :param selection: The camera selection; keeps every image when None.
    :return: True if the log was written, False if it existed already and ``force`` is not set.
    """
    source_dir = Path(source_logs_root) / split / log_name
    target_dir = Path(target_logs_root) / split / log_name
    if source_dir.resolve() == target_dir.resolve():
        raise ValueError(f"Source and target of {split}/{log_name} are the same directory.")
    if (target_dir / _SYNC_FILE).exists() and not config.force:
        return False
    work_dir = Path(target_logs_root) / ".incomplete" / split / log_name
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True)

    files = sorted(p for p in source_dir.iterdir() if p.is_file())
    camera_files = [p for p in files if p.name.startswith(_CAMERA_PREFIX) and p.suffix == ".arrow"]
    camera_videos = {p for p in files if p.name.startswith(_CAMERA_PREFIX) and p.name.endswith(".mp4")}
    lidar_files = [p for p in files if p.name.startswith(_LIDAR_PREFIX) and p.suffix == ".arrow"]
    for path in files:
        if path in camera_videos or path in camera_files or path in lidar_files or path.name == _SYNC_FILE:
            continue
        shutil.copy2(path, work_dir / path.name)

    camera_timestamps = {p.stem: _timestamps(p) for p in camera_files}
    kept = (selection or KeepAllImages()).select(source_dir, camera_timestamps)
    written: Dict[str, npt.NDArray[np.int64]] = {}
    with ThreadPoolExecutor(max_workers=max(1, config.camera_threads)) as pool:
        futures = {p.stem: pool.submit(transcode_camera, p, work_dir, kept[p.stem], config) for p in camera_files}
        for key, future in futures.items():
            written[key] = future.result()

    log_metadata = get_metadata_from_arrow_schema(_open(source_dir / _SYNC_FILE).schema, LogMetadata)
    for path in lidar_files:
        transcode_lidar(path, work_dir, log_metadata, config)
    rewrite_sync_cameras(source_dir / _SYNC_FILE, work_dir / _SYNC_FILE, written)

    target_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(work_dir.iterdir(), key=lambda p: p.name == _SYNC_FILE):
        path.replace(target_dir / path.name)
    os.removedirs(work_dir)
    return True
