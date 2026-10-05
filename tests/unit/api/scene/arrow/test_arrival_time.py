"""Tests for the optional per-row arrival time (``<modality_key>.arrival_us``) of the sensor modalities.

The arrival time is stored only when the modality's metadata sets ``has_arrival_time``. Without the flag a
writer must produce the schema it produced before the column existed, and the measurement time
``timestamp_us`` must remain the only time used for synchronization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Type

import msgpack
import numpy as np
import pyarrow as pa
import pytest

from py123d.api.scene.arrow.arrow_log_writer import ArrowLogWriter, SyncConfig
from py123d.api.scene.arrow.arrow_scene_api import ArrowSceneAPI
from py123d.api.scene.arrow.modalities.arrow_barometer import ArrowBarometerReader, ArrowBarometerWriter
from py123d.api.scene.arrow.modalities.arrow_base import ArrowBaseModalityReader, ArrowBaseModalityWriter
from py123d.api.scene.arrow.modalities.arrow_camera import ArrowCameraReader, ArrowCameraWriter
from py123d.api.scene.arrow.modalities.arrow_gnss import ArrowGnssReader, ArrowGnssWriter
from py123d.api.scene.arrow.modalities.arrow_imu import ArrowImuReader, ArrowImuWriter
from py123d.api.scene.arrow.modalities.arrow_lidar import ArrowLidarReader, ArrowLidarWriter
from py123d.api.scene.arrow.modalities.arrow_magnetometer import ArrowMagnetometerReader, ArrowMagnetometerWriter
from py123d.api.scene.arrow.modalities.arrow_radar import ArrowRadarReader, ArrowRadarWriter
from py123d.api.scene.arrow.modalities.utils import ARRIVAL_TIME_FIELD
from py123d.api.scene.arrow.utils.log_writer_config import LogWriterConfig
from py123d.api.utils.arrow_metadata_utils import get_metadata_from_arrow_schema, resolve_metadata_class
from py123d.datatypes import (
    Barometer,
    BarometerMetadata,
    Gnss,
    GnssMetadata,
    Imu,
    ImuMetadata,
    Magnetometer,
    MagnetometerMetadata,
    SceneMetadata,
    Timestamp,
)
from py123d.datatypes.modalities.base_modality import BaseModality, BaseModalityMetadata, ModalityType
from py123d.datatypes.sensors.base_camera import Camera, CameraID, camera_metadata_from_dict
from py123d.datatypes.sensors.fisheye_mei_camera import FisheyeMEICameraMetadata
from py123d.datatypes.sensors.ftheta_camera import FThetaCameraMetadata
from py123d.datatypes.sensors.lidar import (
    Lidar,
    LidarFeature,
    LidarID,
    LidarMergedMetadata,
    LidarMetadata,
    get_individual_lidar,
    get_merged_lidar,
)
from py123d.datatypes.sensors.pinhole_camera import PinholeCameraMetadata, PinholeIntrinsics
from py123d.datatypes.sensors.radar import (
    Radar,
    RadarFeature,
    RadarID,
    RadarMergedMetadata,
    RadarMetadata,
    get_individual_radar,
    get_merged_radar,
)
from py123d.geometry.pose import PoseSE3
from py123d.geometry.vector import Vector3D
from py123d.parser.base_dataset_parser import ParsedCamera, ParsedLidar, ParsedRadar

from ..conftest import make_ego_metadata, make_log_metadata

# ----------------------------------------------------------------------------------------------------------------------
# Modality factories
# ----------------------------------------------------------------------------------------------------------------------

_IMAGE_HEIGHT, _IMAGE_WIDTH = 8, 16


def _pinhole_metadata(has_arrival_time: bool) -> PinholeCameraMetadata:
    return PinholeCameraMetadata(
        camera_name="front_camera",
        camera_id=CameraID.PCAM_F0,
        intrinsics=PinholeIntrinsics(fx=10.0, fy=10.0, cx=8.0, cy=4.0),
        distortion=None,
        width=_IMAGE_WIDTH,
        height=_IMAGE_HEIGHT,
        camera_to_imu_se3=PoseSE3.identity(),
        has_arrival_time=has_arrival_time,
    )


def _fisheye_metadata(has_arrival_time: bool) -> FisheyeMEICameraMetadata:
    return FisheyeMEICameraMetadata(
        camera_name="left_fisheye",
        camera_id=CameraID.FMCAM_L,
        mirror_parameter=None,
        distortion=None,
        projection=None,
        width=_IMAGE_WIDTH,
        height=_IMAGE_HEIGHT,
        camera_to_imu_se3=PoseSE3.identity(),
        has_arrival_time=has_arrival_time,
    )


def _ftheta_metadata(has_arrival_time: bool) -> FThetaCameraMetadata:
    return FThetaCameraMetadata(
        camera_name="front_wide",
        camera_id=CameraID.FTCAM_F0,
        intrinsics=None,
        width=_IMAGE_WIDTH,
        height=_IMAGE_HEIGHT,
        camera_to_imu_se3=PoseSE3.identity(),
        has_arrival_time=has_arrival_time,
    )


def _lidar_metadata(has_arrival_time: bool, lidar_id: LidarID = LidarID.LIDAR_TOP) -> LidarMetadata:
    return LidarMetadata(lidar_name=lidar_id.name, lidar_id=lidar_id, has_arrival_time=has_arrival_time)


def _radar_metadata(has_arrival_time: bool, radar_id: RadarID = RadarID.RADAR_FRONT) -> RadarMetadata:
    return RadarMetadata(radar_name=radar_id.name, radar_id=radar_id, has_arrival_time=has_arrival_time)


def _imu(metadata: BaseModalityMetadata, time_us: int, arrival: Optional[Timestamp]) -> Imu:
    assert isinstance(metadata, ImuMetadata)
    return Imu(
        timestamp=Timestamp.from_us(time_us),
        metadata=metadata,
        angular_velocity=Vector3D(0.1, 0.2, 0.3),
        linear_acceleration=Vector3D(0.0, 0.0, -9.81),
        arrival_timestamp=arrival,
    )


def _gnss(metadata: BaseModalityMetadata, time_us: int, arrival: Optional[Timestamp]) -> Gnss:
    assert isinstance(metadata, GnssMetadata)
    return Gnss(
        timestamp=Timestamp.from_us(time_us),
        metadata=metadata,
        latitude=48.5,
        longitude=9.05,
        altitude=350.0,
        status=0,
        arrival_timestamp=arrival,
    )


def _barometer(metadata: BaseModalityMetadata, time_us: int, arrival: Optional[Timestamp]) -> Barometer:
    assert isinstance(metadata, BarometerMetadata)
    return Barometer(
        timestamp=Timestamp.from_us(time_us), metadata=metadata, pressure=96_000.0, arrival_timestamp=arrival
    )


def _magnetometer(metadata: BaseModalityMetadata, time_us: int, arrival: Optional[Timestamp]) -> Magnetometer:
    assert isinstance(metadata, MagnetometerMetadata)
    return Magnetometer(
        timestamp=Timestamp.from_us(time_us),
        metadata=metadata,
        magnetic_field=Vector3D(2e-5, 0.0, -4e-5),
        arrival_timestamp=arrival,
    )


def _camera(metadata: BaseModalityMetadata, time_us: int, arrival: Optional[Timestamp]) -> Camera:
    assert isinstance(metadata, PinholeCameraMetadata)
    image = np.arange(_IMAGE_HEIGHT * _IMAGE_WIDTH * 3, dtype=np.uint8).reshape(_IMAGE_HEIGHT, _IMAGE_WIDTH, 3)
    return Camera(
        metadata=metadata,
        image=image,
        camera_to_global_se3=PoseSE3(x=1.0, y=2.0, z=3.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0),
        timestamp=Timestamp.from_us(time_us),
        arrival_timestamp=arrival,
    )


def _point_cloud(num_points: int = 16) -> np.ndarray:
    return np.arange(num_points * 3, dtype=np.float32).reshape(num_points, 3)


def _lidar(metadata: BaseModalityMetadata, time_us: int, arrival: Optional[Timestamp]) -> Lidar:
    assert isinstance(metadata, (LidarMetadata, LidarMergedMetadata))
    return Lidar(
        timestamp=Timestamp.from_us(time_us),
        timestamp_end=Timestamp.from_us(time_us + 50_000),
        metadata=metadata,
        point_cloud_3d=_point_cloud(),
        point_cloud_features={LidarFeature.INTENSITY.serialize(): np.ones(16, dtype=np.float32)},
        arrival_timestamp=arrival,
    )


def _radar(metadata: BaseModalityMetadata, time_us: int, arrival: Optional[Timestamp]) -> Radar:
    assert isinstance(metadata, (RadarMetadata, RadarMergedMetadata))
    return Radar(
        timestamp=Timestamp.from_us(time_us),
        metadata=metadata,
        point_cloud_3d=_point_cloud(),
        point_cloud_features={RadarFeature.RCS.serialize(): np.ones(16, dtype=np.float32)},
        arrival_timestamp=arrival,
    )


@dataclass(frozen=True)
class _Case:
    """One sensor modality: how to build its metadata, rows and writer, and the columns written before."""

    name: str
    make_metadata: Callable[[bool], BaseModalityMetadata]
    make_modality: Callable[[BaseModalityMetadata, int, Optional[Timestamp]], BaseModality]
    make_writer: Callable[[Path, BaseModalityMetadata], ArrowBaseModalityWriter]
    reader: Type[ArrowBaseModalityReader]
    previous_fields: Tuple[str, ...]
    """The fields the writer produced before the arrival-time column existed, in order."""


_CASES: List[_Case] = [
    _Case(
        name="imu",
        make_metadata=lambda flag: ImuMetadata(imu_name="imu", has_arrival_time=flag),
        make_modality=_imu,
        make_writer=lambda log_dir, metadata: ArrowImuWriter(log_dir, metadata),
        reader=ArrowImuReader,
        previous_fields=("timestamp_us", "angular_velocity", "linear_acceleration"),
    ),
    _Case(
        name="gnss",
        make_metadata=lambda flag: GnssMetadata(gnss_name="gnss", has_arrival_time=flag),
        make_modality=_gnss,
        make_writer=lambda log_dir, metadata: ArrowGnssWriter(log_dir, metadata),
        reader=ArrowGnssReader,
        previous_fields=(
            "timestamp_us",
            "lla",
            "position_covariance",
            "position_covariance_type",
            "status",
            "service",
        ),
    ),
    _Case(
        name="barometer",
        make_metadata=lambda flag: BarometerMetadata(barometer_name="baro", has_arrival_time=flag),
        make_modality=_barometer,
        make_writer=lambda log_dir, metadata: ArrowBarometerWriter(log_dir, metadata),
        reader=ArrowBarometerReader,
        previous_fields=("timestamp_us", "pressure", "msl_altitude", "temperature", "humidity"),
    ),
    _Case(
        name="magnetometer",
        make_metadata=lambda flag: MagnetometerMetadata(magnetometer_name="mag", has_arrival_time=flag),
        make_modality=_magnetometer,
        make_writer=lambda log_dir, metadata: ArrowMagnetometerWriter(log_dir, metadata),
        reader=ArrowMagnetometerReader,
        previous_fields=("timestamp_us", "magnetic_field", "magnetic_field_covariance"),
    ),
    _Case(
        name="camera",
        make_metadata=_pinhole_metadata,
        make_modality=_camera,
        make_writer=lambda log_dir, metadata: ArrowCameraWriter(log_dir, metadata, camera_codec="png_binary"),
        reader=ArrowCameraReader,
        previous_fields=("timestamp_us", "data", "camera_to_global_se3", "exposure_factor"),
    ),
    _Case(
        name="lidar",
        make_metadata=_lidar_metadata,
        make_modality=_lidar,
        make_writer=lambda log_dir, metadata: ArrowLidarWriter(
            log_dir, metadata, make_log_metadata(), lidar_store_option="binary", lidar_codec="ipc"
        ),
        reader=ArrowLidarReader,
        previous_fields=("timestamp_us", "end_timestamp_us", "data"),
    ),
    _Case(
        name="radar",
        make_metadata=_radar_metadata,
        make_modality=_radar,
        make_writer=lambda log_dir, metadata: ArrowRadarWriter(
            log_dir, metadata, make_log_metadata(), radar_store_option="binary", radar_codec="ipc"
        ),
        reader=ArrowRadarReader,
        previous_fields=("timestamp_us", "data"),
    ),
]
_CASE_IDS = [case.name for case in _CASES]

_TIMES_US = (1_000_000, 1_010_000, 1_020_000)
# The second row has no arrival time; the column must hold a null for it.
_ARRIVALS_US: Tuple[Optional[int], ...] = (1_004_000, None, 1_023_500)


def _write(case: _Case, log_dir: Path, has_arrival_time: bool, arrivals_us: Tuple[Optional[int], ...]) -> Path:
    metadata = case.make_metadata(has_arrival_time)
    writer = case.make_writer(log_dir, metadata)
    for time_us, arrival_us in zip(_TIMES_US, arrivals_us):
        arrival = Timestamp.from_us(arrival_us) if arrival_us is not None else None
        writer.write_modality(case.make_modality(metadata, time_us, arrival))
    writer.close()
    return log_dir / f"{metadata.modality_key}.arrow"


def _read_table(path: Path) -> pa.Table:
    with pa.memory_map(str(path), "rb") as source:
        return pa.ipc.open_file(source).read_all()


def _read_metadata(table: pa.Table, modality_key: str) -> BaseModalityMetadata:
    return get_metadata_from_arrow_schema(table.schema, resolve_metadata_class(modality_key))


# ----------------------------------------------------------------------------------------------------------------------
# Writer and reader
# ----------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("case", _CASES, ids=_CASE_IDS)
class TestArrowRoundTrip:
    def test_round_trip_with_flag(self, case: _Case, tmp_path: Path):
        path = _write(case, tmp_path, has_arrival_time=True, arrivals_us=_ARRIVALS_US)
        table = _read_table(path)
        key = path.stem

        arrival_column = f"{key}.{ARRIVAL_TIME_FIELD}"
        assert table.column_names == [f"{key}.{field}" for field in case.previous_fields] + [arrival_column]
        assert table.schema.field(arrival_column).type == pa.int64()
        assert table[arrival_column].to_pylist() == list(_ARRIVALS_US)

        metadata = _read_metadata(table, key)
        assert metadata.has_arrival_time
        for index, (time_us, arrival_us) in enumerate(zip(_TIMES_US, _ARRIVALS_US)):
            modality = case.reader.read_at_index(index, table, metadata, dataset="test-dataset")
            assert modality is not None
            assert modality.timestamp.time_us == time_us
            arrival = modality.arrival_timestamp
            assert (arrival.time_us if arrival is not None else None) == arrival_us

        raw = case.reader.read_column_at_index(0, table, metadata, ARRIVAL_TIME_FIELD, dataset="test-dataset")
        assert raw == _ARRIVALS_US[0]
        deserialized = case.reader.read_column_at_index(
            0, table, metadata, ARRIVAL_TIME_FIELD, dataset="test-dataset", deserialize=True
        )
        assert deserialized == Timestamp.from_us(_ARRIVALS_US[0])
        assert case.reader.read_column_at_index(1, table, metadata, ARRIVAL_TIME_FIELD, dataset="test-dataset") is None

    def test_without_flag_the_schema_is_the_previous_one(self, case: _Case, tmp_path: Path):
        # A file in the schema written before the arrival time existed: no column, no metadata key.
        path = _write(case, tmp_path, has_arrival_time=False, arrivals_us=(None, None, None))
        table = _read_table(path)
        key = path.stem

        assert table.column_names == [f"{key}.{field}" for field in case.previous_fields]
        stored_metadata = msgpack.unpackb(table.schema.metadata[b"metadata"], raw=False, strict_map_key=False)
        assert "has_arrival_time" not in str(stored_metadata)

        metadata = _read_metadata(table, key)
        assert not metadata.has_arrival_time
        for index, time_us in enumerate(_TIMES_US):
            modality = case.reader.read_at_index(index, table, metadata, dataset="test-dataset")
            assert modality is not None
            assert modality.timestamp.time_us == time_us
            assert modality.arrival_timestamp is None
        assert case.reader.read_column_at_index(0, table, metadata, ARRIVAL_TIME_FIELD, dataset="test-dataset") is None

    def test_writer_rejects_an_arrival_time_without_the_flag(self, case: _Case, tmp_path: Path):
        metadata = case.make_metadata(False)
        writer = case.make_writer(tmp_path, metadata)
        modality = case.make_modality(metadata, _TIMES_US[0], Timestamp.from_us(_TIMES_US[0] + 1))
        with pytest.raises(AssertionError, match="has_arrival_time is False"):
            writer.write_modality(modality)
        writer.close()

    def test_arrival_times_out_of_order_are_accepted(self, case: _Case, tmp_path: Path):
        # Only timestamp_us must increase. Arrival times may be reordered by the transport; the writer's
        # order check must not read them.
        _write(case, tmp_path, has_arrival_time=True, arrivals_us=(1_030_000, 1_005_000, 1_021_000))


class TestParsedModalities:
    """The parser helpers carry the arrival time to the writers without loading sensor data."""

    def test_parsed_camera(self, tmp_path: Path):
        metadata = _pinhole_metadata(has_arrival_time=True)
        writer = ArrowCameraWriter(tmp_path, metadata, camera_codec="path")
        for time_us, arrival_us in zip(_TIMES_US, _ARRIVALS_US):
            writer.write_modality(
                ParsedCamera(
                    metadata=metadata,
                    timestamp=Timestamp.from_us(time_us),
                    camera_to_global_se3=None,
                    dataset_root=tmp_path,
                    relative_path=f"{time_us}.jpg",
                    arrival_timestamp=Timestamp.from_us(arrival_us) if arrival_us is not None else None,
                )
            )
        writer.close()
        table = _read_table(tmp_path / f"{metadata.modality_key}.arrow")
        assert table[f"{metadata.modality_key}.{ARRIVAL_TIME_FIELD}"].to_pylist() == list(_ARRIVALS_US)

    def test_parsed_lidar_and_radar(self, tmp_path: Path):
        lidar_metadata = _lidar_metadata(has_arrival_time=True)
        radar_metadata = _radar_metadata(has_arrival_time=True)
        lidar_writer = ArrowLidarWriter(
            tmp_path, lidar_metadata, make_log_metadata(), lidar_store_option="path", lidar_codec=None
        )
        radar_writer = ArrowRadarWriter(
            tmp_path, radar_metadata, make_log_metadata(), radar_store_option="path", radar_codec=None
        )
        for time_us, arrival_us in zip(_TIMES_US, _ARRIVALS_US):
            arrival = Timestamp.from_us(arrival_us) if arrival_us is not None else None
            lidar_writer.write_modality(
                ParsedLidar(
                    metadata=lidar_metadata,
                    start_timestamp=Timestamp.from_us(time_us),
                    end_timestamp=Timestamp.from_us(time_us + 50_000),
                    dataset_root=tmp_path,
                    relative_path=f"{time_us}.pcd",
                    arrival_timestamp=arrival,
                )
            )
            radar_writer.write_modality(
                ParsedRadar(
                    metadata=radar_metadata,
                    timestamp=Timestamp.from_us(time_us),
                    dataset_root=tmp_path,
                    relative_path=f"{time_us}.pcd",
                    arrival_timestamp=arrival,
                )
            )
        lidar_writer.close()
        radar_writer.close()
        for metadata in (lidar_metadata, radar_metadata):
            table = _read_table(tmp_path / f"{metadata.modality_key}.arrow")
            assert table.column_names[-1] == f"{metadata.modality_key}.{ARRIVAL_TIME_FIELD}"
            assert table[f"{metadata.modality_key}.{ARRIVAL_TIME_FIELD}"].to_pylist() == list(_ARRIVALS_US)

    def test_parsed_camera_rejected_without_flag(self, tmp_path: Path):
        metadata = _pinhole_metadata(has_arrival_time=False)
        writer = ArrowCameraWriter(tmp_path, metadata, camera_codec="path")
        parsed = ParsedCamera(
            metadata=metadata,
            timestamp=Timestamp.from_us(_TIMES_US[0]),
            camera_to_global_se3=None,
            dataset_root=tmp_path,
            relative_path="frame.jpg",
            arrival_timestamp=Timestamp.from_us(_TIMES_US[0] + 1),
        )
        with pytest.raises(AssertionError, match="has_arrival_time is False"):
            writer.write_modality(parsed)
        writer.close()


# ----------------------------------------------------------------------------------------------------------------------
# Metadata
# ----------------------------------------------------------------------------------------------------------------------

_METADATA_FACTORIES: Dict[str, Callable[[bool], BaseModalityMetadata]] = {
    "imu": lambda flag: ImuMetadata(imu_name="imu", has_arrival_time=flag),
    "gnss": lambda flag: GnssMetadata(gnss_name="gnss", has_arrival_time=flag),
    "barometer": lambda flag: BarometerMetadata(barometer_name="baro", has_arrival_time=flag),
    "magnetometer": lambda flag: MagnetometerMetadata(magnetometer_name="mag", has_arrival_time=flag),
    "lidar": _lidar_metadata,
    "radar": _radar_metadata,
    "pinhole_camera": _pinhole_metadata,
    "fisheye_mei_camera": _fisheye_metadata,
    "ftheta_camera": _ftheta_metadata,
}


@pytest.mark.parametrize("make_metadata", list(_METADATA_FACTORIES.values()), ids=list(_METADATA_FACTORIES))
class TestMetadataFlag:
    def test_flag_survives_the_dict_round_trip(self, make_metadata: Callable[[bool], BaseModalityMetadata]):
        metadata = make_metadata(True)
        data_dict = metadata.to_dict()
        assert data_dict["has_arrival_time"] is True
        assert type(metadata).from_dict(data_dict).has_arrival_time

    def test_unset_flag_is_not_serialized(self, make_metadata: Callable[[bool], BaseModalityMetadata]):
        # Keeps the schema metadata, and with it the file bytes, of logs without arrival times unchanged.
        metadata = make_metadata(False)
        assert not metadata.has_arrival_time
        assert "has_arrival_time" not in metadata.to_dict()

    def test_metadata_of_older_logs_reads_as_false(self, make_metadata: Callable[[bool], BaseModalityMetadata]):
        data_dict = make_metadata(True).to_dict()
        del data_dict["has_arrival_time"]
        assert not type(make_metadata(True)).from_dict(data_dict).has_arrival_time

    def test_flag_can_be_set_by_a_dict_round_trip(self, make_metadata: Callable[[bool], BaseModalityMetadata]):
        # The way a converter adds the flag to metadata built elsewhere.
        metadata = make_metadata(False)
        flagged = type(metadata).from_dict({**metadata.to_dict(), "has_arrival_time": True})
        assert flagged.has_arrival_time
        assert flagged.modality_key == metadata.modality_key


def test_camera_metadata_factory_reads_the_flag():
    for make_metadata in (_pinhole_metadata, _fisheye_metadata, _ftheta_metadata):
        assert camera_metadata_from_dict(make_metadata(True).to_dict()).has_arrival_time
        assert not camera_metadata_from_dict(make_metadata(False).to_dict()).has_arrival_time


def test_modalities_without_arrival_time_default_to_none():
    metadata = make_ego_metadata()
    assert not metadata.has_arrival_time
    imu = _imu(ImuMetadata(imu_name="imu"), 0, None)
    assert imu.arrival_timestamp is None


# ----------------------------------------------------------------------------------------------------------------------
# Merged lidar and radar
# ----------------------------------------------------------------------------------------------------------------------


class TestMergedSensors:
    def test_merged_metadata_flag_requires_every_member(self):
        top, front = _lidar_metadata(True, LidarID.LIDAR_TOP), _lidar_metadata(True, LidarID.LIDAR_FRONT)
        assert LidarMergedMetadata({LidarID.LIDAR_TOP: top, LidarID.LIDAR_FRONT: front}).has_arrival_time
        unflagged = _lidar_metadata(False, LidarID.LIDAR_FRONT)
        assert not LidarMergedMetadata({LidarID.LIDAR_TOP: top, LidarID.LIDAR_FRONT: unflagged}).has_arrival_time
        assert not LidarMergedMetadata({}).has_arrival_time

        radar_front = _radar_metadata(True, RadarID.RADAR_FRONT)
        radar_back = _radar_metadata(False, RadarID.RADAR_BACK_LEFT)
        assert RadarMergedMetadata({RadarID.RADAR_FRONT: radar_front}).has_arrival_time
        assert not RadarMergedMetadata(
            {RadarID.RADAR_FRONT: radar_front, RadarID.RADAR_BACK_LEFT: radar_back}
        ).has_arrival_time

    def test_merged_lidar_carries_the_latest_arrival(self):
        top, front = _lidar_metadata(True, LidarID.LIDAR_TOP), _lidar_metadata(True, LidarID.LIDAR_FRONT)
        first = _lidar(top, 1_000_000, Timestamp.from_us(1_060_000))
        second = _lidar(front, 1_001_000, Timestamp.from_us(1_070_000))
        merged = get_merged_lidar([first, second])
        assert merged is not None
        assert merged.timestamp.time_us == 1_000_000
        assert merged.arrival_timestamp == Timestamp.from_us(1_070_000)

        without_arrival = _lidar(front, 1_001_000, None)
        partial = get_merged_lidar([first, without_arrival])
        assert partial is not None and partial.arrival_timestamp is None

    def test_merged_lidar_written_and_split(self, tmp_path: Path):
        top, front = _lidar_metadata(True, LidarID.LIDAR_TOP), _lidar_metadata(True, LidarID.LIDAR_FRONT)
        merged_metadata = LidarMergedMetadata({LidarID.LIDAR_TOP: top, LidarID.LIDAR_FRONT: front})
        ids = np.array([int(LidarID.LIDAR_TOP), int(LidarID.LIDAR_FRONT)] * 8, dtype=np.uint8)
        merged = Lidar(
            timestamp=Timestamp.from_us(1_000_000),
            timestamp_end=Timestamp.from_us(1_050_000),
            metadata=merged_metadata,
            point_cloud_3d=_point_cloud(),
            point_cloud_features={LidarFeature.IDS.serialize(): ids},
            arrival_timestamp=Timestamp.from_us(1_070_000),
        )
        writer = ArrowLidarWriter(
            tmp_path, merged_metadata, make_log_metadata(), lidar_store_option="binary", lidar_codec="ipc"
        )
        writer.write_modality(merged)
        writer.close()

        table = _read_table(tmp_path / f"{merged_metadata.modality_key}.arrow")
        metadata = _read_metadata(table, merged_metadata.modality_key)
        assert metadata.has_arrival_time
        restored = ArrowLidarReader.read_at_index(0, table, metadata, dataset="test-dataset")
        assert restored is not None and restored.arrival_timestamp == Timestamp.from_us(1_070_000)
        only_top = ArrowLidarReader.read_at_index(
            0, table, metadata, dataset="test-dataset", lidar_id=LidarID.LIDAR_TOP
        )
        assert only_top is not None and only_top.arrival_timestamp == Timestamp.from_us(1_070_000)

        split = get_individual_lidar(restored, LidarID.LIDAR_FRONT)
        assert split is not None and split.arrival_timestamp == Timestamp.from_us(1_070_000)

    def test_merged_radar_carries_the_latest_arrival(self):
        def _tagged_radar(radar_id: RadarID, time_us: int, arrival_us: int) -> Radar:
            return Radar(
                timestamp=Timestamp.from_us(time_us),
                metadata=_radar_metadata(True, radar_id),
                point_cloud_3d=_point_cloud(),
                point_cloud_features={RadarFeature.IDS.serialize(): np.full(16, int(radar_id), dtype=np.uint8)},
                arrival_timestamp=Timestamp.from_us(arrival_us),
            )

        merged = get_merged_radar(
            [
                _tagged_radar(RadarID.RADAR_FRONT, 1_000_000, 1_030_000),
                _tagged_radar(RadarID.RADAR_BACK_LEFT, 1_002_000, 1_020_000),
            ]
        )
        assert merged is not None
        assert merged.timestamp.time_us == 1_000_000
        assert merged.arrival_timestamp == Timestamp.from_us(1_030_000)
        split = get_individual_radar(merged, RadarID.RADAR_BACK_LEFT)
        assert split is not None and split.arrival_timestamp == Timestamp.from_us(1_030_000)


# ----------------------------------------------------------------------------------------------------------------------
# End to end: ArrowLogWriter and the scene API
# ----------------------------------------------------------------------------------------------------------------------

_NUM_ITERATIONS = 4
_ITERATION_US = 100_000
_LATENCY_US = {"imu": 2_000, "gnss": 40_000, "barometer": 5_000, "magnetometer": 3_000, "camera": 35_000}
_LATENCY_US.update({"lidar": 60_000, "radar": 25_000})

# (modality type, modality id) under which the scene API reads each case.
_SCENE_KEYS = {
    "imu": (ModalityType.IMU, None),
    "gnss": (ModalityType.GNSS, None),
    "barometer": (ModalityType.BAROMETER, None),
    "magnetometer": (ModalityType.MAGNETOMETER, None),
    "camera": (ModalityType.CAMERA, CameraID.PCAM_F0),
    "lidar": (ModalityType.LIDAR, LidarID.LIDAR_TOP),
    "radar": (ModalityType.RADAR, RadarID.RADAR_FRONT),
}


def _arrival_us(case_name: str, iteration: int, time_us: int) -> int:
    # The lidar's arrival times decrease from row to row, so any use of them as the measurement
    # time would fail the writer's order check or change the sync table.
    if case_name == "lidar":
        return 10_000_000 - iteration
    return time_us + _LATENCY_US[case_name]


def _write_log(logs_root: Path, has_arrival_time: bool) -> Path:
    writer = ArrowLogWriter(
        LogWriterConfig(
            camera_store_option="png_binary",
            lidar_store_option="binary",
            lidar_codec="ipc",
            radar_store_option="binary",
            radar_codec="ipc",
        ),
        logs_root=logs_root,
        sensors_root=logs_root,
        sync_config=SyncConfig(reference_column="camera.pcam_f0.timestamp_us"),
    )
    writer.reset(make_log_metadata())
    metadatas = {case.name: case.make_metadata(has_arrival_time) for case in _CASES}
    for iteration in range(_NUM_ITERATIONS):
        for case in _CASES:
            # The camera frame defines the iteration; every other sensor measures a little after it.
            offset_us = 0 if case.name == "camera" else 10 * (_CASES.index(case) + 1)
            time_us = _ITERATION_US * (iteration + 1) + offset_us
            arrival = Timestamp.from_us(_arrival_us(case.name, iteration, time_us)) if has_arrival_time else None
            writer.write_async(case.make_modality(metadatas[case.name], time_us, arrival))
    log_dir = writer._state.log_dir
    writer.close()
    return log_dir


def _scene(log_dir: Path) -> ArrowSceneAPI:
    return ArrowSceneAPI(
        log_dir,
        SceneMetadata(
            dataset="test-dataset",
            split="test-dataset_train",
            initial_uuid="00000000-0000-0000-0000-000000000001",
            initial_idx=0,
            num_future_iterations=_NUM_ITERATIONS - 1,
            num_history_iterations=0,
            future_duration_s=0.3,
            history_duration_s=0.0,
            iteration_duration_s=0.1,
        ),
    )


class TestLogWriterEndToEnd:
    def test_scene_api_returns_arrival_times(self, tmp_path: Path):
        scene = _scene(_write_log(tmp_path, has_arrival_time=True))
        for case in _CASES:
            modality_type, modality_id = _SCENE_KEYS[case.name]
            metadata = scene.get_modality_metadata(modality_type, modality_id)
            assert metadata is not None and metadata.has_arrival_time, case.name
            for iteration in range(_NUM_ITERATIONS):
                modality = scene.get_modality_at_iteration(iteration, modality_type, modality_id)
                assert modality is not None, case.name
                expected_us = _arrival_us(case.name, iteration, modality.timestamp.time_us)
                assert modality.arrival_timestamp == Timestamp.from_us(expected_us), case.name

    def test_arrival_times_do_not_change_synchronization(self, tmp_path: Path):
        with_arrival = _write_log(tmp_path / "with", has_arrival_time=True)
        without_arrival = _write_log(tmp_path / "without", has_arrival_time=False)

        sync_with = _read_table(with_arrival / "sync.arrow")
        sync_without = _read_table(without_arrival / "sync.arrow")
        assert sync_with.column_names == sync_without.column_names
        for column in sync_with.column_names:
            assert sync_with[column].to_pylist() == sync_without[column].to_pylist(), column

        scene_with, scene_without = _scene(with_arrival), _scene(without_arrival)
        for case in _CASES:
            modality_type, modality_id = _SCENE_KEYS[case.name]
            assert scene_with.get_all_modality_timestamps(modality_type, modality_id) == (
                scene_without.get_all_modality_timestamps(modality_type, modality_id)
            ), case.name
            timestamps = scene_with.get_all_modality_timestamps(modality_type, modality_id)
            found = scene_with.get_modality_at_timestamp(timestamps[1], modality_type, modality_id)
            assert found is not None and found.timestamp == timestamps[1], case.name

        # Without the flag the modality files are those of a log written before the column existed.
        for path in sorted(without_arrival.glob("*.arrow")):
            assert not any(name.endswith(ARRIVAL_TIME_FIELD) for name in _read_table(path).column_names), path.name
