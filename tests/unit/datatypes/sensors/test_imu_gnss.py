"""Unit tests for the IMU and GNSS modalities: datatypes, arrow round-trip, registration."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from py123d.api.scene.arrow.arrow_scene_api import MODALITY_READERS
from py123d.api.scene.arrow.modalities.arrow_gnss import ArrowGnssReader, ArrowGnssWriter
from py123d.api.scene.arrow.modalities.arrow_imu import ArrowImuReader, ArrowImuWriter
from py123d.api.utils.arrow_metadata_utils import get_metadata_from_arrow_schema, resolve_metadata_class
from py123d.datatypes import Gnss, GnssMetadata, Imu, ImuMetadata, ModalityType, Timestamp
from py123d.geometry import Vector3D
from py123d.geometry.rotation import Quaternion


def _read_table(path: Path) -> pa.Table:
    with pa.memory_map(str(path), "rb") as source:
        return pa.ipc.open_file(source).read_all()


class TestModalityRegistration:
    def test_modality_types_exist(self):
        assert ModalityType.IMU.value == 10
        assert ModalityType.GNSS.value == 11

    def test_modality_keys_without_id(self):
        assert ImuMetadata(imu_name="x").modality_key == "imu"
        assert GnssMetadata(gnss_name="x").modality_key == "gnss"

    def test_modality_keys_with_id(self):
        assert ImuMetadata(imu_name="x", imu_id="rear").modality_key == "imu.rear"
        assert GnssMetadata(gnss_name="x", gnss_id="aux").modality_key == "gnss.aux"

    def test_readers_registered(self):
        assert MODALITY_READERS[ModalityType.IMU] is ArrowImuReader
        assert MODALITY_READERS[ModalityType.GNSS] is ArrowGnssReader

    def test_metadata_class_resolution(self):
        assert resolve_metadata_class("imu") is ImuMetadata
        assert resolve_metadata_class("gnss") is GnssMetadata


class TestImuArrowRoundTrip:
    def test_minimal_imu_omits_optional_columns(self, tmp_path: Path):
        log_dir = tmp_path
        metadata = ImuMetadata(imu_name="imx5")
        writer = ArrowImuWriter(log_dir, metadata)
        for i in range(5):
            writer.write_modality(
                Imu(
                    timestamp=Timestamp.from_us(1000 + i),
                    metadata=metadata,
                    angular_velocity=Vector3D(0.1 * i, 0.2, 0.3),
                    linear_acceleration=Vector3D(0.0, 0.0, -9.81),
                )
            )
        writer.close()

        table = _read_table(log_dir / "imu.arrow")
        assert table.column_names == [
            "imu.timestamp_us",
            "imu.angular_velocity",
            "imu.linear_acceleration",
        ]

        restored_metadata = get_metadata_from_arrow_schema(table.schema, ImuMetadata)
        assert restored_metadata.imu_name == "imx5"
        assert not restored_metadata.has_orientation
        assert not restored_metadata.has_covariances

        imu = ArrowImuReader.read_at_index(3, table, restored_metadata, dataset="test")
        assert imu.timestamp.time_us == 1003
        assert imu.angular_velocity.x == pytest.approx(0.3)
        assert imu.linear_acceleration.z == pytest.approx(-9.81)
        assert imu.orientation is None
        assert imu.orientation_covariance is None

    def test_full_imu_round_trip(self, tmp_path: Path):
        log_dir = tmp_path
        metadata = ImuMetadata(imu_name="fused", has_orientation=True, has_covariances=True)
        writer = ArrowImuWriter(log_dir, metadata)
        covariance = np.arange(9, dtype=np.float64)
        writer.write_modality(
            Imu(
                timestamp=Timestamp.from_us(5),
                metadata=metadata,
                angular_velocity=Vector3D(1.0, 2.0, 3.0),
                linear_acceleration=Vector3D(4.0, 5.0, 6.0),
                orientation=Quaternion(1.0, 0.0, 0.0, 0.0),
                orientation_covariance=covariance,
                angular_velocity_covariance=covariance * 2,
                linear_acceleration_covariance=covariance * 3,
            )
        )
        writer.close()

        table = _read_table(log_dir / "imu.arrow")
        assert len(table.column_names) == 7

        restored_metadata = get_metadata_from_arrow_schema(table.schema, ImuMetadata)
        imu = ArrowImuReader.read_at_index(0, table, restored_metadata, dataset="test")
        assert imu.orientation.array == pytest.approx([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(imu.orientation_covariance, covariance)
        np.testing.assert_allclose(imu.angular_velocity_covariance, covariance * 2)
        np.testing.assert_allclose(imu.linear_acceleration_covariance, covariance * 3)


class TestGnssArrowRoundTrip:
    def test_gnss_round_trip_with_datum(self, tmp_path: Path):
        log_dir = tmp_path
        metadata = GnssMetadata(gnss_name="ublox", datum_lla=(49.0203, 8.4376, 162.4))
        writer = ArrowGnssWriter(log_dir, metadata)
        writer.write_modality(
            Gnss(
                timestamp=Timestamp.from_us(7),
                metadata=metadata,
                latitude=49.02037,
                longitude=8.43765,
                altitude=162.41,
                position_covariance=np.full(9, 0.5),
                position_covariance_type=3,
                status=1,
                service=1,
            )
        )
        # A fix without the optional quality fields.
        writer.write_modality(
            Gnss(
                timestamp=Timestamp.from_us(8),
                metadata=metadata,
                latitude=49.02038,
                longitude=8.43766,
                altitude=162.42,
            )
        )
        writer.close()

        table = _read_table(log_dir / "gnss.arrow")
        restored_metadata = get_metadata_from_arrow_schema(table.schema, GnssMetadata)
        assert restored_metadata.datum_lla == (49.0203, 8.4376, 162.4)

        full_fix = ArrowGnssReader.read_at_index(0, table, restored_metadata, dataset="test")
        assert full_fix.latitude == pytest.approx(49.02037)
        assert full_fix.status == 1
        assert full_fix.position_covariance_type == 3
        np.testing.assert_allclose(full_fix.position_covariance, 0.5)
        np.testing.assert_allclose(full_fix.lla, [49.02037, 8.43765, 162.41])

        bare_fix = ArrowGnssReader.read_at_index(1, table, restored_metadata, dataset="test")
        assert bare_fix.status is None
        assert bare_fix.position_covariance is None

        # Without has_solution_quality the columns do not exist and the fields read back None.
        assert f"{metadata.modality_key}.num_satellites" not in table.column_names
        assert full_fix.num_satellites is None
        assert full_fix.ground_speed is None

    def test_gnss_round_trip_with_solution_quality(self, tmp_path: Path):
        log_dir = tmp_path
        metadata = GnssMetadata(gnss_name="ublox", has_solution_quality=True)
        writer = ArrowGnssWriter(log_dir, metadata)
        writer.write_modality(
            Gnss(
                timestamp=Timestamp.from_us(7),
                metadata=metadata,
                latitude=49.02037,
                longitude=8.43765,
                altitude=162.41,
                status=1,
                num_satellites=17,
                fix_type=3,
                horizontal_accuracy=1.43,
                vertical_accuracy=2.10,
                position_dop=1.45,
                velocity_ned=np.array([1.039, -0.220, 0.076]),
            )
        )
        # A fix whose solution-quality fields are missing, e.g. no PVT message nearby.
        writer.write_modality(
            Gnss(timestamp=Timestamp.from_us(8), metadata=metadata, latitude=49.02038, longitude=8.43766, altitude=1.0)
        )
        writer.close()

        table = _read_table(log_dir / "gnss.arrow")
        restored_metadata = get_metadata_from_arrow_schema(table.schema, GnssMetadata)
        assert restored_metadata.has_solution_quality

        fix = ArrowGnssReader.read_at_index(0, table, restored_metadata, dataset="test")
        assert fix.num_satellites == 17
        assert fix.fix_type == 3
        assert fix.horizontal_accuracy == pytest.approx(1.43)
        assert fix.vertical_accuracy == pytest.approx(2.10)
        assert fix.position_dop == pytest.approx(1.45)
        np.testing.assert_allclose(fix.velocity_ned, [1.039, -0.220, 0.076])
        assert fix.ground_speed == pytest.approx(1.062, abs=1e-3)

        unmatched = ArrowGnssReader.read_at_index(1, table, restored_metadata, dataset="test")
        assert unmatched.num_satellites is None
        assert unmatched.velocity_ned is None
        assert unmatched.ground_speed is None

    def test_gnss_solution_quality_dropped_without_flag(self, tmp_path: Path):
        """A fix carrying quality fields must not be written silently when the flag is off."""
        log_dir = tmp_path
        metadata = GnssMetadata(gnss_name="ublox")
        writer = ArrowGnssWriter(log_dir, metadata)
        with pytest.raises(AssertionError, match="has_solution_quality"):
            writer.write_modality(
                Gnss(
                    timestamp=Timestamp.from_us(7),
                    metadata=metadata,
                    latitude=49.0,
                    longitude=8.4,
                    altitude=162.0,
                    num_satellites=17,
                )
            )
        writer.close()

    def test_gnss_metadata_without_datum(self):
        metadata = GnssMetadata(gnss_name="ublox")
        restored = GnssMetadata.from_dict(metadata.to_dict())
        assert restored.datum_lla is None
        assert restored.has_solution_quality is False
        assert restored.gnss_name == "ublox"


class TestBarometerMagnetometerRoundTrip:
    def test_registration(self):
        from py123d.api.scene.arrow.modalities.arrow_barometer import ArrowBarometerReader
        from py123d.api.scene.arrow.modalities.arrow_magnetometer import ArrowMagnetometerReader
        from py123d.datatypes import BarometerMetadata, MagnetometerMetadata

        assert ModalityType.BAROMETER.value == 12
        assert ModalityType.MAGNETOMETER.value == 13
        assert BarometerMetadata(barometer_name="x").modality_key == "barometer"
        assert MagnetometerMetadata(magnetometer_name="x").modality_key == "magnetometer"
        assert MODALITY_READERS[ModalityType.BAROMETER] is ArrowBarometerReader
        assert MODALITY_READERS[ModalityType.MAGNETOMETER] is ArrowMagnetometerReader
        assert resolve_metadata_class("barometer") is BarometerMetadata
        assert resolve_metadata_class("magnetometer") is MagnetometerMetadata

    def test_barometer_round_trip(self, tmp_path: Path):
        from py123d.api.scene.arrow.modalities.arrow_barometer import ArrowBarometerReader, ArrowBarometerWriter
        from py123d.datatypes import Barometer, BarometerMetadata

        log_dir = tmp_path
        metadata = BarometerMetadata(barometer_name="imx5")
        writer = ArrowBarometerWriter(log_dir, metadata)
        writer.write_modality(
            Barometer(
                timestamp=Timestamp.from_us(1),
                metadata=metadata,
                pressure=96283.3,
                msl_altitude=430.5,
                temperature=45.9,
                humidity=0.0,
            )
        )
        writer.write_modality(Barometer(timestamp=Timestamp.from_us(2), metadata=metadata, pressure=96280.0))
        writer.close()

        table = _read_table(log_dir / "barometer.arrow")
        from py123d.datatypes import BarometerMetadata as BM

        restored_metadata = get_metadata_from_arrow_schema(table.schema, BM)
        full = ArrowBarometerReader.read_at_index(0, table, restored_metadata, dataset="test")
        assert full.pressure == pytest.approx(96283.3)
        assert full.msl_altitude == pytest.approx(430.5)
        bare = ArrowBarometerReader.read_at_index(1, table, restored_metadata, dataset="test")
        assert bare.msl_altitude is None and bare.temperature is None

    def test_magnetometer_round_trip(self, tmp_path: Path):
        from py123d.api.scene.arrow.modalities.arrow_magnetometer import (
            ArrowMagnetometerReader,
            ArrowMagnetometerWriter,
        )
        from py123d.datatypes import Magnetometer, MagnetometerMetadata

        log_dir = tmp_path
        metadata = MagnetometerMetadata(magnetometer_name="imx5")
        writer = ArrowMagnetometerWriter(log_dir, metadata)
        writer.write_modality(
            Magnetometer(
                timestamp=Timestamp.from_us(1),
                metadata=metadata,
                magnetic_field=Vector3D(0.5e-6, 2.3e-6, 10.6e-6),
            )
        )
        writer.close()

        table = _read_table(log_dir / "magnetometer.arrow")
        restored_metadata = get_metadata_from_arrow_schema(table.schema, MagnetometerMetadata)
        reading = ArrowMagnetometerReader.read_at_index(0, table, restored_metadata, dataset="test")
        assert reading.magnetic_field.array == pytest.approx([0.5e-6, 2.3e-6, 10.6e-6])
        assert reading.magnetic_field_covariance is None
