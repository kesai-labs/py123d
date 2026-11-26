"""Tests of the kedec lidar codec: error bounds, exact features, sections, modes and corrupt blobs."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from py123d.common.io.lidar.kedec_lidar_io import (
    KEDEC_MAGIC,
    encode_point_cloud_as_kedec_binary,
    is_kedec_binary,
    load_point_cloud_from_kedec_binary,
    read_kedec_head,
)
from py123d.common.io.lidar.point_cloud_codec_config import PointCloudCodecConfig

from .synthetic_scans import grid_bound, random_pose, spherical_bound, spinning_scan, to_vehicle_frame

IDENTITY = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]


def _roundtrip(points, features=None, config=None, sensor_poses=None):
    blob = encode_point_cloud_as_kedec_binary(points, features, config=config, sensor_poses=sensor_poses)
    decoded, decoded_features = load_point_cloud_from_kedec_binary(blob)
    return blob, decoded, decoded_features


def _errors(points, decoded, uid):
    """Distance of each decoded point from its input point, and the input points in decoded order."""
    original = points[uid].astype(np.float64)
    return np.linalg.norm(decoded.astype(np.float64) - original, axis=1), original


def _random_directions(rng, n):
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


class TestSphericalMode:
    @pytest.mark.parametrize("seed", range(5))
    def test_error_bound_random_poses(self, seed):
        """Every point lies within the spherical bound around a random sensor pose, up to 500 m."""
        rng = np.random.default_rng(seed)
        pose = random_pose(rng)
        center = np.asarray(pose[:3])
        r = np.concatenate([rng.uniform(0.0, 5.0, 2000), rng.uniform(5.0, 500.0, 8000)])
        points = (center + _random_directions(rng, len(r)) * r[:, None]).astype(np.float32)
        uid = np.arange(len(points), dtype=np.int64)
        blob, decoded, features = _roundtrip(points, {"uid": uid}, sensor_poses={0: pose})
        errors, original = _errors(points, decoded, features["uid"])
        ranges = np.linalg.norm(original - center, axis=1)
        assert np.all(errors <= spherical_bound(ranges)), float((errors - spherical_bound(ranges)).max())
        assert read_kedec_head(blob)[0]["sections"][0]["mode"] == "spherical"

    def test_bound_values_of_the_decision(self):
        """2 cm up to about 175 m and 5.04 cm at 500 m for the default steps (without the test tolerances)."""
        angle = np.radians(0.004)
        assert np.sqrt(0.01**2 + 2 * (175.0 * angle) ** 2) < 0.02
        assert abs(np.sqrt(0.01**2 + 2 * (500.0 * angle) ** 2) - 0.0504) < 1e-4

    def test_azimuth_wrap_and_poles(self):
        """Points at +-180 degrees azimuth and at +-90 degrees elevation decode within the bound."""
        r = 100.0
        az = np.radians([180.0, -180.0, 179.9999, -179.9999, 0.0, 90.0])
        horizontal = np.stack([np.cos(az), np.sin(az), np.zeros_like(az)], 1)
        poles = np.array([[0, 0, 1.0], [0, 0, -1.0], [1e-9, 0, 1.0]])
        points = (np.concatenate([horizontal, poles]) * r).astype(np.float32)
        uid = np.arange(len(points))
        _, decoded, features = _roundtrip(points, {"uid": uid})
        errors, original = _errors(points, decoded, features["uid"])
        assert np.all(errors <= spherical_bound(np.linalg.norm(original, axis=1)))

    def test_points_at_the_sensor_position(self):
        """Points at the sensor position decode to the sensor position."""
        pose = [1.5, -0.25, 2.0, 1.0, 0.0, 0.0, 0.0]
        points = np.array([[1.5, -0.25, 2.0], [1.5, -0.25, 2.0], [11.5, -0.25, 2.0]], dtype=np.float32)
        _, decoded, features = _roundtrip(points, {"uid": np.arange(3)}, sensor_poses={7: pose})
        errors, _ = _errors(points, decoded, features["uid"])
        assert errors.max() <= 0.0101

    def test_pose_frame_option(self):
        """The spherical mode in the full sensor frame keeps the same bound."""
        rng = np.random.default_rng(3)
        pose = random_pose(rng)
        points, features = spinning_scan(rng, pose, n_beams=16, n_columns=180)
        del features["column"]
        config = PointCloudCodecConfig(kedec_frame="pose")
        blob, decoded, decoded_features = _roundtrip(points, features, config, {0: pose})
        errors, original = _errors(points, decoded, decoded_features["uid"])
        ranges = np.linalg.norm(original - np.asarray(pose[:3]), axis=1)
        assert np.all(errors <= spherical_bound(ranges))
        assert read_kedec_head(blob)[0]["sections"][0]["params"]["frame"] == "pose"

    def test_custom_steps(self):
        """Coarser steps give a smaller blob and keep their own bound."""
        rng = np.random.default_rng(4)
        points, features = spinning_scan(rng, IDENTITY, n_beams=16, n_columns=180)
        del features["column"]
        fine = encode_point_cloud_as_kedec_binary(points, features)
        config = PointCloudCodecConfig(kedec_range_step_m=0.05, kedec_angle_step_deg=0.02)
        coarse, decoded, decoded_features = _roundtrip(points, features, config)
        assert len(coarse) < len(fine)
        errors, original = _errors(points, decoded, decoded_features["uid"])
        assert np.all(errors <= spherical_bound(np.linalg.norm(original, axis=1), 0.05, 0.02))

    def test_ring_orders_the_points(self):
        """With a ring (channel) but no column, the points are sorted by ring, then azimuth, and the ring is kept."""
        rng = np.random.default_rng(5)
        points, features = spinning_scan(rng, IDENTITY, n_beams=8, n_columns=90)
        del features["column"]
        blob, decoded, decoded_features = _roundtrip(points, features)
        section = read_kedec_head(blob)[0]["sections"][0]
        assert section["mode"] == "spherical" and section["params"]["order"] == "channel"
        ring = decoded_features["channel"]
        assert np.all(np.diff(ring.astype(np.int64)) >= 0)
        assert np.array_equal(ring, features["channel"][decoded_features["uid"]])


class TestGridMode:
    @pytest.mark.parametrize("seed", range(4))
    def test_error_bound(self, seed):
        """Every point lies within the grid bound (1.73 cm for 2 cm steps), around a random sensor pose."""
        rng = np.random.default_rng(100 + seed)
        pose = random_pose(rng)
        points, features = spinning_scan(rng, pose)
        blob, decoded, decoded_features = _roundtrip(points, features, sensor_poses={3: pose})
        section = read_kedec_head(blob)[0]["sections"][0]
        assert section["mode"] == "grid"
        errors, original = _errors(points, decoded, decoded_features["uid"])
        ranges = np.linalg.norm(original - np.asarray(pose[:3]), axis=1)
        assert np.all(errors <= grid_bound(ranges)), float((errors - grid_bound(ranges)).max())
        assert errors.max() < 0.0175

    def test_order_is_beam_then_column(self):
        rng = np.random.default_rng(7)
        points, features = spinning_scan(rng, IDENTITY, n_beams=8, n_columns=60)
        _, _, decoded_features = _roundtrip(points, features)
        key = decoded_features["channel"].astype(np.int64) * 100_000 + decoded_features["column"]
        assert np.all(np.diff(key) >= 0)

    def test_missing_beams_and_columns(self):
        """Beam and column numbers with gaps (unused table entries) decode within the bound."""
        rng = np.random.default_rng(8)
        points, features = spinning_scan(rng, IDENTITY, n_beams=8, n_columns=60)
        features["channel"] = (features["channel"].astype(np.int64) * 3 + 5).astype(np.uint8)
        features["column"] = (features["column"].astype(np.int64) * 7).astype(np.uint16)
        _, decoded, decoded_features = _roundtrip(points, features)
        errors, original = _errors(points, decoded, decoded_features["uid"])
        assert np.all(errors <= grid_bound(np.linalg.norm(original, axis=1)))

    def test_grid_is_smaller_than_spherical_for_a_beam_scan(self):
        rng = np.random.default_rng(9)
        pose = random_pose(rng)
        points, features = spinning_scan(rng, pose, n_beams=64, n_columns=720)
        del features["uid"]
        grid = encode_point_cloud_as_kedec_binary(points, features, sensor_poses={0: pose})
        spherical = encode_point_cloud_as_kedec_binary(
            points, features, PointCloudCodecConfig(kedec_mode="spherical"), sensor_poses={0: pose}
        )
        assert len(grid) < len(spherical)


class TestModes:
    def test_auto_rule(self):
        """Auto: grid only with both a beam (channel) and a column index; otherwise spherical."""
        rng = np.random.default_rng(10)
        points, features = spinning_scan(rng, IDENTITY, n_beams=4, n_columns=30)
        cases = {
            ("channel", "column"): "grid",
            ("channel",): "spherical",
            ("column",): "spherical",
            (): "spherical",
        }
        for kept, mode in cases.items():
            subset = {k: v for k, v in features.items() if k in kept or k in ("intensity", "timestamps", "uid")}
            head, _ = read_kedec_head(encode_point_cloud_as_kedec_binary(points, subset))
            assert head["sections"][0]["mode"] == mode, kept

    def test_negative_or_float_indices_use_spherical(self):
        rng = np.random.default_rng(11)
        points, features = spinning_scan(rng, IDENTITY, n_beams=4, n_columns=30)
        signed = dict(features, column=features["column"].astype(np.int32) - 1)
        floats = dict(features, column=features["column"].astype(np.float32))
        for case in (signed, floats):
            head, _ = read_kedec_head(encode_point_cloud_as_kedec_binary(points, case))
            assert head["sections"][0]["mode"] == "spherical"

    def test_forced_modes(self):
        rng = np.random.default_rng(12)
        points, features = spinning_scan(rng, IDENTITY, n_beams=4, n_columns=30)
        spherical = PointCloudCodecConfig(kedec_mode="spherical")
        head, _ = read_kedec_head(encode_point_cloud_as_kedec_binary(points, features, spherical))
        assert head["sections"][0]["mode"] == "spherical"
        no_column = {k: v for k, v in features.items() if k != "column"}
        with pytest.raises(ValueError, match="grid mode"):
            encode_point_cloud_as_kedec_binary(points, no_column, PointCloudCodecConfig(kedec_mode="grid"))

    def test_invalid_config(self):
        with pytest.raises(AssertionError):
            PointCloudCodecConfig(kedec_mode="cartesian")
        with pytest.raises(AssertionError):
            PointCloudCodecConfig(kedec_angle_step_deg=0.0)
        with pytest.raises(AssertionError):
            PointCloudCodecConfig(kedec_zstd_level=30)


class TestFeatures:
    def test_features_are_exact_and_permuted_together(self):
        """All features decode bit-exactly and stay with their points."""
        rng = np.random.default_rng(20)
        n = 5000
        points = rng.uniform(-80, 80, (n, 3)).astype(np.float32)
        features = {
            "uid": np.arange(n, dtype=np.int64),
            "intensity": rng.integers(0, 256, n).astype(np.uint8),
            "timestamps": 1_700_000_000_000_000 + rng.integers(0, 100_000, n),
            "range": rng.uniform(0, 100, n).astype(np.float32),
            "elongation": rng.normal(size=n).astype(np.float64),
            "flag": rng.random(n) < 0.5,
            "signed": rng.integers(-(2**62), 2**62, n),
            "rgb": rng.integers(0, 256, (n, 3)).astype(np.uint8),
            "nan_values": np.where(rng.random(n) < 0.1, np.nan, 1.0).astype(np.float32),
        }
        _, decoded, decoded_features = _roundtrip(points, features)
        uid = decoded_features["uid"]
        assert sorted(uid.tolist()) == list(range(n))
        assert list(decoded_features) == list(features)
        for name, values in features.items():
            assert decoded_features[name].dtype == values.dtype, name
            assert decoded_features[name].shape == values.shape, name
            np.testing.assert_array_equal(decoded_features[name], values[uid], err_msg=name)

    def test_point_times_bit_identical_with_long_spans(self):
        """Point times are exact, also for spans of 2**32 us and more within one scan."""
        rng = np.random.default_rng(21)
        n = 3000
        points = rng.uniform(-50, 50, (n, 3)).astype(np.float32)
        times = 1_784_208_918_099_704 + np.sort(rng.integers(0, 2**40, n))
        times[-1] = times[0] + 2**33 + 17
        _, _, decoded_features = _roundtrip(points, {"timestamps": times, "uid": np.arange(n)})
        np.testing.assert_array_equal(decoded_features["timestamps"], times[decoded_features["uid"]])

    def test_extreme_integers(self):
        n = 4
        points = np.ones((n, 3), dtype=np.float32)
        features = {
            "uid": np.arange(n),
            "i64": np.array([np.iinfo(np.int64).min, np.iinfo(np.int64).max, 0, -1], np.int64),
            "u64": np.array([0, np.iinfo(np.uint64).max, 1, 2**63], np.uint64),
        }
        _, _, decoded_features = _roundtrip(points, features)
        for name in ("i64", "u64"):
            np.testing.assert_array_equal(decoded_features[name], features[name][decoded_features["uid"]])

    def test_non_finite_points_are_dropped_and_counted(self):
        rng = np.random.default_rng(22)
        points = rng.uniform(-20, 20, (100, 3)).astype(np.float32)
        points[[3, 50]] = np.nan
        points[70, 1] = np.inf
        uid = np.arange(100)
        blob, decoded, decoded_features = _roundtrip(points, {"uid": uid})
        head, _ = read_kedec_head(blob)
        assert head["dropped"] == 3 and head["n"] == 97 and len(decoded) == 97
        assert not set(decoded_features["uid"].tolist()) & {3, 50, 70}

    def test_empty_scan(self):
        features = {"intensity": np.zeros(0, np.uint8), "timestamps": np.zeros(0, np.int64)}
        blob, decoded, decoded_features = _roundtrip(np.zeros((0, 3), np.float32), features)
        assert decoded.shape == (0, 3) and decoded.dtype == np.float32
        assert decoded_features["intensity"].dtype == np.uint8 and len(decoded_features["timestamps"]) == 0
        _, decoded, decoded_features = _roundtrip(np.zeros((0, 3), np.float32))
        assert decoded.shape == (0, 3) and decoded_features is None

    def test_no_features(self):
        points = np.random.default_rng(23).uniform(-5, 5, (50, 3)).astype(np.float32)
        _, decoded, decoded_features = _roundtrip(points)
        assert decoded_features is None and decoded.shape == (50, 3)

    def test_invalid_inputs(self):
        points = np.zeros((10, 3), np.float32)
        with pytest.raises(ValueError):
            encode_point_cloud_as_kedec_binary(np.zeros((10, 2), np.float32))
        with pytest.raises(ValueError):
            encode_point_cloud_as_kedec_binary(points, {"short": np.zeros(9)})
        with pytest.raises(ValueError):
            encode_point_cloud_as_kedec_binary(points, {"@r": np.zeros(10)})
        with pytest.raises(ValueError):
            encode_point_cloud_as_kedec_binary(points, {"names": np.array(["a"] * 10, dtype=object)})


class TestSections:
    def test_merged_cloud_is_coded_per_sensor(self):
        """A cloud with ids is split per sensor; each section uses its own pose; ids decode exactly."""
        rng = np.random.default_rng(30)
        poses = {2: random_pose(rng), 5: random_pose(rng)}
        clouds, feature_list = [], []
        for sensor_id, pose in poses.items():
            points, features = spinning_scan(rng, pose, n_beams=8, n_columns=90)
            features["ids"] = np.full(len(points), sensor_id, np.uint8)
            clouds.append(points)
            feature_list.append(features)
        points = np.concatenate(clouds)
        features = {name: np.concatenate([f[name] for f in feature_list]) for name in feature_list[0]}
        features["uid"] = np.arange(len(points))
        blob, decoded, decoded_features = _roundtrip(points, features, sensor_poses=poses)
        head, _ = read_kedec_head(blob)
        assert [s["id"] for s in head["sections"]] == [2, 5]
        assert [s["pose"] for s in head["sections"]] == [poses[2], poses[5]]
        uid = decoded_features["uid"]
        np.testing.assert_array_equal(decoded_features["ids"], features["ids"][uid])
        assert np.all(np.diff(decoded_features["ids"].astype(int)) >= 0)
        errors, original = _errors(points, decoded, uid)
        centers = np.array([poses[int(i)][:3] for i in decoded_features["ids"]])
        assert np.all(errors <= grid_bound(np.linalg.norm(original - centers, axis=1)))

    def test_unknown_sensor_uses_identity(self):
        points = np.random.default_rng(31).uniform(-5, 5, (20, 3)).astype(np.float32)
        blob = encode_point_cloud_as_kedec_binary(points, {"ids": np.full(20, 9, np.uint8)}, sensor_poses={1: IDENTITY})
        assert read_kedec_head(blob)[0]["sections"][0]["pose"] == IDENTITY

    def test_single_pose_without_ids(self):
        pose = [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]
        points = np.random.default_rng(32).uniform(-5, 5, (20, 3)).astype(np.float32)
        blob = encode_point_cloud_as_kedec_binary(points, sensor_poses={4: pose})
        section = read_kedec_head(blob)[0]["sections"][0]
        assert section["pose"] == pose and section["id"] is None
        blob = encode_point_cloud_as_kedec_binary(points, sensor_poses={4: pose, 5: IDENTITY})
        assert read_kedec_head(blob)[0]["sections"][0]["pose"] == IDENTITY

    def test_pose_convention_matches_py123d(self):
        """The codec's sensor frame is py123d's: a grid scan in a rotated sensor keeps its tables exact."""
        rng = np.random.default_rng(33)
        pose = random_pose(rng)
        sensor = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        points = to_vehicle_frame(sensor, pose).astype(np.float32)
        config = PointCloudCodecConfig(kedec_frame="pose")
        blob, decoded, features = _roundtrip(points, {"uid": np.arange(3)}, config, {0: pose})
        errors, _ = _errors(points, decoded, features["uid"])
        assert errors.max() < 0.011


class TestBlobFormat:
    def _blob(self):
        rng = np.random.default_rng(40)
        points, features = spinning_scan(rng, IDENTITY, n_beams=4, n_columns=30)
        return encode_point_cloud_as_kedec_binary(points, features)

    def test_magic_and_version(self):
        blob = self._blob()
        assert blob[:4] == KEDEC_MAGIC and blob[4] == 1 and is_kedec_binary(blob)
        assert not is_kedec_binary(b"DRACO") and not is_kedec_binary(b"")

    def test_unknown_version_raises(self):
        blob = bytearray(self._blob())
        blob[4] = 2
        with pytest.raises(ValueError, match="version 2"):
            load_point_cloud_from_kedec_binary(bytes(blob))

    def test_truncated_or_extended_blob_raises(self):
        blob = self._blob()
        for broken in (blob[:-1], blob + b"\x00", blob[:8], blob[:20]):
            with pytest.raises(ValueError):
                load_point_cloud_from_kedec_binary(broken)

    def test_corrupt_head_raises(self):
        blob = self._blob()
        head_size = struct.unpack_from("<I", blob, 6)[0]
        broken = blob[:10] + b"\xc1" * head_size + blob[10 + head_size :]
        with pytest.raises(ValueError):
            load_point_cloud_from_kedec_binary(broken)

    def test_corrupt_part_raises(self):
        blob = bytearray(self._blob())
        blob[-5] ^= 0xFF
        with pytest.raises(ValueError):
            load_point_cloud_from_kedec_binary(bytes(blob))

    def test_encoding_is_deterministic(self):
        assert self._blob() == self._blob()
