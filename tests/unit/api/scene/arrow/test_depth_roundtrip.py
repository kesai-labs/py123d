"""Roundtrip tests for the per-pixel depth camera modality.

Depth is a quantized integer raster: unlike a segmentation class-id map, the *stored* values are a
lossy projection of continuous metric depth. Two distinct properties therefore need pinning down:

1. The **raster** must survive the storage layer bit-exactly (no lossy codec, no blending resample).
2. The **quantization** must be a faithful, bounded approximation of the metric depth (round-to-nearest
   with error at most half a step, saturating rather than wrapping at the far plane).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from py123d.api.scene.arrow.arrow_log_writer import ArrowLogWriter, SyncConfig
from py123d.api.scene.arrow.arrow_scene_api import ArrowSceneAPI
from py123d.api.scene.arrow.modalities.arrow_camera import ArrowCameraReader, ArrowCameraWriter
from py123d.api.scene.arrow.utils.log_writer_config import LogWriterConfig
from py123d.api.utils.arrow_metadata_utils import resolve_metadata_class
from py123d.datatypes import Timestamp
from py123d.datatypes.metadata import SceneMetadata
from py123d.datatypes.modalities.base_modality import ModalityType
from py123d.datatypes.sensors.base_camera import Camera, CameraChannelType, CameraID
from py123d.datatypes.sensors.depth_camera import DepthCameraMetadata
from py123d.datatypes.sensors.pinhole_camera import PinholeCameraMetadata, PinholeDistortion, PinholeIntrinsics
from py123d.geometry.pose import PoseSE3

from ..conftest import make_log_metadata

# ----------------------------------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------------------------------


def _make_rgb_camera_metadata(camera_id: CameraID = CameraID.PCAM_F0) -> PinholeCameraMetadata:
    return PinholeCameraMetadata(
        camera_name="front_camera",
        camera_id=camera_id,
        intrinsics=PinholeIntrinsics(fx=500.0, fy=500.0, cx=160.0, cy=120.0),
        distortion=PinholeDistortion(k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0),
        width=320,
        height=240,
        camera_to_imu_se3=PoseSE3.identity(),
    )


def _make_depth_camera_metadata(
    camera_id: CameraID = CameraID.PCAM_F0,
    max_depth: float = 96.0,
    depth_bits: int = 16,
) -> DepthCameraMetadata:
    return DepthCameraMetadata(
        camera_metadata=_make_rgb_camera_metadata(camera_id),
        max_depth=max_depth,
        depth_bits=depth_bits,
    )


def _make_metric_depth(height: int = 240, width: int = 320, high: float = 120.0) -> np.ndarray:
    """A metric depth map in metres, deliberately exceeding any far plane used below so clipping is exercised."""
    return np.random.RandomState(7).uniform(0.0, high, size=(height, width)).astype(np.float64)


# ----------------------------------------------------------------------------------------------------------------------
# Quantization contract (pure, no I/O)
# ----------------------------------------------------------------------------------------------------------------------


class TestDepthQuantization:
    """encode_depth / decode_depth must be a bounded, saturating, round-to-nearest quantizer."""

    @pytest.mark.parametrize(("depth_bits", "max_depth", "dtype"), [(8, 50.0, np.uint8), (16, 96.0, np.uint16)])
    def test_dtype_and_derived_quantities(self, depth_bits: int, max_depth: float, dtype: type):
        metadata = _make_depth_camera_metadata(max_depth=max_depth, depth_bits=depth_bits)
        assert metadata.depth_dtype == dtype
        assert metadata.max_raw == (1 << depth_bits) - 1
        assert metadata.depth_resolution == pytest.approx(max_depth / metadata.max_raw)
        assert metadata.encode_depth(_make_metric_depth()).dtype == dtype

    @pytest.mark.parametrize(("depth_bits", "max_depth"), [(8, 50.0), (16, 96.0)])
    def test_roundtrip_error_within_half_a_step(self, depth_bits: int, max_depth: float):
        metadata = _make_depth_camera_metadata(max_depth=max_depth, depth_bits=depth_bits)
        depth_m = _make_metric_depth()
        decoded = metadata.decode_depth(metadata.encode_depth(depth_m))

        # Only the unclipped region is recoverable; beyond the far plane the value is saturated by design.
        clipped = np.clip(depth_m, 0.0, max_depth)
        error = np.abs(decoded - clipped).max()
        # Half a quantization step, plus a float32 ulp at the top of the range for the dequantized dtype.
        tolerance = metadata.depth_resolution / 2.0 + np.spacing(np.float32(max_depth))
        assert error <= tolerance, f"{error} > {tolerance}"

    @pytest.mark.parametrize("depth_bits", [8, 16])
    def test_round_to_nearest_not_truncation(self, depth_bits: int):
        # A naive `(d / max_depth * max_raw).astype(uint)` truncates, biasing every pixel downward by up
        # to a full step (half a step on average). Rounding halves the worst-case error.
        metadata = _make_depth_camera_metadata(max_depth=100.0, depth_bits=depth_bits)
        step = metadata.depth_resolution
        # Just above a step boundary's midpoint: must round up, not truncate down.
        depth_m = np.full((1, 1), step * 0.51, dtype=np.float64)
        assert metadata.encode_depth(depth_m)[0, 0] == 1

    @pytest.mark.parametrize(("depth_bits", "max_depth"), [(8, 50.0), (16, 96.0)])
    def test_far_plane_saturates_and_does_not_wrap(self, depth_bits: int, max_depth: float):
        metadata = _make_depth_camera_metadata(max_depth=max_depth, depth_bits=depth_bits)
        # A CARLA sky pixel sits far beyond the far plane. It must clamp to max_raw, never wrap to 0.
        raw = metadata.encode_depth(np.array([[0.0, max_depth, max_depth * 20.0]]))
        assert raw.tolist() == [[0, metadata.max_raw, metadata.max_raw]]
        assert metadata.decode_depth(raw)[0, 2] == pytest.approx(max_depth, rel=1e-6)

    @pytest.mark.parametrize("depth_bits", [8, 16])
    def test_non_finite_and_negative_depth_are_clamped(self, depth_bits: int):
        # astype() on NaN/inf is undefined behaviour in numpy and yields garbage integers; encode_depth
        # must neutralize them before the cast. NaN/+inf read as "infinitely far", -inf/negative as zero.
        metadata = _make_depth_camera_metadata(max_depth=50.0, depth_bits=depth_bits)
        raw = metadata.encode_depth(np.array([[np.nan, np.inf, -np.inf, -5.0]]))
        assert raw.tolist() == [[metadata.max_raw, metadata.max_raw, 0, 0]]

    def test_rejects_unsupported_bit_depth(self):
        with pytest.raises(AssertionError):
            _make_depth_camera_metadata(depth_bits=32)

    def test_rejects_non_positive_max_depth(self):
        with pytest.raises(AssertionError):
            _make_depth_camera_metadata(max_depth=0.0)

    def test_rejects_non_2d_depth(self):
        metadata = _make_depth_camera_metadata()
        with pytest.raises(AssertionError):
            metadata.encode_depth(np.zeros((240, 320, 3)))


class TestDepthTransform:
    """The ``inverse`` transform must trade absolute precision for near-constant relative precision."""

    def test_inverse_requires_positive_min_depth(self):
        with pytest.raises(AssertionError):
            DepthCameraMetadata(
                camera_metadata=_make_rgb_camera_metadata(), max_depth=96.0, depth_transform="inverse", min_depth=0.0
            )

    def test_rejects_unknown_transform(self):
        with pytest.raises(AssertionError):
            DepthCameraMetadata(camera_metadata=_make_rgb_camera_metadata(), max_depth=96.0, depth_transform="log")

    def test_inverse_roundtrip_is_bounded_far_and_fine_near(self):
        # Disparity spacing concentrates resolution near the camera: relative error stays bounded across
        # the whole range but is orders of magnitude finer up close than at the far plane.
        metadata = DepthCameraMetadata(
            camera_metadata=_make_rgb_camera_metadata(),
            max_depth=1000.0,
            depth_bits=16,
            depth_transform="inverse",
            min_depth=0.5,
        )
        depth_m = np.geomspace(0.5, 1000.0, num=4000).astype(np.float64)
        decoded = metadata.decode_depth(metadata.encode_depth(depth_m.reshape(40, 100))).reshape(-1)
        relative_error = np.abs(decoded - depth_m) / depth_m
        assert relative_error.max() < 2e-2  # bounded everywhere
        assert relative_error[depth_m < 10.0].max() < 2e-4  # and ~100x finer in the near field

    def test_inverse_beats_linear_in_the_near_field(self):
        # The whole point: at short range, disparity spacing is far more accurate than uniform metric
        # spacing for the same bit budget and far plane.
        near = np.full((1, 1), 1.0, dtype=np.float64)
        common = dict(camera_metadata=_make_rgb_camera_metadata(), max_depth=1000.0, depth_bits=16)
        linear = DepthCameraMetadata(depth_transform="linear", **common)
        inverse = DepthCameraMetadata(depth_transform="inverse", min_depth=0.5, **common)
        linear_error = abs(linear.decode_depth(linear.encode_depth(near))[0, 0] - 1.0)
        inverse_error = abs(inverse.decode_depth(inverse.encode_depth(near))[0, 0] - 1.0)
        assert inverse_error < linear_error


class TestDepthInvalidSentinel:
    """With ``has_invalid`` set, code 0 is reserved for "no measurement" and decodes to NaN."""

    def _metadata(self, **kwargs) -> DepthCameraMetadata:
        return DepthCameraMetadata(
            camera_metadata=_make_rgb_camera_metadata(), max_depth=96.0, has_invalid=True, **kwargs
        )

    def test_invalid_pixels_encode_to_zero_and_decode_to_nan(self):
        metadata = self._metadata()
        raw = metadata.encode_depth(np.array([[np.nan, np.inf, -np.inf, 0.0, -5.0, 10.0]]))
        # Everything without a finite, positive measurement collapses to the sentinel; 10 m is valid.
        assert raw[0, :5].tolist() == [0, 0, 0, 0, 0]
        assert raw[0, 5] >= 1
        decoded = metadata.decode_depth(raw)
        assert np.isnan(decoded[0, :5]).all()
        assert decoded[0, 5] == pytest.approx(10.0, abs=metadata.depth_resolution)

    def test_valid_depth_never_uses_the_sentinel(self):
        # A real (finite, positive) depth must land in [1, max_raw], never on 0, or it would be lost.
        metadata = self._metadata()
        raw = metadata.encode_depth(_make_metric_depth(high=96.0))
        assert raw.min() >= 1

    def test_metadata_dict_roundtrip_preserves_new_knobs(self):
        metadata = DepthCameraMetadata(
            camera_metadata=_make_rgb_camera_metadata(),
            max_depth=200.0,
            depth_bits=16,
            depth_transform="inverse",
            min_depth=1.0,
            has_invalid=True,
            depth_type="ray_distance",
        )
        restored = DepthCameraMetadata.from_dict(metadata.to_dict())
        assert restored.depth_transform == "inverse"
        assert restored.min_depth == 1.0
        assert restored.has_invalid is True
        assert restored.depth_type == "ray_distance"

    def test_from_dict_defaults_match_historic_behaviour(self):
        # A blob written before these knobs existed must read back as linear / z-depth / no sentinel.
        legacy = {"camera_metadata": _make_rgb_camera_metadata().to_dict(), "max_depth": 96.0, "depth_bits": 16}
        restored = DepthCameraMetadata.from_dict(legacy)
        assert (restored.depth_transform, restored.min_depth) == ("linear", 0.0)
        assert restored.has_invalid is False
        assert restored.depth_type == "z_depth"


# ----------------------------------------------------------------------------------------------------------------------
# Modality routing and metadata
# ----------------------------------------------------------------------------------------------------------------------


class TestDepthCameraMetadata:
    """The DEPTH channel must route to its own ModalityType, Arrow file, and metadata class."""

    def test_modality_routing(self):
        metadata = _make_depth_camera_metadata()
        assert metadata.channel_type == CameraChannelType.DEPTH
        assert metadata.modality_type == ModalityType.CAMERA_DEPTH
        assert metadata.modality_id == CameraID.PCAM_F0
        # Never collides with the RGB `camera.pcam_f0` sharing its camera_id.
        assert metadata.modality_key == "camera_depth.pcam_f0"
        assert metadata.modality_key != _make_rgb_camera_metadata().modality_key

    def test_geometry_is_delegated_to_the_sibling_camera(self):
        rgb = _make_rgb_camera_metadata()
        metadata = DepthCameraMetadata(camera_metadata=rgb, max_depth=96.0)
        assert (metadata.width, metadata.height) == (rgb.width, rgb.height)
        assert metadata.camera_model == rgb.camera_model
        assert metadata.camera_name == rgb.camera_name

        points_cam = np.array([[0.0, 0.0, 10.0], [1.0, 1.0, 20.0]])
        for actual, expected in zip(metadata.project_to_image(points_cam), rgb.project_to_image(points_cam)):
            np.testing.assert_array_equal(actual, expected)

    @pytest.mark.parametrize(("depth_bits", "max_depth"), [(8, 50.0), (16, 96.0)])
    def test_metadata_dict_roundtrip(self, depth_bits: int, max_depth: float):
        metadata = _make_depth_camera_metadata(max_depth=max_depth, depth_bits=depth_bits)
        restored = DepthCameraMetadata.from_dict(metadata.to_dict())
        assert restored.max_depth == metadata.max_depth
        assert restored.depth_bits == metadata.depth_bits
        assert restored.camera_id == metadata.camera_id
        assert restored.modality_key == metadata.modality_key

    def test_reader_path_resolves_depth_metadata(self):
        # Without this, `camera_depth.*.arrow` files are silently skipped at read time with only a warning.
        assert resolve_metadata_class("camera_depth.pcam_f0") is DepthCameraMetadata

    def test_rgb_image_colorizes_depth(self):
        # `Camera.rgb_image` raises on unknown channel types, so viser/matplotlib would hard-fail
        # on a depth stream without a colorization branch.
        metadata = _make_depth_camera_metadata()
        camera = Camera(
            metadata=metadata,
            image=metadata.encode_depth(_make_metric_depth()),
            camera_to_global_se3=PoseSE3.identity(),
            timestamp=Timestamp.from_us(1000),
        )
        rgb = camera.rgb_image
        assert rgb.shape == (240, 320, 3)
        assert rgb.dtype == np.uint8


# ----------------------------------------------------------------------------------------------------------------------
# Codec selection
# ----------------------------------------------------------------------------------------------------------------------


class TestDepthCodecSelection:
    """Depth must never reach a lossy codec, whatever the dataset's store option asks for."""

    def _camera_codec_for(self, tmp_path: Path, store_option: str) -> str:
        writer = ArrowLogWriter(
            LogWriterConfig(camera_store_option=store_option),
            logs_root=tmp_path,
            sensors_root=tmp_path,
            sync_config=SyncConfig(reference_column="camera_depth.pcam_f0.timestamp_us"),
        )
        writer.reset(make_log_metadata())
        modality_writer = writer._build_modality_writer(_make_depth_camera_metadata())
        return modality_writer._camera_codec  # type: ignore[attr-defined]

    def test_path_store_option_stores_depth_by_path(self, tmp_path: Path):
        assert self._camera_codec_for(tmp_path, "path") == "path"

    @pytest.mark.parametrize("store_option", ["jpeg_binary", "png_binary", "mp4"])
    def test_inline_store_options_fall_back_to_label_png(self, tmp_path: Path, store_option: str):
        # jpeg_binary and mp4 are lossy; png_binary applies an RGB colour conversion. All three would
        # corrupt depth. Only the lossless single-channel integer path is acceptable.
        assert self._camera_codec_for(tmp_path, store_option) == "label_png"


# ----------------------------------------------------------------------------------------------------------------------
# Storage roundtrip
# ----------------------------------------------------------------------------------------------------------------------


class TestDepthCameraRoundtrip:
    """The stored integer raster must survive the Arrow + PNG storage layer bit-exactly."""

    def _write_and_read(self, log_dir: Path, camera: Camera, scale: int | None = None) -> Camera:
        metadata = camera.metadata
        writer = ArrowCameraWriter(log_dir=log_dir, metadata=metadata, camera_codec="label_png")
        writer.write_modality(camera)
        writer.close()
        table = pa.ipc.open_file(str(log_dir / f"{metadata.modality_key}.arrow")).read_all()
        result = ArrowCameraReader.read_at_index(0, table, metadata, "test-dataset", scale=scale)
        assert result is not None
        return result

    def _make_depth_camera(self, depth_bits: int, max_depth: float) -> Camera:
        metadata = _make_depth_camera_metadata(max_depth=max_depth, depth_bits=depth_bits)
        return Camera(
            metadata=metadata,
            image=metadata.encode_depth(_make_metric_depth()),
            camera_to_global_se3=PoseSE3.identity(),
            timestamp=Timestamp.from_us(1000),
        )

    @pytest.mark.parametrize(("depth_bits", "max_depth", "dtype"), [(8, 50.0, np.uint8), (16, 96.0, np.uint16)])
    def test_depth_raster_lossless(self, tmp_path: Path, depth_bits: int, max_depth: float, dtype: type):
        # 8-bit rasters must land in an 8-bit PNG and 16-bit rasters in a 16-bit PNG, both bit-exact.
        camera = self._make_depth_camera(depth_bits, max_depth)
        result = self._write_and_read(tmp_path, camera)

        assert result.image.dtype == dtype
        np.testing.assert_array_equal(result.image, camera.image)

    @pytest.mark.parametrize(("depth_bits", "max_depth"), [(8, 50.0), (16, 96.0)])
    def test_metric_depth_survives_storage(self, tmp_path: Path, depth_bits: int, max_depth: float):
        # End to end: metres in, metres out, within the quantizer's error bound.
        metadata = _make_depth_camera_metadata(max_depth=max_depth, depth_bits=depth_bits)
        depth_m = _make_metric_depth()
        camera = Camera(
            metadata=metadata,
            image=metadata.encode_depth(depth_m),
            camera_to_global_se3=PoseSE3.identity(),
            timestamp=Timestamp.from_us(1000),
        )
        decoded = metadata.decode_depth(self._write_and_read(tmp_path, camera).image)

        clipped = np.clip(depth_m, 0.0, max_depth)
        tolerance = metadata.depth_resolution / 2.0 + np.spacing(np.float32(max_depth))
        assert np.abs(decoded - clipped).max() <= tolerance

    @pytest.mark.parametrize("depth_bits", [8, 16])
    def test_downscale_uses_nearest_neighbour(self, tmp_path: Path, depth_bits: int):
        # Bilinear/area resampling would average depth across occlusion boundaries, inventing surfaces
        # that float between the foreground and the background. Every output value must be an input value.
        camera = self._make_depth_camera(depth_bits, max_depth=96.0)
        result = self._write_and_read(tmp_path, camera, scale=2)

        assert result.image.shape == (120, 160)
        assert result.image.dtype == camera.image.dtype
        np.testing.assert_array_equal(result.image, camera.image[::2, ::2])

    def test_depth_and_rgb_coexist_in_separate_files(self, tmp_path: Path):
        # A depth stream shares its camera_id with the RGB camera it is pixel-aligned to; the two must
        # land in distinct Arrow files rather than clobbering one another.
        depth_metadata = _make_depth_camera_metadata()
        rgb_metadata = _make_rgb_camera_metadata()
        assert depth_metadata.camera_id == rgb_metadata.camera_id

        writer = ArrowLogWriter(
            LogWriterConfig(camera_store_option="png_binary"),
            logs_root=tmp_path,
            sensors_root=tmp_path,
            sync_config=SyncConfig(reference_column="camera.pcam_f0.timestamp_us"),
        )
        writer.reset(make_log_metadata())
        timestamp = Timestamp.from_us(1000)
        writer.write_async(
            Camera(
                metadata=rgb_metadata,
                image=np.zeros((240, 320, 3), dtype=np.uint8),
                camera_to_global_se3=PoseSE3.identity(),
                timestamp=timestamp,
            )
        )
        writer.write_async(
            Camera(
                metadata=depth_metadata,
                image=depth_metadata.encode_depth(_make_metric_depth()),
                camera_to_global_se3=PoseSE3.identity(),
                timestamp=timestamp,
            )
        )
        writer.close()
        log_dir = tmp_path / make_log_metadata().split / make_log_metadata().log_name

        assert (log_dir / "camera.pcam_f0.arrow").exists()
        assert (log_dir / "camera_depth.pcam_f0.arrow").exists()


# ----------------------------------------------------------------------------------------------------------------------
# SceneAPI surface
# ----------------------------------------------------------------------------------------------------------------------


class TestDepthSceneAPI:
    """A written depth stream must be discoverable and readable through the public SceneAPI."""

    NUM_ITERATIONS = 3

    def _write_depth_log(self, tmp_path: Path, depth_bits: int, max_depth: float) -> tuple[Path, np.ndarray]:
        metadata = _make_depth_camera_metadata(max_depth=max_depth, depth_bits=depth_bits)
        writer = ArrowLogWriter(
            LogWriterConfig(camera_store_option="png_binary"),
            logs_root=tmp_path,
            sensors_root=tmp_path,
            sync_config=SyncConfig(reference_column="camera_depth.pcam_f0.timestamp_us"),
        )
        writer.reset(make_log_metadata())
        depth_m = _make_metric_depth()
        for iteration in range(self.NUM_ITERATIONS):
            writer.write_async(
                Camera(
                    metadata=metadata,
                    image=metadata.encode_depth(depth_m),
                    camera_to_global_se3=PoseSE3.identity(),
                    timestamp=Timestamp.from_us(100_000 * (iteration + 1)),
                )
            )
        writer.close()
        log_dir = tmp_path / make_log_metadata().split / make_log_metadata().log_name
        return log_dir, depth_m

    def _scene(self, log_dir: Path) -> ArrowSceneAPI:
        return ArrowSceneAPI(
            log_dir,
            SceneMetadata(
                dataset="test-dataset",
                split="test-dataset_train",
                initial_uuid="00000000-0000-0000-0000-000000000001",
                initial_idx=0,
                num_future_iterations=self.NUM_ITERATIONS - 1,
                num_history_iterations=0,
                future_duration_s=0.2,
                history_duration_s=0.0,
                iteration_duration_s=0.1,
            ),
        )

    @pytest.mark.parametrize(("depth_bits", "max_depth"), [(8, 50.0), (16, 96.0)])
    def test_scene_api_exposes_depth(self, tmp_path: Path, depth_bits: int, max_depth: float):
        log_dir, depth_m = self._write_depth_log(tmp_path, depth_bits, max_depth)
        scene = self._scene(log_dir)

        assert scene.available_camera_depth_ids == [CameraID.PCAM_F0]
        assert scene.available_camera_depth_names == ["front_camera"]

        metadata = scene.get_camera_depth_metadatas()[CameraID.PCAM_F0]
        assert isinstance(metadata, DepthCameraMetadata)
        # The quantization contract must survive the msgpack round-trip through the Arrow schema metadata.
        assert (metadata.max_depth, metadata.depth_bits) == (max_depth, depth_bits)

        assert len(scene.get_all_camera_depth_timestamps(CameraID.PCAM_F0)) == self.NUM_ITERATIONS

        camera = scene.get_camera_depth_at_iteration(0, camera_id=CameraID.PCAM_F0)
        assert camera is not None
        # `image` is the raw quantized raster; metric depth comes back through decode_depth().
        assert camera.image.dtype == metadata.depth_dtype
        decoded = metadata.decode_depth(camera.image)
        tolerance = metadata.depth_resolution / 2.0 + np.spacing(np.float32(max_depth))
        assert np.abs(decoded - np.clip(depth_m, 0.0, max_depth)).max() <= tolerance

    def test_scene_api_accepts_lowercase_camera_name(self, tmp_path: Path):
        log_dir, _ = self._write_depth_log(tmp_path, depth_bits=16, max_depth=96.0)
        scene = self._scene(log_dir)
        assert scene.get_camera_depth_at_iteration(0, camera_id="pcam_f0") is not None

    def test_scene_api_reads_depth_at_timestamp(self, tmp_path: Path):
        log_dir, _ = self._write_depth_log(tmp_path, depth_bits=8, max_depth=50.0)
        scene = self._scene(log_dir)
        timestamps = scene.get_all_camera_depth_timestamps(CameraID.PCAM_F0)

        assert scene.get_camera_depth_at_timestamp(timestamps[1], camera_id=CameraID.PCAM_F0) is not None
        assert scene.get_camera_depth_at_timestamp(0, camera_id=CameraID.PCAM_F0, criteria="exact") is None
        assert scene.get_camera_depth_at_timestamp(0, camera_id=CameraID.PCAM_F0, criteria="nearest") is not None

    def test_scene_without_depth_reports_none(self, tmp_path: Path):
        # An RGB-only log must not grow a phantom depth stream.
        writer = ArrowLogWriter(
            LogWriterConfig(camera_store_option="png_binary"),
            logs_root=tmp_path,
            sensors_root=tmp_path,
            sync_config=SyncConfig(reference_column="camera.pcam_f0.timestamp_us"),
        )
        writer.reset(make_log_metadata())
        for iteration in range(self.NUM_ITERATIONS):
            writer.write_async(
                Camera(
                    metadata=_make_rgb_camera_metadata(),
                    image=np.zeros((240, 320, 3), dtype=np.uint8),
                    camera_to_global_se3=PoseSE3.identity(),
                    timestamp=Timestamp.from_us(100_000 * (iteration + 1)),
                )
            )
        writer.close()
        log_dir = tmp_path / make_log_metadata().split / make_log_metadata().log_name

        scene = self._scene(log_dir)
        assert scene.available_camera_depth_ids == []
        assert scene.get_camera_depth_metadatas() == {}
        assert scene.get_camera_depth_at_iteration(0, camera_id=CameraID.PCAM_F0) is None
