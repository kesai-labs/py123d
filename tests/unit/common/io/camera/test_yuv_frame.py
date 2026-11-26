"""Tests of YUV420Frame: plane layout, color conversions and the JPEG planes."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from py123d.common.io.camera.yuv_frame import YUV420Frame


def _smooth_rgb(height=96, width=128, seed=0):
    """A smooth color image (4:2:0 subsampling then changes it by about one level)."""
    small = np.random.default_rng(seed).integers(0, 256, (height // 16, width // 16, 3)).astype(np.uint8)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)


def _jfif_ycbcr(rgb):
    """The JFIF (full-range BT.601) YCbCr of RGB, without subsampling, in float."""
    r, g, b = (rgb[..., i].astype(np.float64) for i in range(3))
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return y, cb, cr


class TestValidation:
    def test_odd_or_wrong_planes_raise(self):
        with pytest.raises(ValueError):
            YUV420Frame(np.zeros((3, 4), np.uint8), np.zeros((1, 2), np.uint8), np.zeros((1, 2), np.uint8))
        with pytest.raises(ValueError):
            YUV420Frame(np.zeros((4, 4), np.uint8), np.zeros((2, 3), np.uint8), np.zeros((2, 2), np.uint8))
        with pytest.raises(ValueError):
            YUV420Frame(np.zeros((4, 4), np.uint16), np.zeros((2, 2), np.uint8), np.zeros((2, 2), np.uint8))
        with pytest.raises(ValueError):
            YUV420Frame(
                np.zeros((4, 4), np.uint8), np.zeros((2, 2), np.uint8), np.zeros((2, 2), np.uint8), matrix="xyz"
            )
        with pytest.raises(ValueError):
            YUV420Frame.from_rgb(np.zeros((5, 4, 3), np.uint8))

    def test_i420_layout_roundtrip(self):
        frame = YUV420Frame.from_rgb(_smooth_rgb())
        i420 = frame.to_i420()
        assert i420.shape == (96 * 3 // 2, 128)
        back = YUV420Frame.from_i420(i420)
        for name in ("y", "u", "v"):
            np.testing.assert_array_equal(getattr(back, name), getattr(frame, name))


class TestConversions:
    def test_jfif_matrix_within_one_level(self):
        """from_rgb follows JFIF: luma within 0.6 levels, chroma means within one level."""
        rgb = _smooth_rgb(seed=1)
        frame = YUV420Frame.from_rgb(rgb)
        y, cb, cr = _jfif_ycbcr(rgb)
        assert np.abs(frame.y.astype(np.float64) - y).max() <= 0.6  # float32 coefficients: rounding to 0.55
        cb_half = cv2.resize(cb, (64, 48), interpolation=cv2.INTER_AREA)
        cr_half = cv2.resize(cr, (64, 48), interpolation=cv2.INTER_AREA)
        assert np.abs(frame.u - cb_half).max() <= 1.0 and np.abs(frame.v - cr_half).max() <= 1.0

    def test_gray_has_neutral_chroma(self):
        gray = np.repeat(np.tile(np.arange(128, dtype=np.uint8) * 2, (96, 1))[..., None], 3, axis=2)
        frame = YUV420Frame.from_rgb(gray)
        np.testing.assert_array_equal(frame.y, gray[..., 0])
        assert np.all(np.abs(frame.u.astype(int) - 128) <= 1) and np.all(np.abs(frame.v.astype(int) - 128) <= 1)

    def test_full_range_ramp(self):
        """A full-range gray ramp 0..255 keeps every level through to_rgb."""
        ramp = np.repeat(np.arange(256, dtype=np.uint8)[None, :], 4, axis=0)
        frame = YUV420Frame(ramp, np.full((2, 128), 128, np.uint8), np.full((2, 128), 128, np.uint8))
        rgb = frame.to_rgb()
        np.testing.assert_array_equal(rgb[..., 0], ramp)
        np.testing.assert_array_equal(rgb[..., 1], ramp)

    def test_limited_range_bt709(self):
        """Limited range: luma 16 is black and 235 is white; BT.709 red maps to its own luma value."""
        black = YUV420Frame(
            np.full((2, 2), 16, np.uint8),
            np.full((1, 1), 128, np.uint8),
            np.full((1, 1), 128, np.uint8),
            False,
            "bt709",
        )
        white = YUV420Frame(
            np.full((2, 2), 235, np.uint8),
            np.full((1, 1), 128, np.uint8),
            np.full((1, 1), 128, np.uint8),
            False,
            "bt709",
        )
        assert black.to_rgb().max() == 0 and white.to_rgb().min() == 255
        red = np.zeros((4, 4, 3), np.uint8)
        red[..., 0] = 255
        frame = YUV420Frame.from_rgb(red, full_range=False, matrix="bt709")
        assert abs(int(frame.y[0, 0]) - round(16 + 219 * 0.2126)) <= 1
        assert np.abs(frame.to_rgb().astype(int) - red).max() <= 2

    @pytest.mark.parametrize(
        "full_range, matrix", [(True, "bt601"), (False, "bt601"), (True, "bt709"), (False, "bt709")]
    )
    def test_rgb_roundtrip(self, full_range, matrix):
        rgb = _smooth_rgb(seed=2)
        back = YUV420Frame.from_rgb(rgb, full_range=full_range, matrix=matrix).to_rgb()
        difference = np.abs(back.astype(int) - rgb.astype(int))
        assert difference.mean() < 1.5 and difference.max() <= 12


class TestJpeg:
    def _jpeg(self, rgb, sampling):
        params = [cv2.IMWRITE_JPEG_QUALITY, 90, cv2.IMWRITE_JPEG_SAMPLING_FACTOR, sampling]
        ok, data = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), params)
        assert ok
        return data.tobytes()

    def test_420_jpeg_keeps_its_planes(self):
        """The planes of a 4:2:0 JPEG are FFmpeg's decoded planes, unchanged."""
        import av

        jpeg = self._jpeg(_smooth_rgb(seed=3), cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420)
        frame = YUV420Frame.from_jpeg(jpeg)
        decoded = av.CodecContext.create("mjpeg", "r").decode(av.Packet(jpeg))[0]
        assert decoded.format.name == "yuvj420p"
        np.testing.assert_array_equal(frame.to_i420(), decoded.to_ndarray())
        assert frame.full_range and frame.matrix == "bt601"

    def test_jpeg_colors_match_opencv(self):
        rgb = _smooth_rgb(seed=4)
        jpeg = self._jpeg(rgb, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420)
        reference = cv2.cvtColor(cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        difference = np.abs(YUV420Frame.from_jpeg(jpeg).to_rgb().astype(int) - reference.astype(int))
        assert difference.mean() < 1.0 and difference.max() <= 8

    def test_444_jpeg_is_converted(self):
        rgb = _smooth_rgb(seed=5)
        jpeg = self._jpeg(rgb, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444)
        frame = YUV420Frame.from_jpeg(jpeg)
        assert (frame.height, frame.width) == (96, 128)
        assert np.abs(frame.to_rgb().astype(int) - rgb.astype(int)).mean() < 2.0

    def test_invalid_jpeg_raises(self):
        with pytest.raises(Exception):
            YUV420Frame.from_jpeg(b"\xff\xd8not a jpeg\xff\xd9")
