"""Tests for LogWriterConfig validation."""

from __future__ import annotations

import pytest

from py123d.api.scene.arrow.utils.log_writer_config import LogWriterConfig


class TestLogWriterConfig:
    def test_defaults_valid(self):
        config = LogWriterConfig()
        assert config.camera_store_option == "path"
        assert config.lidar_store_option == "path"

    def test_invalid_camera_option(self):
        with pytest.raises(AssertionError, match="Invalid camera store option"):
            LogWriterConfig(camera_store_option="bmp")

    def test_invalid_lidar_option(self):
        with pytest.raises(AssertionError, match="Invalid Lidar store option"):
            LogWriterConfig(lidar_store_option="numpy")

    def test_binary_lidar_requires_codec(self):
        with pytest.raises(AssertionError, match="Invalid Lidar codec"):
            LogWriterConfig(lidar_store_option="binary", lidar_codec=None)

    def test_binary_lidar_with_valid_codecs(self):
        for codec in ["laz", "draco", "ipc_zstd", "ipc_lz4", "ipc"]:
            config = LogWriterConfig(lidar_store_option="binary", lidar_codec=codec)
            assert config.lidar_codec == codec

    def test_path_lidar_ignores_codec(self):
        config = LogWriterConfig(lidar_store_option="path", lidar_codec=None)
        assert config.lidar_codec is None

    def test_valid_camera_options(self):
        for opt in ["path", "jpeg_binary", "png_binary", "mp4"]:
            config = LogWriterConfig(camera_store_option=opt)
            assert config.camera_store_option == opt
