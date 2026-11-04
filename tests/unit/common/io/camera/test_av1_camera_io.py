"""Tests of the AV1 camera video: writing, random and sequential reads, threads, processes and the GPU paths.

The CPU tests encode with SVT-AV1 at small sizes. Tests marked ``gpu`` use NVENC and the NVIDIA decoder; they are
deselected by default (run them with ``pytest -m gpu`` on a machine with an AV1-capable NVIDIA GPU).
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from py123d.common.io.camera import av1_camera_io
from py123d.common.io.camera.av1_camera_io import (
    AV1Reader,
    AV1Writer,
    get_av1_reader,
    nvenc_av1_available,
    read_av1_frame,
    resolve_av1_encoder,
)
from py123d.common.io.camera.video_codec_config import VideoCodecConfig
from py123d.common.io.camera.yuv_frame import YUV420Frame

SVT = VideoCodecConfig(backend="svt", svt_preset=12, svt_crf=30)
HEIGHT, WIDTH, FRAMES = 96, 128, 25


def _frames(n=FRAMES, full_range=True, matrix="bt601"):
    """Images that differ clearly from each other: a gray level per image on the left, texture on the right."""
    rng = np.random.default_rng(0)
    texture = rng.integers(0, 256, (HEIGHT, WIDTH // 2, 3)).astype(np.uint8)
    out = []
    for i in range(n):
        rgb = np.empty((HEIGHT, WIDTH, 3), np.uint8)
        rgb[:, : WIDTH // 2] = (i * 37) % 200 + 20
        rgb[:, WIDTH // 2 :] = np.roll(texture, i * 3, axis=1)
        out.append(YUV420Frame.from_rgb(rgb, full_range=full_range, matrix=matrix))
    return out


def _write(path, frames, config=SVT):
    with AV1Writer(path, config) as writer:
        indices = [writer.write_frame(frame) for frame in frames]
    assert indices == list(range(len(frames)))
    return path


def _psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(255.0**2 / max(mse, 1e-12))


def _identify(decoded, frames):
    """The index of the source image closest to a decoded image."""
    return int(np.argmin([np.mean((decoded.y.astype(float) - f.y.astype(float)) ** 2) for f in frames]))


@pytest.fixture(scope="module")
def svt_video(tmp_path_factory):
    frames = _frames()
    path = _write(tmp_path_factory.mktemp("av1") / "camera.pcam_f0.av1.mp4", frames)
    return path, frames


class TestWriter:
    def test_file_tags_and_count(self, svt_video):
        path, frames = svt_video
        reader = AV1Reader(path)
        assert reader.frame_count == FRAMES and (reader.width, reader.height) == (WIDTH, HEIGHT)
        assert reader.full_range and reader.matrix == "bt601"

    def test_limited_bt709_tags(self, tmp_path):
        frames = _frames(5, full_range=False, matrix="bt709")
        reader = AV1Reader(_write(tmp_path / "v.av1.mp4", frames))
        assert not reader.full_range and reader.matrix == "bt709"
        assert _psnr(reader.get_yuv(2).y, frames[2].y) > 30

    def test_rgb_input_and_quality(self, tmp_path):
        frames = _frames(5)
        with AV1Writer(tmp_path / "v.av1.mp4", SVT) as writer:
            for frame in frames:
                writer.write_frame(frame.to_rgb())
        reader = AV1Reader(tmp_path / "v.av1.mp4")
        assert all(_psnr(reader.get_yuv(i).y, frames[i].y) > 30 for i in range(5))

    def test_size_or_tag_change_raises_and_close_is_idempotent(self, tmp_path):
        writer = AV1Writer(tmp_path / "v.av1.mp4", SVT)
        writer.write_frame(_frames(1)[0])
        with pytest.raises(ValueError):
            writer.write_frame(YUV420Frame.from_rgb(np.zeros((64, 64, 3), np.uint8)))
        with pytest.raises(ValueError):
            writer.write_frame(_frames(1, full_range=False)[0])
        writer.close()
        writer.close()
        with pytest.raises(RuntimeError):
            writer.write_frame(_frames(1)[0])
        assert AV1Reader(tmp_path / "v.av1.mp4").frame_count == 1

    def test_close_without_frames_writes_nothing(self, tmp_path):
        writer = AV1Writer(tmp_path / "v.av1.mp4", SVT)
        writer.close()
        assert not (tmp_path / "v.av1.mp4").exists()

    def test_backend_resolution(self, monkeypatch):
        assert resolve_av1_encoder(VideoCodecConfig(backend="svt")) == "libsvtav1"
        monkeypatch.setattr(av1_camera_io, "nvenc_av1_available", lambda gpu=-1: False)
        with pytest.raises(RuntimeError, match="NVENC"):
            resolve_av1_encoder(VideoCodecConfig(backend="nvenc"))
        assert resolve_av1_encoder(VideoCodecConfig(backend="auto")) == "libsvtav1"
        monkeypatch.setattr(av1_camera_io, "nvenc_av1_available", lambda gpu=-1: True)
        assert resolve_av1_encoder(VideoCodecConfig(backend="auto")) == "av1_nvenc"

    def test_config_validation_and_options(self):
        with pytest.raises(AssertionError):
            VideoCodecConfig(backend="x264")
        with pytest.raises(AssertionError):
            VideoCodecConfig(nvenc_cq=64)
        options = VideoCodecConfig().nvenc_options()
        assert options["cq"] == "38" and options["g"] == "10" and options["maxrate"] == "200M"
        assert options["bf"] == "0"  # no B-frames (kedec default since 3 November 2026)
        assert VideoCodecConfig(nvenc_b_frames=3).nvenc_options()["bf"] == "3"
        assert VideoCodecConfig(svt_threads=4).svt_options()["svtav1-params"] == "lp=4"


class TestBackgroundEncoding:
    def test_threaded_and_unthreaded_videos_decode_the_same(self, tmp_path):
        frames = _frames()
        for threaded in (True, False):
            with AV1Writer(tmp_path / f"v{int(threaded)}.av1.mp4", SVT, threaded=threaded, queue_size=2) as writer:
                assert [writer.write_frame(f) for f in frames] == list(range(len(frames)))
        a, b = AV1Reader(tmp_path / "v1.av1.mp4"), AV1Reader(tmp_path / "v0.av1.mp4")
        assert a.frame_count == b.frame_count == len(frames)
        for i in range(len(frames)):
            np.testing.assert_array_equal(a.get_yuv(i).y, b.get_yuv(i).y)
            assert _identify(a.get_yuv(i), frames) == i

    def test_an_encoder_error_is_raised_and_the_file_closed(self, tmp_path, monkeypatch):
        writer = AV1Writer(tmp_path / "e.av1.mp4", SVT, queue_size=1)
        original = AV1Writer._encode

        def failing(self, index, yuv):
            if index == 3:
                raise ValueError("encoder failure")
            original(self, index, yuv)

        monkeypatch.setattr(AV1Writer, "_encode", failing)
        with pytest.raises(RuntimeError, match="encoder failure"):  # raised by a later write_frame
            for frame in _frames(40):
                writer.write_frame(frame)
        with pytest.raises(RuntimeError, match="encoder failure"):  # and by close, which releases the encoder
            writer.close()
        writer.close()  # no effect when already closed
        assert writer._container is None and writer._thread is None


class TestReader:
    def test_random_reads_equal_sequential_reads(self, svt_video):
        path, frames = svt_video
        sequential = AV1Reader(path)
        reference = [sequential.get_yuv(i) for i in range(FRAMES)]
        assert [_identify(f, frames) for f in reference] == list(range(FRAMES))
        random_reader = AV1Reader(path)
        order = np.random.default_rng(1).permutation(FRAMES).tolist() + [3, 3, 24, 0, 11, 10, 9]
        for index in order:
            np.testing.assert_array_equal(random_reader.get_yuv(index).y, reference[index].y)

    def test_out_of_range_raises(self, svt_video):
        reader = AV1Reader(svt_video[0])
        with pytest.raises(IndexError):
            reader.get_yuv(FRAMES)
        with pytest.raises(IndexError):
            reader.get_yuv(-1)
        with pytest.raises(FileNotFoundError):
            AV1Reader(svt_video[0].parent / "missing.av1.mp4")

    def test_rgb_and_scale(self, svt_video):
        path, frames = svt_video
        reader = AV1Reader(path)
        rgb = reader.get_frame(4)
        assert rgb.shape == (HEIGHT, WIDTH, 3) and rgb.dtype == np.uint8
        assert _psnr(rgb, frames[4].to_rgb()) > 28
        assert reader.get_frame(4, scale=2).shape == (HEIGHT // 2, WIDTH // 2, 3)

    def test_threads_share_a_cached_reader(self, svt_video):
        path, _ = svt_video
        reference = [AV1Reader(path).get_frame(i) for i in range(FRAMES)]
        indices = np.random.default_rng(2).integers(0, FRAMES, 200).tolist()
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda i: read_av1_frame(path, i), indices))
        for index, image in zip(indices, results):
            np.testing.assert_array_equal(image, reference[index])
        assert get_av1_reader(path) is get_av1_reader(path)

    def test_forked_process_opens_its_own_reader(self, svt_video):
        path, _ = svt_video
        expected = read_av1_frame(path, 7)  # the parent process has a cached reader
        context = multiprocessing.get_context("fork")
        with context.Pool(2) as pool:
            results = pool.starmap(read_av1_frame, [(path, 7), (path, 12)])
        np.testing.assert_array_equal(results[0], expected)
        np.testing.assert_array_equal(results[1], read_av1_frame(path, 12))


@pytest.mark.gpu
class TestGpu:
    @pytest.fixture(autouse=True)
    def _require_nvenc(self):
        if not nvenc_av1_available():
            pytest.skip("no NVENC AV1 encoder")

    @staticmethod
    def _large(frames):
        """The images at twice the size: NVENC encodes AV1 from 144 x 144 pixels up."""
        return [YUV420Frame.from_rgb(np.repeat(np.repeat(f.to_rgb(), 2, 0), 2, 1)) for f in frames]

    def test_nvenc_roundtrip(self, tmp_path):
        frames = self._large(_frames(30))
        path = _write(tmp_path / "v.av1.mp4", frames, VideoCodecConfig())
        reader = AV1Reader(path)
        assert reader.frame_count == 30
        assert all(_identify(reader.get_yuv(i), frames) == i for i in range(30))

    def test_cuvid_decodes_the_same_planes(self, tmp_path):
        frames = self._large(_frames(20))
        path = _write(tmp_path / "v.av1.mp4", frames, VideoCodecConfig())
        cpu, gpu = AV1Reader(path), AV1Reader(path, decoder="av1_cuvid")
        for index in [0, 1, 2, 15, 3, 19, 10]:
            a, b = cpu.get_yuv(index), gpu.get_yuv(index)
            for name in ("y", "u", "v"):
                np.testing.assert_array_equal(getattr(a, name), getattr(b, name))

    def test_many_writers_release_their_sessions(self, tmp_path):
        """Opening and closing many NVENC writers in a row does not run out of encoder sessions."""
        frame = self._large(_frames(1))[0]
        for i in range(40):
            writer = AV1Writer(tmp_path / f"v{i}.av1.mp4", VideoCodecConfig())
            writer.write_frame(frame)
            writer.close()
        assert Path(tmp_path / "v39.av1.mp4").exists()
