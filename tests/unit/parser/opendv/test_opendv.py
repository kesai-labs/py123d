"""The OpenDV parser: grid of kept images, nearest-frame selection, video list and an end-to-end parse of a small
VP9 video (as YouTube delivers it)."""

from __future__ import annotations

import json
from fractions import Fraction

import av
import numpy as np
import pytest

from py123d.datatypes.sensors.base_camera import CameraID
from py123d.parser.base_dataset_parser import ParsedCamera
from py123d.parser.opendv.opendv_parser import (
    OpenDVParser,
    grid_frame_count,
    grid_times_s,
    select_grid_frames,
)


def _select(times, grid, max_offset_s=0.05):
    return [(k, t) for k, t, _ in select_grid_frames(((t, i) for i, t in enumerate(times)), grid, max_offset_s)]


def test_grid_frame_count_as_the_toolkit():
    # int(10 * (total_frames // fps - start - end)); total_frames // fps is the duration rounded down.
    assert grid_frame_count(20.9, 2, 1) == 170
    assert grid_frame_count(20.0, 2, 1) == 170
    assert grid_frame_count(3.0, 2, 1) == 0
    assert grid_frame_count(2.0, 2, 1) == 0  # never negative
    np.testing.assert_allclose(grid_times_s(90, 3), [90.0, 90.1, 90.2])


@pytest.mark.parametrize("fps", [59.94, 60.0, 30.0, 29.97, 25.0, 24.0])
def test_nearest_frame_per_grid_time(fps):
    times = np.arange(int(12 * fps)) / fps
    grid = grid_times_s(1.0, 100)
    selected = _select(times, grid)
    assert [k for k, _ in selected] == list(range(100))
    for k, t in selected:
        nearest = times[np.argmin(np.abs(times - grid[k]))]
        assert t == nearest and abs(t - grid[k]) <= 0.5 / fps + 1e-12
    assert len({t for _, t in selected}) == 100


def test_a_tie_goes_to_the_earlier_frame():
    assert _select([0.97, 1.03], [1.0]) == [(0, 0.97)]


def test_grid_times_in_a_gap_have_no_frame():
    times = [t for t in np.arange(0, 4, 1 / 30) if not 2.0 <= t < 2.5]
    selected = dict(_select(times, grid_times_s(1.0, 25)))
    skipped = [k for k in range(25) if k not in selected]
    assert skipped == [11, 12, 13, 14]  # 2.1 to 2.4 s lie more than 50 ms from a frame (2.0 s: 33 ms)
    for k, t in selected.items():
        assert abs(t - (1.0 + k / 10)) <= 0.05 + 1e-12


def test_each_frame_is_kept_at_most_once():
    times = np.arange(0, 2, 0.125)  # 8 frames per second: fewer frames than grid times
    selected = _select(times, grid_times_s(0.0, 20))
    kept = [t for _, t in selected]
    assert kept == sorted(set(kept))
    assert (2, 0.25) in selected and all(k != 3 for k, _ in selected)  # 0.3 s: its nearest frame 0.25 s is used


def test_the_last_frame_serves_grid_times_after_the_end():
    assert _select([0.0, 1.0], [0.98, 1.02, 1.2]) == [(0, 1.0)]


def _write_video(path, fps: int, seconds: float, size=(64, 48)):
    """A VP9 video whose frame j is gray level 2 * (j % 120)."""
    with av.open(str(path), "w") as out:
        stream = out.add_stream("libvpx-vp9", rate=fps, options={"crf": "10", "deadline": "realtime", "cpu-used": "8"})
        stream.width, stream.height, stream.pix_fmt = size[0], size[1], "yuv420p"
        stream.time_base = Fraction(1, fps)
        for j in range(int(fps * seconds)):
            image = np.full((size[1], size[0], 3), 2 * (j % 120), dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts = j
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)


@pytest.fixture
def data_root(tmp_path):
    videos = [
        {"videoid": "-a_Video01", "youtuber": "Some Channel", "split": "Train", "start_discard": 1, "end_discard": 1},
        {
            "videoid": "missing0001",
            "youtuber": "Some Channel",
            "split": "Train",
            "start_discard": 90,
            "end_discard": 30,
        },
        {"videoid": "valVideo002", "youtuber": "Other", "split": "Val", "start_discard": 90, "end_discard": 30},
    ]
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "OpenDV-YouTube.json").write_text(json.dumps(videos))
    folder = tmp_path / "videos" / "Some_Channel"  # the toolkit's folder name: spaces as "_"
    folder.mkdir(parents=True)
    _write_video(folder / "-a_Video01.webm", fps=60, seconds=6.5)
    return tmp_path


def test_parse_a_video(data_root):
    parser = OpenDVParser(splits=["opendv_train"], opendv_data_root=data_root)
    assert parser.get_map_parsers() == []
    log_parsers = parser.get_log_parsers()
    assert len(log_parsers) == 1  # the missing video and the validation video are not listed
    log_parser = log_parsers[0]
    metadata = log_parser.get_log_metadata()
    assert (metadata.dataset, metadata.split, metadata.log_name) == ("opendv", "opendv_train", "-a_Video01")
    with pytest.raises(NotImplementedError):
        next(log_parser.iter_modalities_sync())

    cameras = list(log_parser.iter_modalities_async())
    assert all(isinstance(c, ParsedCamera) for c in cameras)
    assert len(cameras) == grid_frame_count(6.5, 1, 1) == 40
    times_s = np.array([c.timestamp.time_us for c in cameras]) / 1e6
    np.testing.assert_allclose(times_s, grid_times_s(1.0, 40), atol=0.5 / 60 + 1e-6)
    camera_metadata = cameras[0].metadata
    assert camera_metadata.camera_id == CameraID.PCAM_F0 and camera_metadata.intrinsics is None
    assert (camera_metadata.width, camera_metadata.height) == (64, 48)
    assert cameras[0].camera_to_global_se3 is None
    for c, t in zip(cameras, times_s):
        j = int(round(t * 60))  # the source frame of the kept image
        y = c.frame.y.astype(np.float64).mean()
        expected_y = 16 + 219 * (2 * (j % 120)) / 255  # limited-range luma of the gray level
        assert abs(y - expected_y) < 3


def test_one_decoder_thread_gives_the_same_images(data_root):
    """decode_threads=1 decodes without frame threading (a fallback for videos that fail with it)."""
    threaded = OpenDVParser(splits=["opendv_train"], opendv_data_root=data_root).get_log_parsers()[0]
    single = OpenDVParser(splits=["opendv_train"], opendv_data_root=data_root, decode_threads=1).get_log_parsers()[0]
    a = list(threaded.iter_modalities_async())
    b = list(single.iter_modalities_async())
    assert [c.timestamp.time_us for c in a] == [c.timestamp.time_us for c in b]
    assert all(np.array_equal(x.frame.y, y.frame.y) for x, y in zip(a, b))


def test_video_selection(data_root):
    assert len(OpenDVParser(splits=["opendv_train", "opendv_val"], opendv_data_root=data_root).get_log_parsers()) == 1
    assert OpenDVParser(splits=["opendv_train"], opendv_data_root=data_root, video_ids=["x"]).get_log_parsers() == []
    with pytest.raises(AssertionError):
        OpenDVParser(splits=["opendv_test"], opendv_data_root=data_root)
