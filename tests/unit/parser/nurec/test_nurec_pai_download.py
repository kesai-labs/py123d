"""Fetching the Physical AI AV files of NuRec scenes out of the chunk archives, through HTTP range requests."""

import http.server
import shutil
import threading
import zipfile
from pathlib import Path
from typing import Iterator, List

import pandas as pd
import pytest

from py123d.parser.nurec.nurec_pai_download import NuRecPAIDownloader

RELEASE = "26.01"
SPLIT = "nurec-test_train"
CHUNK = 7
CLIP_IDS = ["00000000-0000-4000-8000-00000000000a", "00000000-0000-4000-8000-00000000000b"]
OTHER_CLIP_ID = "00000000-0000-4000-8000-00000000000c"  # in the archives, not a NuRec scene
NOT_A_CLIP_ID = "00000000-0000-4000-8000-00000000000d"  # a NuRec scene that is no Physical AI AV clip
CAMERA = "camera_front_wide_120fov"
SCENE_LISTS = {SPLIT: [f"{RELEASE}/{clip_id}" for clip_id in [*CLIP_IDS, NOT_A_CLIP_ID]]}


def member_content(member: str) -> bytes:
    """Content of an archive member, which tells the members apart."""
    return (member * 3000).encode()


class _RangeRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Serves files like the Hub does: whole, or the byte range a request asks for."""

    def log_message(self, format, *args):  # noqa: A002 - signature of the base class
        pass

    def do_GET(self):
        path = Path(self.translate_path(self.path))
        if not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        range_header = self.headers.get("Range")
        if range_header is None:
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        start, end = range_header.removeprefix("bytes=").split("-")
        first = int(start) if start else len(data) - int(end)
        last = min(int(end), len(data) - 1) if start and end else len(data) - 1
        chunk = data[first : last + 1]
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {first}-{last}/{len(data)}")
        self.send_header("Content-Length", str(len(chunk)))
        self.end_headers()
        self.wfile.write(chunk)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A directory laid out like the Hub repo: the clip index, the calibration, and one archive per sensor."""
    root = tmp_path / "repo"
    clip_index = pd.DataFrame(
        {"clip_is_valid": True, "chunk": CHUNK, "split": "train"},
        index=pd.Index([*CLIP_IDS, OTHER_CLIP_ID], name="clip_id"),
    )
    (root / "calibration").mkdir(parents=True)
    clip_index.to_parquet(root / "clip_index.parquet")
    for table in ("camera_intrinsics", "sensor_extrinsics", "vehicle_dimensions"):
        path = root / "calibration" / table / f"{table}.chunk_{CHUNK:04d}.parquet"
        path.parent.mkdir()
        pd.DataFrame({"value": [1.0]}, index=pd.Index(CLIP_IDS, name="clip_id")[:1]).to_parquet(path)
    for group, sensor, extensions in [
        ("camera", CAMERA, ["mp4", "timestamps.parquet"]),
        ("lidar", "lidar_top_360fov", ["parquet"]),
        ("labels", "egomotion", ["parquet"]),
    ]:
        archive_path = root / group / sensor / f"{sensor}.chunk_{CHUNK:04d}.zip"
        archive_path.parent.mkdir(parents=True)
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
            for clip_id in [*CLIP_IDS, OTHER_CLIP_ID]:
                for extension in extensions:
                    member = f"{clip_id}.{sensor}.{extension}"
                    archive.writestr(member, member_content(member))
    return root


@pytest.fixture
def server(repo: Path) -> Iterator[str]:
    """Serves the repo over HTTP, with range requests."""
    handler = lambda *args, **kwargs: _RangeRequestHandler(*args, directory=str(repo), **kwargs)  # noqa: E731
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()


@pytest.fixture
def downloader(tmp_path: Path, repo: Path, server: str, monkeypatch) -> NuRecPAIDownloader:
    """A downloader that reads the local repo instead of the Hub."""
    downloader = NuRecPAIDownloader(
        output_dir=tmp_path / "physical_ai_av",
        hf_token="unused",
        splits=[SPLIT],
        scene_lists=SCENE_LISTS,
        cameras=[CAMERA],
        max_workers=2,
    )

    def fetch_repo_file(filename: str) -> Path:
        destination = downloader.output_dir / filename
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(repo / filename, destination)
        return destination

    monkeypatch.setattr(downloader, "_fetch_repo_file", fetch_repo_file)
    monkeypatch.setattr(
        downloader,
        "_archive_url",
        lambda group, sensor, chunk: f"{server}/{group}/{sensor}/{sensor}.chunk_{chunk:04d}.zip",
    )
    return downloader


def _expected_files(output_dir: Path) -> List[Path]:
    files = []
    for clip_id in CLIP_IDS:
        files += [
            output_dir / "camera" / CAMERA / f"{clip_id}.{CAMERA}.mp4",
            output_dir / "camera" / CAMERA / f"{clip_id}.{CAMERA}.timestamps.parquet",
            output_dir / "lidar" / "lidar_top_360fov" / f"{clip_id}.lidar_top_360fov.parquet",
            output_dir / "labels" / "egomotion" / f"{clip_id}.egomotion.parquet",
        ]
    return files


def test_selected_clips_are_the_scenes_in_the_clip_index(downloader: NuRecPAIDownloader):
    assert downloader.selected_clips() == [(clip_id, CHUNK) for clip_id in CLIP_IDS]


def test_members_of_the_scenes_are_fetched(downloader: NuRecPAIDownloader):
    downloader.download()

    output_dir = downloader.output_dir
    for path in _expected_files(output_dir):
        assert path.read_bytes() == member_content(path.name)
    assert not list(output_dir.rglob(f"{OTHER_CLIP_ID}*"))
    assert not list(output_dir.rglob("*.part"))
    for table in ("camera_intrinsics", "sensor_extrinsics", "vehicle_dimensions"):
        assert (output_dir / "calibration" / table / f"{table}.chunk_{CHUNK:04d}.parquet").exists()


def test_complete_files_are_kept_and_partial_ones_replaced(downloader: NuRecPAIDownloader):
    downloader.download()
    output_dir = downloader.output_dir
    complete, partial = _expected_files(output_dir)[:2]
    complete_mtime = complete.stat().st_mtime_ns
    partial.write_bytes(partial.read_bytes()[:100])

    downloader.download()

    assert complete.stat().st_mtime_ns == complete_mtime
    assert partial.read_bytes() == member_content(partial.name)


def test_dry_run_fetches_nothing(downloader: NuRecPAIDownloader):
    downloader.dry_run = True

    downloader.download()

    assert not list(downloader.output_dir.rglob("*.mp4"))


def test_unknown_camera_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError):
        NuRecPAIDownloader(output_dir=tmp_path, hf_token="unused", cameras=["camera_top"])
