import io
import tarfile
import types
from pathlib import Path

import pytest

from py123d.parser.kitscenes import kitscenes_download
from py123d.parser.kitscenes.kitscenes_download import KITScenesDownloader

SCENES = {"scene-a": "val", "scene-b": "val", "scene-c": "train"}


@pytest.fixture
def fake_hf_hub(tmp_path: Path, monkeypatch):
    """Serve a scene index and one tiny archive per scene from a local folder, like ``hf_hub_download``."""
    remote = tmp_path / "remote"
    (remote / "data").mkdir(parents=True)
    rows = ["sequence_id,split,archive_path,archive_sha256,archive_size_bytes"]
    for scene_id, split in SCENES.items():
        archive_path = remote / "data" / split / f"{scene_id}.tar"
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "w") as tar:
            content = b"0.0 0 0 0 0 0 0 1\n"
            info = tarfile.TarInfo(f"{scene_id}/poses.txt")
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
        rows.append(f"{scene_id},{split},data/{split}/{scene_id}.tar,sha,{archive_path.stat().st_size}")
    (remote / "data" / "sequence_archives.csv").write_text("\n".join(rows))

    downloaded = []

    def hf_hub_download(repo_id, repo_type, filename, revision, token, local_dir=None):
        downloaded.append(filename)
        if local_dir is None:
            return str(remote / filename)
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((remote / filename).read_bytes())
        return str(target)

    fake_module = types.SimpleNamespace(hf_hub_download=hf_hub_download)
    monkeypatch.setattr(kitscenes_download, "_require_hf_hub", lambda: fake_module)
    return downloaded


class TestKITScenesDownloader:
    def test_download_and_extract_selected_split(self, tmp_path: Path, fake_hf_hub):
        output_dir = tmp_path / "kitscenes"
        KITScenesDownloader(output_dir, splits=["val"], delete_archives=True).download()
        assert (output_dir / "data" / "val" / "scene-a" / "poses.txt").is_file()
        assert (output_dir / "data" / "val" / "scene-b" / "poses.txt").is_file()
        assert not (output_dir / "data" / "train").exists()
        assert not (output_dir / "data" / "val" / "scene-a.tar").exists()

    def test_scene_ids_and_max_num_scenes(self, tmp_path: Path, fake_hf_hub):
        KITScenesDownloader(tmp_path, scene_ids=["scene-c"]).download()
        assert (tmp_path / "data" / "train" / "scene-c").is_dir()
        KITScenesDownloader(tmp_path / "capped", max_num_scenes=1).download()
        assert len(list((tmp_path / "capped" / "data").glob("*/*/poses.txt"))) == 1

    def test_dry_run_downloads_only_the_index(self, tmp_path: Path, fake_hf_hub):
        KITScenesDownloader(tmp_path, dry_run=True).download()
        assert fake_hf_hub == ["data/sequence_archives.csv"]

    def test_extracted_scenes_are_skipped(self, tmp_path: Path, fake_hf_hub):
        KITScenesDownloader(tmp_path, scene_ids=["scene-a"]).download()
        fake_hf_hub.clear()
        KITScenesDownloader(tmp_path, scene_ids=["scene-a"]).download()
        assert fake_hf_hub == ["data/sequence_archives.csv"]

    def test_unknown_split_is_rejected(self, tmp_path: Path):
        with pytest.raises(AssertionError):
            KITScenesDownloader(tmp_path, splits=["training"])
