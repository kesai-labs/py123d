"""The codec=kedec preset, the GPU share of conversion tasks and the py123d-transcode command."""

from __future__ import annotations

import os
import subprocess
import sys

import pyarrow as pa
import pytest
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from py123d.common.execution.executor import ExecutorResources
from py123d.common.execution.ray_executor import RayExecutor, _local_storage_options
from py123d.common.execution.sequential_executor import SequentialExecutor
from py123d.common.execution.utils import gpu_share_per_task
from py123d.script.run_conversion import gpus_per_task, uses_gpu_video


def _compose(*overrides):
    with initialize_config_module(config_module="py123d.script.config.conversion", version_base=None):
        return compose(config_name="default_conversion", overrides=list(overrides))


@pytest.mark.parametrize(
    "dataset",
    ["nuplan-mini", "nuplan-tar", "physical-ai-av", "physical-ai-av-zip", "opendv", "nuscenes-mini", "pandaset"],
)
def test_without_codec_the_dataset_config_is_unchanged(dataset):
    """Without codec=, the composed dataset configuration equals the dataset file (no codec keys added)."""
    cfg = _compose(f"dataset={dataset}")
    assert "codec" not in cfg
    import importlib.resources

    path = importlib.resources.files("py123d.script.config.conversion.dataset") / f"{dataset}.yaml"
    raw = OmegaConf.load(str(path))
    assert OmegaConf.to_container(cfg.dataset, resolve=False) == OmegaConf.to_container(raw, resolve=False)


def test_kedec_preset():
    cfg = _compose("dataset=nuplan-mini", "codec=kedec")
    lwc = cfg.dataset.log_writer_config
    assert lwc.camera_store_option == "av1" and lwc.lidar_codec == "kedec" and lwc.lidar_store_option == "binary"
    assert lwc.radar_codec == "kedec" and lwc.radar_store_option == "binary"
    assert lwc.radar_codec_config.kedec_range_step_m == lwc.lidar_codec_config.kedec_range_step_m
    assert cfg.dataset.log_writer.sync_config.direction_overrides == {"camera": "nearest"}
    import hydra

    config = hydra.utils.instantiate(lwc)  # no GPU access: the encoder is resolved by the camera writer
    assert config.video_codec_config.nvenc_cq == 38 and config.lidar_codec_config.kedec_angle_step_deg == 0.008
    sync = hydra.utils.instantiate(cfg.dataset.log_writer.sync_config)
    assert sync.direction_of("camera.pcam_f0") == "nearest" and sync.direction_of("lidar.lidar_top") == "forward"


def test_kedec_preset_on_the_nuplan_tar_config():
    """nuPlan from the sensor archives with kedec: AV1 cameras from the original JPEGs, kedec lidar, the parser
    target and the archive folders from the dataset paths."""
    cfg = _compose("dataset=nuplan-tar", "codec=kedec", "dataset_paths.nuplan_archive_roots=/a:/b")
    lwc = cfg.dataset.log_writer_config
    assert lwc.camera_store_option == "av1" and lwc.lidar_codec == "kedec" and lwc.lidar_store_option == "binary"
    assert cfg.dataset.parser._target_ == "py123d.parser.nuplan.nuplan_tar_parser.NuplanTarParser"
    assert cfg.dataset.parser.jpeg_quality is None
    assert cfg.dataset.parser.nuplan_archive_roots == "/a:/b"
    assert not cfg.async_conversion  # sync mode: lidar at 10 Hz, one image per camera and row
    import hydra

    config = hydra.utils.instantiate(lwc)
    assert config.lidar_codec_config.kedec_range_step_m == 0.02 and config.video_codec_config.nvenc_cq == 38


def test_kedec_preset_on_the_physical_ai_av_zip_config():
    cfg = _compose("dataset=physical-ai-av-zip", "codec=kedec", "async_conversion=true")
    lwc = cfg.dataset.log_writer_config
    assert lwc.camera_store_option == "av1" and lwc.lidar_codec == "kedec" and lwc.async_conversion
    assert cfg.dataset.parser._target_ == "py123d.parser.physical_ai_av.physical_ai_av_zip_parser.PhysicalAIAVZipParser"
    assert cfg.dataset.log_writer.sync_config.reference_column == "lidar.lidar_merged.timestamp_us"
    assert cfg.dataset.log_writer.sync_config.direction_overrides == {"camera": "nearest"}


def test_kedec_preset_on_the_opendv_config():
    """OpenDV: AV1 cameras, the kept images as sync reference, the data folder from the dataset paths."""
    cfg = _compose("dataset=opendv", "codec=kedec", "async_conversion=true", "dataset_paths.opendv_data_root=/d")
    lwc = cfg.dataset.log_writer_config
    assert lwc.camera_store_option == "av1" and lwc.async_conversion
    assert not lwc.infer_ego_dynamics and not lwc.infer_box_dynamics
    assert cfg.dataset.parser._target_ == "py123d.parser.opendv.opendv_parser.OpenDVParser"
    assert cfg.dataset.parser.opendv_data_root == "/d"
    assert cfg.dataset.log_writer.sync_config.reference_column == "camera.pcam_f0.timestamp_us"
    assert cfg.dataset.log_writer.sync_config.direction_overrides == {"camera": "nearest"}


def _ray_executor(gpus: int, threads: int) -> RayExecutor:
    executor = RayExecutor.__new__(RayExecutor)
    executor.config = ExecutorResources(
        number_of_nodes=1, number_of_gpus_per_node=gpus, number_of_cpus_per_node=threads
    )
    return executor


def test_gpu_share_per_task():
    assert gpu_share_per_task(_ray_executor(1, 64), True) == pytest.approx(0.0156)
    assert gpu_share_per_task(_ray_executor(8, 64), True) == 0.125
    assert gpu_share_per_task(_ray_executor(8, 4), True) == 1.0
    assert gpu_share_per_task(_ray_executor(0, 64), True) is None
    assert gpu_share_per_task(_ray_executor(8, 64), False) is None
    assert gpu_share_per_task(SequentialExecutor(), True) is None


def test_gpus_per_task_setting():
    kedec = _compose("dataset=nuplan-mini", "codec=kedec").dataset.log_writer_config
    plain = _compose("dataset=nuplan-mini").dataset.log_writer_config
    assert uses_gpu_video(kedec) and not uses_gpu_video(plain)
    svt = OmegaConf.merge(kedec, {"video_codec_config": {"backend": "svt"}})
    assert not uses_gpu_video(svt)
    assert gpus_per_task("auto", kedec, _ray_executor(2, 32)) == 0.0625
    assert gpus_per_task("auto", plain, _ray_executor(2, 32)) is None
    assert gpus_per_task(None, kedec, _ray_executor(2, 32)) is None
    assert gpus_per_task(0.5, plain, _ray_executor(2, 32)) == 0.5


def test_ray_storage_under_ray_tmpdir(monkeypatch, tmp_path):
    monkeypatch.delenv("RAY_TMPDIR", raising=False)
    assert _local_storage_options() == {}
    monkeypatch.setenv("RAY_TMPDIR", str(tmp_path / "ray"))
    assert _local_storage_options() == {"_temp_dir": str(tmp_path / "ray"), "_plasma_directory": str(tmp_path / "ray")}
    assert (tmp_path / "ray").is_dir()


def test_transcode_command(tmp_path):
    """py123d-transcode on a converted log, sequentially, with SVT-AV1."""
    from tests.unit.api.scene.arrow.test_transcode import _log_dir, make_log_metadata, write_source_log

    source = write_source_log(tmp_path / "source")
    target = tmp_path / "target"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "py123d.script.run_transcode",
            f"source_logs_root={source}",
            f"target_logs_root={target}",
            "execution=sequential_executor",
            "log_writer_config.video_codec_config.backend=svt",
            "+log_writer_config.video_codec_config.svt_preset=12",
        ],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path), "PY123D_DATA_ROOT": str(tmp_path)},
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=300,
        check=False,
    )
    assert "Transcoding completed for all 1 logs." in result.stdout, result.stdout[-3000:]
    log_metadata = make_log_metadata()
    camera = pa.ipc.open_file(str(_log_dir(target) / "camera.pcam_f0.arrow")).read_all()
    assert camera.num_rows == 24  # KeepAllImages
    assert (target / log_metadata.split / log_metadata.log_name / "camera.pcam_f0.av1.mp4").exists()
