"""Transcode converted 123D logs into other camera and lidar codecs (the kedec codecs by default).

Example::

    py123d-transcode source_logs_root=/data/123D/logs target_logs_root=/data/123D_kedec/logs splits=[nuplan_mini]

See :mod:`py123d.api.scene.arrow.transcode` and docs/notes/kedec.rst.
"""

import gc
import logging
import traceback
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple

import hydra
from omegaconf import DictConfig, OmegaConf

from py123d.api.scene.arrow.transcode import CameraSelection, TranscodeConfig, find_logs, transcode_log
from py123d.common.execution.utils import executor_map_queued
from py123d.common.runtime import setup_runtime
from py123d.script.builders.execution_builder import build_executor
from py123d.script.builders.logging_builder import build_logger
from py123d.script.run_conversion import gpus_per_task

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/transcode"
CONFIG_NAME = "default_transcode"


def _transcode_one(log: Tuple[str, str], cfg: DictConfig) -> Optional[str]:
    """Transcode one log; returns its name if it failed, else None."""
    split, log_name = log
    config = TranscodeConfig(
        log_writer_config=hydra.utils.instantiate(cfg.log_writer_config),
        camera_threads=cfg.camera_threads,
        force=cfg.force,
        ipc_compression=cfg.ipc_compression,
    )
    selection: Optional[CameraSelection] = (
        hydra.utils.instantiate(cfg.camera_selection) if cfg.get("camera_selection") else None
    )
    try:
        written = transcode_log(
            Path(cfg.source_logs_root), Path(cfg.target_logs_root), split, log_name, config, selection
        )
        logger.info(f"{'Transcoded' if written else 'Skipped (exists)'} {split}/{log_name}")
        return None
    except Exception as error:
        logger.error(f"Error transcoding log {split}/{log_name}: {error}")
        logger.error(traceback.format_exc())
        gc.collect()
        return f"{split}/{log_name}"


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """Main entry point of ``py123d-transcode``."""
    build_logger(cfg)
    setup_runtime(cfg)
    # Names as strings: an unquoted override such as log_names=[2021_05] reaches the config as a number.
    splits = [str(s) for s in OmegaConf.to_container(cfg.splits)] if cfg.splits is not None else None
    log_names = [str(s) for s in OmegaConf.to_container(cfg.log_names)] if cfg.log_names is not None else None
    logs = find_logs(Path(cfg.source_logs_root), splits, log_names)  # type: ignore[arg-type]
    logger.info(f"Transcoding {len(logs)} logs from {cfg.source_logs_root} to {cfg.target_logs_root}.")
    executor = build_executor(cfg)
    results: List[Optional[str]] = executor_map_queued(
        executor,
        partial(_transcode_one, cfg=cfg),
        logs,
        name="Transcode",
        num_gpus=gpus_per_task(cfg.get("gpus_per_task", "auto"), cfg.log_writer_config, executor),
    )
    failed = [name for name in results if name is not None]
    if failed:
        logger.warning(f"Transcoding did NOT complete for {len(failed)} of {len(logs)} logs:\n  " + "\n  ".join(failed))
    else:
        logger.info(f"Transcoding completed for all {len(logs)} logs.")


if __name__ == "__main__":
    main()
