import contextlib
import gc
import logging
import os
import shutil
import traceback
from functools import partial
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig

from py123d.common.execution.utils import executor_map_chunked_list
from py123d.common.runtime import setup_runtime
from py123d.parser.base_dataset_parser import BaseDatasetParser, BaseLogParser, BaseMapParser
from py123d.script.builders.execution_builder import build_executor
from py123d.script.builders.logging_builder import build_logger
from py123d.script.builders.writer_builder import build_log_writer, build_map_writer

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/conversion"
CONFIG_NAME = "default_conversion"


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for dataset conversion.
    :param cfg: omegaconf dictionary
    """

    build_logger(cfg)
    setup_runtime(cfg)

    logger.info("Starting Dataset Conversion...")
    dataset_parser: BaseDatasetParser = hydra.utils.instantiate(cfg.dataset.parser)

    executor = build_executor(cfg)
    parser_class_name = dataset_parser.__class__.__name__

    map_parsers: List[BaseMapParser] = dataset_parser.get_map_parsers()
    failed_maps: List[str] = executor_map_chunked_list(
        executor,
        partial(_convert_maps, cfg=cfg),
        map_parsers,
        name=f"Maps {parser_class_name}",
    )

    async_conversion = cfg.async_conversion
    convert_fn = _convert_logs_async if async_conversion else _convert_logs
    log_parsers: List[BaseLogParser] = dataset_parser.get_log_parsers()
    failed_logs: List[str] = executor_map_chunked_list(
        executor,
        partial(convert_fn, cfg=cfg),
        log_parsers,
        name=f"Logs {parser_class_name}",
    )

    if len(failed_maps) + len(failed_logs) > 0:
        logger.warning(
            f"Conversion did NOT complete for {len(failed_maps)} of {len(map_parsers)} maps "
            f"and {len(failed_logs)} of {len(log_parsers)} logs:\n"
            + "\n".join([f"  map {name}" for name in failed_maps] + [f"  log {name}" for name in failed_logs])
        )
    else:
        logger.info(f"Conversion completed successfully for all {len(map_parsers)} maps and {len(log_parsers)} logs.")
    _cleanup_incomplete(cfg, failed_maps, failed_logs)

    # Streaming datasets delete the raw files they downloaded, unless ``keep_downloads`` is set.
    if not cfg.dataset.get("keep_downloads", True):
        shutil.rmtree(cfg.dataset.parser.downloader.output_dir, ignore_errors=True)
        # The download root goes too, once no other conversion has a folder in it (rmdir fails otherwise).
        with contextlib.suppress(OSError):
            os.rmdir(cfg.dataset_paths.py123d_download_root)


def _convert_maps(args: List[BaseMapParser], cfg: DictConfig) -> List[str]:
    """Converts the maps, returns the names of those that did not complete."""
    map_writer = build_map_writer(cfg.dataset.map_writer)
    failed_maps: List[str] = []
    for map_parser in args:
        map_name = "unknown"
        try:
            map_metadata = map_parser.get_map_metadata()
            map_name = f"{map_metadata.split or map_metadata.dataset}/{map_metadata.log_name or map_metadata.location}"
            map_needs_writing = map_writer.reset(map_metadata)
            if map_needs_writing:
                for map_object in map_parser.iter_map_objects():
                    map_writer.write_map_object(map_object)
            map_writer.close()
        except Exception as e:
            logger.error(f"Error converting map {map_name}: {e}")
            logger.error(traceback.format_exc())
            failed_maps.append(map_name)
            try:
                map_writer.close(incomplete=True)  # stays in the incomplete directory, e.g. for debugging
            except Exception:
                map_writer = build_map_writer(cfg.dataset.map_writer)  # a broken map may not close either
            gc.collect()
    return failed_maps


def _convert_logs(args: List[BaseLogParser], cfg: DictConfig) -> List[str]:
    """Converts the logs, returns the names of those that did not complete."""
    log_writer = build_log_writer(cfg.dataset.log_writer)
    failed_logs: List[str] = []
    for log_parser in args:
        log_name = "unknown"
        try:
            log_metadata = log_parser.get_log_metadata()
            log_name = f"{log_metadata.split}/{log_metadata.log_name}"
            log_needs_writing = log_writer.reset(log_metadata)
            if log_needs_writing:
                route_xyz = log_parser.get_route_xyz()
                if route_xyz is not None:
                    log_writer.set_route(route_xyz)
                for modalities_sync in log_parser.iter_modalities_sync():
                    log_writer.write_sync(modalities_sync)
            log_writer.close()
        except Exception as e:
            logger.error(f"Error converting log {log_name}: {e}")
            logger.error(traceback.format_exc())
            failed_logs.append(log_name)
            try:
                log_writer.close(incomplete=True)  # stays in the incomplete directory, e.g. for debugging
            except Exception:
                log_writer = build_log_writer(cfg.dataset.log_writer)  # a broken log may not close either
            gc.collect()
    return failed_logs


def _convert_logs_async(args: List[BaseLogParser], cfg: DictConfig) -> List[str]:
    """Converts the logs, returns the names of those that did not complete."""
    log_writer = build_log_writer(cfg.dataset.log_writer)
    failed_logs: List[str] = []
    for log_parser in args:
        log_name = "unknown"
        try:
            log_metadata = log_parser.get_log_metadata()
            log_name = f"{log_metadata.split}/{log_metadata.log_name}"
            log_needs_writing = log_writer.reset(log_metadata)
            if log_needs_writing:
                route_xyz = log_parser.get_route_xyz()
                if route_xyz is not None:
                    log_writer.set_route(route_xyz)
                for modality in log_parser.iter_modalities_async():
                    log_writer.write_async(modality)
            log_writer.close()
        except Exception as e:
            logger.error(f"Error converting log {log_name}: {e}")
            logger.error(traceback.format_exc())
            failed_logs.append(log_name)
            try:
                log_writer.close(incomplete=True)  # stays in the incomplete directory, e.g. for debugging
            except Exception:
                log_writer = build_log_writer(cfg.dataset.log_writer)  # a broken log may not close either
            gc.collect()
    return failed_logs


def _cleanup_incomplete(cfg: DictConfig, failed_maps: List[str], failed_logs: List[str]) -> None:
    """Deletes the maps and logs that did not complete, unless ``keep_incomplete`` is set."""
    incomplete_logs_root = Path(cfg.dataset_paths.py123d_logs_root) / ".incomplete"
    incomplete_maps_root = Path(cfg.dataset_paths.py123d_maps_root) / ".incomplete"
    if not cfg.keep_incomplete:
        for failed_map in failed_maps:
            (incomplete_maps_root / f"{failed_map}.arrow").unlink(missing_ok=True)
        for failed_log in failed_logs:
            shutil.rmtree(incomplete_logs_root / failed_log, ignore_errors=True)
    # The incomplete directories are only removed if empty, another conversion may be writing to them.
    for incomplete_root in (incomplete_logs_root, incomplete_maps_root):
        for directory in [*incomplete_root.glob("*"), incomplete_root]:
            with contextlib.suppress(OSError):
                directory.rmdir()


if __name__ == "__main__":
    main()
