import gc
import logging
import traceback
from functools import partial
from typing import List

import hydra
from omegaconf import DictConfig

from py123d.common.execution.utils import executor_map_chunked_list
from py123d.parser.base_dataset_parser import BaseDatasetParser, BaseLogParser, BaseMapParser
from py123d.script.builders.execution_builder import build_executor
from py123d.script.builders.logging_builder import build_logger
from py123d.script.builders.writer_builder import build_log_writer, build_map_writer
from py123d.script.utils.dataset_path_utils import setup_dataset_paths

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
    setup_dataset_paths(cfg.dataset_paths)

    logger.info("Starting Dataset Conversion...")
    dataset_parser: BaseDatasetParser = hydra.utils.instantiate(cfg.dataset.parser)

    executor = build_executor(cfg)
    parser_class_name = dataset_parser.__class__.__name__

    map_parsers: List[BaseMapParser] = dataset_parser.get_map_parsers()
    executor_map_chunked_list(
        executor,
        partial(_convert_maps, cfg=cfg),
        map_parsers,
        name=f"Maps {parser_class_name}",
    )

    async_conversion = cfg.async_conversion
    convert_fn = _convert_logs_async if async_conversion else _convert_logs
    executor_map_chunked_list(
        executor,
        partial(convert_fn, cfg=cfg),
        dataset_parser.get_log_parsers(),
        name=f"Logs {parser_class_name}",
    )


def _convert_maps(args: List[BaseMapParser], cfg: DictConfig) -> List:
    map_writer = build_map_writer(cfg.dataset.map_writer)
    for map_parser in args:
        try:
            map_metadata = map_parser.get_map_metadata()
            map_needs_writing = map_writer.reset(map_metadata)
            if map_needs_writing:
                for map_object in map_parser.iter_map_objects():
                    map_writer.write_map_object(map_object)
            map_writer.close()
        except Exception as e:
            logger.error(f"Error converting map: {e}")
            logger.error(traceback.format_exc())
            map_writer.close()
            gc.collect()
    return []


def _convert_logs(args: List[BaseLogParser], cfg: DictConfig) -> List:
    log_writer = build_log_writer(cfg.dataset.log_writer)
    for log_parser in args:
        try:
            log_metadata = log_parser.get_log_metadata()
            log_needs_writing = log_writer.reset(log_metadata)
            if log_needs_writing:
                for modalities_sync in log_parser.iter_modalities_sync():
                    log_writer.write_sync(modalities_sync)
            log_writer.close()
        except Exception as e:
            logger.error(f"Error converting log: {e}")
            logger.error(traceback.format_exc())
            log_writer.close()
            gc.collect()
    return []


def _convert_logs_async(args: List[BaseLogParser], cfg: DictConfig) -> List:
    log_writer = build_log_writer(cfg.dataset.log_writer)
    for log_parser in args:
        try:
            log_metadata = log_parser.get_log_metadata()
            log_needs_writing = log_writer.reset(log_metadata)
            if log_needs_writing:
                for modality in log_parser.iter_modalities_async():
                    log_writer.write_async(modality)
            log_writer.close()
        except Exception as e:
            logger.error(f"Error converting log: {e}")
            logger.error(traceback.format_exc())
            log_writer.close()
            gc.collect()
    return []


if __name__ == "__main__":
    main()
