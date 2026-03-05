import gc
import logging
import traceback
from functools import partial
from typing import List

import hydra
from omegaconf import DictConfig

from py123d.common.execution.utils import executor_map_chunked_list
from py123d.parser.abstract_dataset_parser import DatasetParser, LogParser, MapParser
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
    dataset_parser: DatasetParser = hydra.utils.instantiate(cfg.dataset.parser)

    executor = build_executor(cfg)
    parser_class_name = dataset_parser.__class__.__name__

    map_parsers: List[MapParser] = dataset_parser.get_map_parsers()
    executor_map_chunked_list(
        executor,
        partial(_convert_maps, cfg=cfg),
        map_parsers,
        name=f"Maps {parser_class_name}",
    )

    executor_map_chunked_list(
        executor,
        partial(_convert_logs, cfg=cfg),
        dataset_parser.get_log_parsers(),
        name=f"Logs {parser_class_name}",
    )


def _convert_maps(args: List[MapParser], cfg: DictConfig) -> List:
    map_writer = build_map_writer(cfg.map_writer)
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


def _convert_logs(args: List[LogParser], cfg: DictConfig) -> List:
    log_writer = build_log_writer(cfg.log_writer)
    for log_parser in args:
        try:
            log_metadata = log_parser.get_log_metadata()
            ego_metadata = log_parser.get_ego_metadata()
            box_detection_metadata = log_parser.get_box_detection_metadata()
            pinhole_camera_metadatas = log_parser.get_pinhole_camera_metadatas()
            fisheye_mei_camera_metadatas = log_parser.get_fisheye_mei_camera_metadatas()
            lidar_metadatas = log_parser.get_lidar_metadatas()

            log_needs_writing = log_writer.reset(
                log_metadata,
                ego_metadata,
                box_detection_metadata,
                pinhole_camera_metadatas,
                fisheye_mei_camera_metadatas,
                lidar_metadatas,
            )

            if log_needs_writing:
                for frame in log_parser.iter_frames():
                    log_writer.write(**frame.to_writer_kwargs())

            log_writer.close()
        except Exception as e:
            logger.error(f"Error converting log: {e}")
            logger.error(traceback.format_exc())
            log_writer.close()
            gc.collect()
    return []


if __name__ == "__main__":
    main()
