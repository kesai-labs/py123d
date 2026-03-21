import logging

from hydra.utils import instantiate
from omegaconf import DictConfig

from py123d.api.map.base_map_writer import BaseMapWriter
from py123d.api.scene.base_log_writer import BaseLogWriter
from py123d.script.builders.utils.utils_type import validate_type

logger = logging.getLogger(__name__)


def build_map_writer(cfg: DictConfig) -> BaseMapWriter:
    logger.debug("Building BaseMapWriter...")
    map_writer: BaseMapWriter = instantiate(cfg)
    validate_type(map_writer, BaseMapWriter)
    logger.debug("Building BaseMapWriter...DONE!")
    return map_writer


def build_log_writer(cfg: DictConfig) -> BaseLogWriter:
    logger.debug("Building BaseLogWriter...")
    log_writer: BaseLogWriter = instantiate(cfg)
    validate_type(log_writer, BaseLogWriter)
    logger.debug("Building BaseLogWriter...DONE!")
    return log_writer
