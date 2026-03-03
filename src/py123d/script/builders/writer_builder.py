import logging

from hydra.utils import instantiate
from omegaconf import DictConfig

from py123d.api.map.abstract_map_writer import AbstractMapWriter
from py123d.api.scene.abstract_log_writer import AbstractLogWriter
from py123d.script.builders.utils.utils_type import validate_type

logger = logging.getLogger(__name__)


def build_map_writer(cfg: DictConfig) -> AbstractMapWriter:
    logger.debug("Building AbstractMapWriter...")
    map_writer: AbstractMapWriter = instantiate(cfg)
    validate_type(map_writer, AbstractMapWriter)
    logger.debug("Building AbstractMapWriter...DONE!")
    return map_writer


def build_log_writer(cfg: DictConfig) -> AbstractLogWriter:
    logger.debug("Building AbstractLogWriter...")
    log_writer: AbstractLogWriter = instantiate(cfg)
    validate_type(log_writer, AbstractLogWriter)
    logger.debug("Building AbstractLogWriter...DONE!")
    return log_writer
