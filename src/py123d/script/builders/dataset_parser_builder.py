import logging
from typing import List

from hydra.utils import instantiate
from omegaconf import DictConfig

from py123d.parser.base_dataset_parser import BaseDatasetParser
from py123d.script.builders.utils.utils_type import validate_type

logger = logging.getLogger(__name__)


def build_dataset_parsers(cfg: DictConfig) -> List[BaseDatasetParser]:
    logger.info("Building DatasetParser...")
    instantiated_dataset_parsers: List[BaseDatasetParser] = []
    for dataset_type in cfg.values():
        processor: BaseDatasetParser = instantiate(dataset_type)
        validate_type(processor, BaseDatasetParser)
        instantiated_dataset_parsers.append(processor)

    logger.info("Building DatasetParser...DONE!")
    return instantiated_dataset_parsers
