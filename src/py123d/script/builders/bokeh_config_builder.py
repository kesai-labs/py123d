import logging

from hydra.utils import instantiate
from omegaconf import DictConfig

from py123d.script.builders.utils.utils_type import validate_type
from py123d.visualization.bokeh.bokeh_config import BokehConfig

logger = logging.getLogger(__name__)


def build_bokeh_config(cfg: DictConfig) -> BokehConfig:
    """
    Builds the config dataclass for the Bokeh viewer.
    :param cfg: DictConfig. Configuration that is used to run the viewer.
    :return: Instance of BokehConfig.
    """
    logger.info("Building BokehConfig...")
    bokeh_config: BokehConfig = instantiate(cfg)
    validate_type(bokeh_config, BokehConfig)
    logger.info("Building BokehConfig...DONE!")
    return bokeh_config
