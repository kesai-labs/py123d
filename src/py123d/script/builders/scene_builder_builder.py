# NOTE: @DanielDauner this file name is silly :D.

import logging

from hydra.utils import instantiate
from omegaconf import DictConfig

from py123d.api.scene.scene_builder import SceneBuilder

logger = logging.getLogger(__name__)


def build_scene_builder(cfg: DictConfig) -> SceneBuilder:
    """
    Builds the scene builder.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :return: Instance of SceneBuilder.
    """
    logger.info("Building SceneBuilder...")
    scene_builder: SceneBuilder = instantiate(cfg)
    if not isinstance(scene_builder, SceneBuilder):
        raise TypeError(f"Expected SceneBuilder, got {type(scene_builder)}")
    logger.info("Building SceneBuilder...DONE!")
    return scene_builder
