import logging
import uuid
from typing import Any

from hydra.utils import instantiate
from omegaconf import DictConfig

from py123d.api.scene.scene_filter import SceneFilter

logger = logging.getLogger(__name__)


def is_valid_uuid(token: Any) -> bool:
    """
    Basic check that a scene token is a valid UUID string.
    :token: parsed by hydra.
    :return: true if it looks valid, otherwise false.
    """
    if not isinstance(token, str):
        return False

    try:
        uuid.UUID(token)
        return True
    except ValueError:
        return False


def build_scene_filter(cfg: DictConfig) -> SceneFilter:
    """
    Builds the scene filter.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :return: Instance of SceneFilter.
    """
    logger.info("Building SceneFilter...")
    if cfg.scene_uuids and not all(map(is_valid_uuid, cfg.scene_uuids)):
        raise RuntimeError(
            "Expected all scene tokens to be valid UUID strings. "
            "Your shell may strip quotes causing hydra to parse a token as a float, so consider passing them like "
            "scene_filter.scene_uuids='[\"550e8400-e29b-41d4-a716-446655440000\", ...]'"
        )
    scene_filter: SceneFilter = instantiate(cfg)
    assert isinstance(scene_filter, SceneFilter)
    logger.info("Building SceneFilter...DONE!")
    return scene_filter
