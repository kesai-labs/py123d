import logging

import hydra
from omegaconf import DictConfig

from py123d.script.builders.execution_builder import build_executor
from py123d.script.builders.logging_builder import build_logger
from py123d.script.builders.scene_builder_builder import build_scene_builder
from py123d.script.builders.scene_filter_builder import build_scene_filter
from py123d.script.builders.viser_config_builder import build_viser_config
from py123d.script.utils.dataset_path_utils import setup_dataset_paths
from py123d.visualization.viser.viser_viewer import ViserViewer

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/viser"
CONFIG_NAME = "default_viser"


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    # Setup logging
    build_logger(cfg)

    # Initialize dataset paths
    setup_dataset_paths(cfg.dataset_paths)

    # Build executor
    executor = build_executor(cfg)

    # Build scene filter and scene builder
    scene_filter = build_scene_filter(cfg.scene_filter)
    scene_builder = build_scene_builder(cfg.scene_builder)

    # Get scenes from scene builder
    scenes = scene_builder.get_scenes(scene_filter, executor=executor)

    # Build Viser config
    viser_config = build_viser_config(cfg.viser_config)

    # Launch Viser viewer with the scenes
    ViserViewer(scenes=scenes, viser_config=viser_config)


if __name__ == "__main__":
    main()
