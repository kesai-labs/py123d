import logging

import hydra
from omegaconf import DictConfig

from py123d.script.builders.bokeh_config_builder import build_bokeh_config
from py123d.script.builders.execution_builder import build_executor
from py123d.script.builders.logging_builder import build_logger
from py123d.script.builders.scene_builder_builder import build_scene_builder
from py123d.script.builders.scene_filter_builder import build_scene_filter
from py123d.script.utils.dataset_path_utils import setup_dataset_paths

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/bokeh"
CONFIG_NAME = "default_bokeh"


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

    if len(scenes) == 0:
        raise ValueError("No scenes found for the given filter. Please check your filter criteria and dataset paths.")

    # Build Bokeh config
    bokeh_config = build_bokeh_config(cfg.bokeh_config)

    # Launch Bokeh viewer
    from bokeh.server.server import Server

    from py123d.visualization.bokeh.bokeh_viewer import BokehViewer

    def make_doc(doc):
        viewer = BokehViewer(
            scenes=scenes,
            radius=bokeh_config.bev_radius,
            show_lidar=bokeh_config.show_lidar,
            show_map=bokeh_config.show_map,
        )
        viewer.build(doc)

    server = Server({"/": make_doc}, port=bokeh_config.server_port)
    server.start()

    logger.info(f"123D Bokeh Viewer running at: http://localhost:{bokeh_config.server_port}/")
    print(f"\n  123D Bokeh Viewer running at: http://localhost:{bokeh_config.server_port}/\n")

    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()


if __name__ == "__main__":
    main()
