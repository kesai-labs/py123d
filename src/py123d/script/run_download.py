"""Hydra entry point for ``py123d-download``.

Instantiates a :class:`~py123d.parser.base_downloader.BaseDownloader` from the
composed config (``dataset=<name>``) and runs its :meth:`download` method. The
configuration surface is intentionally thin — all dataset-specific knobs live on
the downloader class itself and are exposed via the per-dataset YAML under
``config/download/dataset/``.

Usage::

    # Bulk-download full splits (output_dir falls back to $<DATASET>_DATA_ROOT)
    py123d-download dataset=wod-motion
    py123d-download dataset=wod-perception
    py123d-download dataset=ncore

    # Select a subset
    py123d-download dataset=wod-motion dataset.downloader.num_shards=5
    py123d-download dataset=ncore dataset.downloader.num_clips=5 \
                    dataset.downloader.sample_random=true

    # Preview the plan without fetching
    py123d-download dataset=ncore dataset.downloader.dry_run=true
"""

import logging

import hydra
from omegaconf import DictConfig

from py123d.parser.base_downloader import BaseDownloader
from py123d.script.builders.logging_builder import build_logger
from py123d.script.utils.dataset_path_utils import setup_dataset_paths

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/download"
CONFIG_NAME = "default_download"


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """Entrypoint for dataset downloading.

    :param cfg: Composed Hydra config carrying a ``dataset.downloader`` block that
        resolves to a :class:`BaseDownloader` subclass via ``_target_``.
    """
    build_logger(cfg)
    setup_dataset_paths(cfg.dataset_paths)

    logger.info("Starting Dataset Download...")
    downloader: BaseDownloader = hydra.utils.instantiate(cfg.dataset.downloader)
    downloader.download()


if __name__ == "__main__":
    main()
