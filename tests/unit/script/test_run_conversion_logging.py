"""Checks that ``py123d-conversion`` prints the messages of loggers that exist before Hydra starts.

The script is run in a subprocess, because ``@hydra.main`` configures logging for the whole process.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator, List

from py123d.datatypes import LogMetadata
from py123d.parser.base_dataset_parser import BaseDatasetParser, BaseLogParser, BaseMapParser, ModalitiesSync


class _FailingLogParser(BaseLogParser):
    """Log parser that fails as soon as the conversion asks for its metadata."""

    def get_log_metadata(self) -> LogMetadata:
        raise RuntimeError("deliberate failure")

    def iter_modalities_sync(self) -> Iterator[ModalitiesSync]:
        raise NotImplementedError


class FailingDatasetParser(BaseDatasetParser):
    """Dataset parser without maps and with a single log that cannot be converted."""

    def __init__(self, **kwargs: Any) -> None:
        """Ignores the arguments of the dataset config whose parser it replaces."""

    def get_map_parsers(self) -> List[BaseMapParser]:
        return []

    def get_log_parsers(self) -> List[BaseLogParser]:
        return [_FailingLogParser()]


def test_conversion_prints_log_messages(tmp_path: Path):
    """The start message and the error of a failed log conversion should reach the console."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "py123d.script.run_conversion",
            "dataset=nureasoning-mini",
            f"dataset.parser._target_={__name__}.FailingDatasetParser",
            "execution=sequential_executor",
        ],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path), "PY123D_DATA_ROOT": str(tmp_path)},
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=120,
        check=False,
    )

    assert "Starting Dataset Conversion..." in result.stdout
    assert "Error converting log" in result.stdout
    assert "RuntimeError: deliberate failure" in result.stdout
