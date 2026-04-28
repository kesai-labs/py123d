"""Abstract base class for dataset downloaders.

A :class:`BaseDownloader` is a small, Hydra-instantiable object that encapsulates
*how* to fetch a specific dataset (bucket layout, auth, selection logic, modalities)
so that two different entry points can share the same implementation:

1. The standalone ``py123d-download`` CLI, which instantiates one downloader and
   calls :meth:`BaseDownloader.download`.

2. Streaming-during-conversion: a dataset parser receives a downloader instance via
   its ``downloader`` constructor argument, optionally redirects
   :attr:`BaseDownloader.output_dir` to a managed temp directory, and calls
   :meth:`BaseDownloader.download` before reading the materialized files.

Subclasses set :attr:`output_dir` and :attr:`dry_run` in their ``__init__`` and
implement :meth:`download`. When ``output_dir`` is ``None`` the parser assigns a
``tempfile.TemporaryDirectory`` before invoking :meth:`download`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class BaseDownloader(ABC):
    """Common contract for dataset downloaders shared by the CLI and parser streaming paths.

    :ivar output_dir: Destination for downloaded files. ``None`` means "assign a temp dir
        at download time" — streaming parsers use this to point the downloader at a
        :class:`tempfile.TemporaryDirectory` they manage.
    :ivar dry_run: When ``True``, :meth:`download` logs the plan (source, selection, size
        estimates if available) without writing any files.
    """

    output_dir: Optional[Path]
    dry_run: bool

    @abstractmethod
    def download(self) -> None:
        """Fetch the selected data into :attr:`output_dir`.

        Implementations must respect :attr:`dry_run` (log-only path) and may assume
        :attr:`output_dir` is non-``None`` when called. The CLI and streaming parsers
        are responsible for ensuring :attr:`output_dir` is set before invoking this method.
        """
