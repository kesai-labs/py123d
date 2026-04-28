"""Download utilities for the PandaSet dataset (unofficial HuggingFace mirror).

The official ScaleAI PandaSet site is offline, so this module targets the community
HuggingFace mirror at https://huggingface.co/datasets/georghess/pandaset. The mirror
ships the dataset as a single monolithic archive::

    georghess/pandaset/
    ├── .gitattributes
    ├── README.md
    └── pandaset.zip                  (~44.5 GB — all 103 logs)

Unlike NCore (one HuggingFace folder per clip) this means we cannot selectively
fetch individual logs from the network: the whole zip must be downloaded once,
cached locally, and then logs are extracted from it on demand.

This module exposes :class:`PandasetDownloader` (Hydra-instantiable) which powers
both paths:

1. ``py123d-download dataset=pandaset`` — downloads the zip to a session-scoped
   ``tempfile.TemporaryDirectory``, extracts the selected logs into
   :attr:`output_dir` (typically ``$PANDASET_DATA_ROOT``), then the temp dir (and
   the zip with it) is cleaned up when ``download()`` returns.
2. The ``PandasetParser`` streaming path — each log parser pulls its assigned log
   from a shared temp-dir zip cache into a per-log ``tempfile.TemporaryDirectory``,
   converts it, and deletes the per-log temp directory before moving on. The
   shared zip cache itself lives at
   ``<system_temp>/py123d-pandaset-cache/pandaset.zip`` by default so multiple Ray
   workers (each with their own unpickled downloader instance) reuse the same
   on-disk zip rather than each re-downloading 44 GB.

Nothing is written to the HuggingFace hub cache — the zip is always fetched
directly into a ``local_dir`` we control.

Inside the archive the layout matches the standard PandaSet on-disk structure::

    pandaset/{log_name}/
    ├── camera/{cam_name}/NN.jpg + intrinsics.json, poses.json, timestamps.json
    ├── lidar/NN.pkl.gz + poses.json, timestamps.json
    ├── annotations/cuboids/NN.pkl.gz
    ├── annotations/semseg/NN.pkl.gz      (unused by the parser)
    └── meta/gps.json, timestamps.json
"""

from __future__ import annotations

import logging
import os
import random as _random_mod
import tempfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

from py123d.parser.base_downloader import BaseDownloader
from py123d.parser.pandaset.utils.pandaset_constants import PANDASET_LOG_NAMES

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

PANDASET_REPO_ID = "georghess/pandaset"
PANDASET_REPO_TYPE = "dataset"
PANDASET_ZIP_FILENAME = "pandaset.zip"
# Prefix of all data entries inside the zip. The archive wraps everything in a
# top-level ``pandaset/`` folder — entries are ``pandaset/001/camera/...``.
PANDASET_ZIP_ROOT = "pandaset"

# Default location for the shared streaming zip cache. Lives under the system temp
# directory so it survives across Ray workers within a single conversion run and
# is cleaned up by the OS on reboot. Users can override via
# :attr:`PandasetDownloader.zip_cache_dir` when they want a persistent, filesystem-
# specific cache location.
DEFAULT_STREAMING_ZIP_CACHE_DIR = Path(tempfile.gettempdir()) / "py123d-pandaset-cache"


def _require_hf_hub():
    """Lazy import — ``huggingface_hub`` is only needed once a download is requested."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required for PandaSet downloads. Install it with:\n"
            "  pip install py123d[pandaset]\n"
            "or directly:\n"
            "  pip install huggingface_hub\n"
        ) from exc
    return hf_hub_download


def resolve_hf_token(cli_token: Optional[str] = None) -> Optional[str]:
    """Resolve the HF token from (in order): explicit arg, ``$HF_TOKEN``, ``$HUGGINGFACE_HUB_TOKEN``.

    The ``georghess/pandaset`` mirror is public, so ``None`` is fine; a token is only
    needed if the user pins a private fork.
    """
    return cli_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def ensure_pandaset_zip_at(
    target_dir: Path,
    hf_token: Optional[str] = None,
    revision: str = "main",
) -> Path:
    """Ensure ``pandaset.zip`` exists in ``target_dir`` and return its path.

    Downloads the file directly into ``target_dir`` via ``hf_hub_download(local_dir=...)``
    — nothing is written to the HuggingFace hub cache. When the target already has a
    ``pandaset.zip``, the function short-circuits and returns without re-downloading
    (no size/hash validation — callers that want a strict check should remove the
    file first).

    :param target_dir: Destination directory. Created if missing.
    :param hf_token: Optional HF token; resolved through :func:`resolve_hf_token`.
    :param revision: Repo revision (branch/tag/commit).
    :return: Path to the zip file inside ``target_dir``.
    """
    hf_hub_download = _require_hf_hub()
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / PANDASET_ZIP_FILENAME
    if zip_path.exists():
        logger.debug("Re-using existing pandaset.zip at %s", zip_path)
        return zip_path
    logger.info("Downloading pandaset.zip (~44.5 GB) to %s ...", target_dir)
    hf_hub_download(
        repo_id=PANDASET_REPO_ID,
        repo_type=PANDASET_REPO_TYPE,
        filename=PANDASET_ZIP_FILENAME,
        revision=revision,
        token=resolve_hf_token(hf_token),
        local_dir=str(target_dir),
    )
    if not zip_path.exists():
        raise RuntimeError(f"Expected {zip_path} after download but it is missing.")
    return zip_path


def extract_pandaset_log(
    zip_path: Path,
    log_name: str,
    output_dir: Path,
) -> Path:
    """Extract the ``pandaset/{log_name}/`` subtree from ``zip_path`` into ``output_dir``.

    The ``pandaset/`` prefix is stripped so the resulting on-disk layout matches
    what :class:`~py123d.parser.pandaset.pandaset_parser.PandasetParser` expects
    in local mode: ``output_dir/{log_name}/camera/...``, ``output_dir/{log_name}/lidar/...``.

    :param zip_path: Path to the cached ``pandaset.zip`` (see :func:`ensure_pandaset_zip`).
    :param log_name: Log identifier (e.g. ``"001"``). Must be in :data:`PANDASET_LOG_NAMES`.
    :param output_dir: Destination root. The log lands at ``output_dir/{log_name}/``.
    :return: Path to the extracted log directory ``output_dir/{log_name}``.
    """
    if log_name not in PANDASET_LOG_NAMES:
        raise ValueError(f"Unknown PandaSet log {log_name!r}; must be one of {PANDASET_LOG_NAMES[:3]}... (103 total).")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_prefix = f"{PANDASET_ZIP_ROOT}/{log_name}/"

    extracted_count = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if not info.filename.startswith(log_prefix) or info.is_dir():
                continue
            # Strip the leading ``pandaset/`` so the output matches the local layout.
            relative = info.filename[len(PANDASET_ZIP_ROOT) + 1 :]  # e.g. "001/camera/front_camera/00.jpg"
            dest = output_dir / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(dest, "wb") as dst:
                # shutil.copyfileobj would work too, but a plain loop keeps the dependency list clean.
                while True:
                    chunk = src.read(1 << 20)  # 1 MB
                    if not chunk:
                        break
                    dst.write(chunk)
            extracted_count += 1

    if extracted_count == 0:
        raise RuntimeError(
            f"No entries matching {log_prefix!r} found in {zip_path}. "
            "The archive may be corrupt or the log name may not exist on this revision."
        )
    logger.info("Extracted %d files for log %s into %s", extracted_count, log_name, output_dir / log_name)
    return output_dir / log_name


# ======================================================================================
# Downloader (Hydra-instantiable, shared by py123d-download and PandasetParser streaming)
# ======================================================================================


class PandasetDownloader(BaseDownloader):
    """Downloader for the PandaSet dataset via the community HuggingFace mirror.

    Operates in two modes:

    * :meth:`download` — ensures the 44.5 GB zip is cached, then (optionally) extracts
      all selected logs into :attr:`output_dir`. Used by ``py123d-download dataset=pandaset``.
    * :meth:`download_single_log` — extracts exactly one log from the cached zip into
      a caller-provided directory. Used by :class:`~py123d.parser.pandaset.pandaset_parser.PandasetParser`
      in streaming mode to drop each log into a per-log temp directory.

    The zip itself is cached persistently by HuggingFace Hub (under ``$HF_HOME/hub``
    by default), so a single ~45 GB download is reused across runs and across both modes.

    The instance is picklable (simple attrs only) so it can be embedded in log-parser
    objects shipped across a Ray process pool.
    """

    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        zip_cache_dir: Optional[Union[str, Path]] = None,
        revision: str = "main",
        hf_token: Optional[str] = None,
        log_names: Optional[List[str]] = None,
        num_logs: Optional[int] = None,
        sample_random: bool = False,
        seed: int = 0,
        dry_run: bool = False,
    ) -> None:
        """Initialize the PandaSet downloader.

        :param output_dir: Destination directory for :meth:`download` — where
            extracted logs land as ``output_dir/{log_name}/``. Ignored by
            :meth:`download_single_log` (which takes its own ``output_dir`` arg).
        :param zip_cache_dir: Persistent location for the ~44.5 GB ``pandaset.zip``
            used by :meth:`download_single_log` (streaming mode). ``None`` (default)
            uses :data:`DEFAULT_STREAMING_ZIP_CACHE_DIR`
            (``<system_temp>/py123d-pandaset-cache``) so Ray workers on the same
            machine share one on-disk zip. Set to a specific filesystem path when
            system temp is undersized for the zip. :meth:`download` (bulk) always
            uses a session-scoped :class:`tempfile.TemporaryDirectory` and ignores
            this field.
        :param revision: HuggingFace dataset branch, tag, or commit.
        :param hf_token: HF access token. Resolves through :func:`resolve_hf_token`
            — falls back to ``$HF_TOKEN`` / ``$HUGGINGFACE_HUB_TOKEN`` when ``None``.
            The ``georghess/pandaset`` mirror is public, so a token is not required.
        :param log_names: Explicit log identifiers (e.g. ``["001", "002"]``).
            Mutually exclusive with ``num_logs``. Each must be in
            :data:`~py123d.parser.pandaset.utils.pandaset_constants.PANDASET_LOG_NAMES`.
        :param num_logs: Select the first N logs (or N random logs when
            ``sample_random=True``) from the full catalog.
        :param sample_random: Randomize ``num_logs`` selection.
        :param seed: RNG seed used when ``sample_random=True``.
        :param dry_run: If ``True``, :meth:`download` logs the plan without fetching.
        """
        if log_names and num_logs is not None:
            raise ValueError("log_names and num_logs are mutually exclusive.")
        if num_logs is not None and num_logs <= 0:
            raise ValueError("num_logs must be a positive integer.")
        if log_names is not None:
            unknown = [ln for ln in log_names if ln not in PANDASET_LOG_NAMES]
            if unknown:
                raise ValueError(f"Unknown PandaSet log(s): {unknown}. Must be in PANDASET_LOG_NAMES (103 entries).")

        self.output_dir: Optional[Path] = Path(output_dir) if output_dir is not None else None
        self.dry_run: bool = dry_run

        self.revision: str = revision
        self.hf_token: Optional[str] = resolve_hf_token(hf_token)
        self.zip_cache_dir: Optional[Path] = Path(zip_cache_dir) if zip_cache_dir is not None else None

        self._explicit_log_names: Optional[List[str]] = list(log_names) if log_names else None
        self._num_logs: Optional[int] = num_logs
        self._sample_random: bool = sample_random
        self._seed: int = seed

    # ----- Selection ------------------------------------------------------------------

    def resolve_log_names(self) -> List[str]:
        """Return the log names selected by the current configuration.

        Used by :meth:`download` for bulk extraction and by the streaming parser to
        enumerate work across Ray workers. Selection is deterministic given the same
        seed.
        """
        if self._explicit_log_names:
            resolved = list(self._explicit_log_names)
        elif self._num_logs is None or self._num_logs >= len(PANDASET_LOG_NAMES):
            resolved = list(PANDASET_LOG_NAMES)
        elif self._sample_random:
            rng = _random_mod.Random(self._seed)
            resolved = sorted(rng.sample(PANDASET_LOG_NAMES, self._num_logs))
        else:
            resolved = list(PANDASET_LOG_NAMES[: self._num_logs])
        return resolved

    # ----- Bulk download (py123d-download) --------------------------------------------

    def download(self) -> None:
        """Inherited, see superclass.

        Bulk flow: the ~44.5 GB zip is downloaded into a session-scoped
        :class:`tempfile.TemporaryDirectory`, the selected logs are extracted into
        :attr:`output_dir`, and the temp directory (with the zip inside it) is
        cleaned up when the context exits. No artifact of the zip survives
        :meth:`download`.
        """
        log_names = self.resolve_log_names()

        logger.info("PandaSet source:         %s@%s/%s", PANDASET_REPO_ID, self.revision, PANDASET_ZIP_FILENAME)
        logger.info("PandaSet logs selected:  %d / %d", len(log_names), len(PANDASET_LOG_NAMES))
        logger.info("PandaSet target dir:     %s", self.output_dir)

        if self.dry_run:
            logger.info("dry_run=True — not downloading. Plan covers %d log(s).", len(log_names))
            return

        if not log_names:
            logger.warning("No logs selected — nothing to download.")
            return

        assert self.output_dir is not None, "PandasetDownloader.output_dir must be set before download()."
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="py123d-pandaset-bulk-") as tmp:
            tmp_dir = Path(tmp)
            logger.info("Downloading zip to temp dir (auto-cleaned): %s", tmp_dir)
            zip_path = ensure_pandaset_zip_at(target_dir=tmp_dir, hf_token=self.hf_token, revision=self.revision)

            for log_name in log_names:
                dest = self.output_dir / log_name
                if dest.exists() and any(dest.iterdir()):
                    logger.debug("Skip already-extracted log %s at %s", log_name, dest)
                    continue
                extract_pandaset_log(zip_path=zip_path, log_name=log_name, output_dir=self.output_dir)

            logger.info("Bulk extraction complete; removing temp zip at %s", zip_path)
            # tempfile.TemporaryDirectory context manager deletes tmp_dir (and the zip) here.

        logger.info("PandaSet download complete: %s", self.output_dir)

    # ----- Per-log extract (streaming conversion) -------------------------------------

    def download_single_log(self, log_name: str, output_dir: Union[str, Path]) -> Path:
        """Extract one log into ``output_dir`` and return its path.

        Intended for the PandaSet streaming parser — each log lands in its own per-log
        temp directory, is converted, and that temp directory is deleted. The shared
        zip itself lives at :attr:`zip_cache_dir` (or
        :data:`DEFAULT_STREAMING_ZIP_CACHE_DIR` — a fixed subdirectory of the system
        temp root) so multiple calls (and multiple Ray workers) reuse the same zip
        rather than re-downloading 44 GB each.

        :param log_name: Log identifier (must be in :data:`PANDASET_LOG_NAMES`).
        :param output_dir: Destination root (typically a :class:`tempfile.TemporaryDirectory`).
        :return: Path to the extracted log directory ``output_dir/{log_name}``.
        """
        target_dir = self.zip_cache_dir if self.zip_cache_dir is not None else DEFAULT_STREAMING_ZIP_CACHE_DIR
        zip_path = ensure_pandaset_zip_at(target_dir=target_dir, hf_token=self.hf_token, revision=self.revision)
        return extract_pandaset_log(zip_path=zip_path, log_name=log_name, output_dir=Path(output_dir))
