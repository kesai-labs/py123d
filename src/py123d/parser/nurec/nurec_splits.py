"""Releases and splits of the NuRec dataset, shared by the parser and the downloader.

A split is either a whole release (``nurec-2601_train``, ``nurec-2604_train``) or a list of
named scenes (``nurec-curated_train``, ``nurec-curated_val``, listed in
``nurec_curated_splits.yaml`` next to this module). A scene name is
``"<release>/<scene uuid>"``: some scene uuids exist in more than one release, so the uuid
alone does not identify a scene.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

NUREC_RELEASE_DIR_SUFFIX = "_release"


@dataclass(frozen=True)
class NuRecRelease:
    """Where one NuRec release lives in the Hugging Face dataset."""

    name: str
    hf_sequences_prefix: str
    revision: str


NUREC_RELEASES: Dict[str, NuRecRelease] = {
    "26.01": NuRecRelease(name="26.01", hf_sequences_prefix="sample_set/26.01_release", revision="26.01"),
    "26.04": NuRecRelease(name="26.04", hf_sequences_prefix="sample_set/26.04_release", revision="26.04"),
}

# Splits covering every scene of one release.
NUREC_RELEASE_SPLITS: Dict[str, str] = {
    "nurec-2601_train": "26.01",
    "nurec-2604_train": "26.04",
}

# Converted and downloaded when no splits are requested. Both are scene lists.
NUREC_DEFAULT_SPLITS: Tuple[str, ...] = ("nurec-curated_train", "nurec-curated_val")

_SPLIT_NAME_PATTERN = re.compile(r"^nurec[\w.-]*_(train|val|test)$")


@lru_cache(maxsize=1)
def _load_bundled_scene_lists() -> Dict[str, Tuple[str, ...]]:
    yaml_file = resources.files("py123d.parser.nurec") / "nurec_curated_splits.yaml"
    with yaml_file.open("r", encoding="utf-8") as stream:
        scene_lists = yaml.safe_load(stream)
    return {split: tuple(scene_names) for split, scene_names in scene_lists.items()}


def load_default_scene_lists() -> Dict[str, List[str]]:
    """Loads the bundled train/val scene lists.

    :return: Mapping of split name to scene names, each ``"<release>/<scene uuid>"``.
    """
    return {split: list(scene_names) for split, scene_names in _load_bundled_scene_lists().items()}


def parse_scene_name(scene_name: str) -> Tuple[str, str]:
    """Splits a scene name into its release and scene uuid.

    :param scene_name: Scene name of the form ``"<release>/<scene uuid>"``.
    :return: Tuple of (release, scene uuid).
    """
    release, separator, scene_uuid = str(scene_name).partition("/")
    if not separator or not scene_uuid or "/" in scene_uuid:
        raise ValueError(f"NuRec scene name {scene_name!r} is not of the form '<release>/<scene uuid>'.")
    if release not in NUREC_RELEASES:
        raise ValueError(
            f"NuRec scene name {scene_name!r} names an unknown release. Available: {sorted(NUREC_RELEASES)}"
        )
    return release, scene_uuid


def release_from_path(path: Path) -> Optional[str]:
    """Reads the release a file belongs to from its closest ``<release>_release`` parent directory.

    :param path: Path of a file inside a downloaded NuRec release.
    :return: The release, e.g. ``"26.04"``, or None if no parent directory names one.
    """
    for part in reversed(Path(path).parent.parts):
        if part.endswith(NUREC_RELEASE_DIR_SUFFIX):
            return part[: -len(NUREC_RELEASE_DIR_SUFFIX)]
    return None


def resolve_scene_lists(scene_lists: Optional[Mapping[str, Sequence[str]]]) -> Dict[str, List[str]]:
    """Returns the scene lists to use, falling back to the bundled train/val lists.

    :param scene_lists: Mapping of split name to scene names, or None for the bundled lists.
    :return: Mapping of split name to scene names.
    """
    if scene_lists is None:
        return load_default_scene_lists()
    resolved: Dict[str, List[str]] = {}
    for split, scene_names in scene_lists.items():
        if split in NUREC_RELEASE_SPLITS:
            raise ValueError(f"NuRec split {split!r} covers a whole release and cannot be given a scene list.")
        if not _SPLIT_NAME_PATTERN.match(str(split)):
            raise ValueError(
                f"NuRec split {split!r} is not a valid name. Use 'nurec<...>_train', 'nurec<...>_val' "
                "or 'nurec<...>_test'."
            )
        resolved[str(split)] = [str(scene_name) for scene_name in scene_names]
    return resolved


def validate_splits(splits: Sequence[str], scene_lists: Mapping[str, Sequence[str]]) -> List[str]:
    """Checks that every requested split is a release split or has a scene list.

    :param splits: Requested split names.
    :param scene_lists: Mapping of split name to scene names.
    :return: The requested splits, with duplicates removed and order kept.
    """
    available = sorted(set(NUREC_RELEASE_SPLITS) | set(scene_lists))
    requested: List[str] = []
    for split in splits:
        if split not in NUREC_RELEASE_SPLITS and split not in scene_lists:
            raise ValueError(f"NuRec split {split!r} is not available. Available splits: {available}")
        if split not in requested:
            requested.append(split)
    if not requested:
        raise ValueError(f"No NuRec splits requested. Available splits: {available}")
    return requested


def scenes_of_split(split: str, scene_lists: Mapping[str, Sequence[str]]) -> List[Tuple[str, str]]:
    """Returns the scenes of a split that is defined by a scene list.

    :param split: Name of a split with a scene list.
    :param scene_lists: Mapping of split name to scene names.
    :return: List of (release, scene uuid), in the order of the scene list.
    """
    scenes = [parse_scene_name(scene_name) for scene_name in scene_lists[split]]
    if len(set(scenes)) != len(scenes):
        raise ValueError(f"NuRec split {split!r} lists a scene more than once.")
    return scenes
