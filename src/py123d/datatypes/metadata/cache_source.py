from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from py123d.datatypes.metadata.base_metadata import BaseMetadata


@dataclass(frozen=True)
class CacheSourceModality(BaseMetadata):
    """A modality of the same log that a derived modality was computed from.

    :param modality_key: Modality key, e.g. ``"ego_state_se3"``.
    :param columns: Column names the computation read, without the modality prefix.
    :param hash: Hash of those columns, see ``hash_modality_columns``.
    """

    modality_key: str
    columns: List[str]
    hash: str

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> CacheSourceModality:
        """Inherited, see superclass."""
        return cls(
            modality_key=data_dict["modality_key"],
            columns=[str(column) for column in data_dict["columns"]],
            hash=data_dict["hash"],
        )

    def to_dict(self) -> Dict[str, Any]:
        """Inherited, see superclass."""
        return {"modality_key": self.modality_key, "columns": list(self.columns), "hash": self.hash}


@dataclass(frozen=True)
class CacheSourceInfo(BaseMetadata):
    """What a derived modality was computed from, stored in its modality metadata.

    Source modalities of the same log are hashed, so a reader can detect that they changed
    after the derived modality was written. External sources (maps, routing services,
    checkpoints) are only listed, not checked.

    :param computed_by: ``"<namespace>:<name>@<version>"``, e.g. ``"py123d:route_from_ego_state_se3@1"``.
    :param computed_at: ISO-8601 UTC timestamp.
    :param source_modalities: Source modalities of the same log.
    :param external_sources: Descriptions of sources outside the log.
    """

    computed_by: str
    computed_at: str
    source_modalities: List[CacheSourceModality]
    external_sources: Optional[List[str]] = None

    def __post_init__(self) -> None:
        assert ":" in self.computed_by and "@" in self.computed_by, (
            f"computed_by must be '<namespace>:<name>@<version>', got {self.computed_by!r}."
        )

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> CacheSourceInfo:
        """Inherited, see superclass."""
        return cls(
            computed_by=data_dict["computed_by"],
            computed_at=data_dict["computed_at"],
            source_modalities=[CacheSourceModality.from_dict(entry) for entry in data_dict["source_modalities"]],
            external_sources=data_dict.get("external_sources"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Inherited, see superclass."""
        return {
            "computed_by": self.computed_by,
            "computed_at": self.computed_at,
            "source_modalities": [source.to_dict() for source in self.source_modalities],
            "external_sources": self.external_sources,
        }
