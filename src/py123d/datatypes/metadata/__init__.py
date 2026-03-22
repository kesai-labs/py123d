from py123d.datatypes.metadata.base_metadata import BaseMetadata
from py123d.datatypes.metadata.map_metadata import MapMetadata
from py123d.datatypes.metadata.scene_metadata import SceneMetadata

__all__ = [
    "BaseMetadata",
    "LogMetadata",
    "MapMetadata",
    "SceneMetadata",
]


def __getattr__(name: str):
    """Lazy import for LogMetadata to avoid circular dependency with custom_modality."""
    if name == "LogMetadata":
        from py123d.datatypes.metadata.log_metadata import LogMetadata

        return LogMetadata
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
