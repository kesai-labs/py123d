from py123d.datatypes.metadata.base_metadata import BaseMetadata
from py123d.datatypes.metadata.cache_source import CacheSourceInfo, CacheSourceModality
from py123d.datatypes.metadata.map_metadata import MapMetadata
from py123d.datatypes.metadata.scene_metadata import SceneMetadata

__all__ = [
    "BaseMetadata",
    "CacheSourceInfo",
    "CacheSourceModality",
    "LogMetadata",
    "MapMetadata",
    "RouteMetadata",
    "SceneMetadata",
]


def __getattr__(name: str):
    """Lazy imports to avoid circular dependencies: LogMetadata pulls in custom_modality,
    RouteMetadata pulls in base_modality — both of which import this package."""
    if name == "LogMetadata":
        from py123d.datatypes.metadata.log_metadata import LogMetadata

        return LogMetadata
    if name == "RouteMetadata":
        from py123d.datatypes.metadata.route_metadata import RouteMetadata

        return RouteMetadata
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
