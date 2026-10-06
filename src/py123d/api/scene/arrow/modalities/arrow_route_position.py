from typing import Any, Callable, Dict, Optional

import pyarrow as pa

from py123d.api.scene.arrow.modalities.arrow_base import ArrowBaseModalityReader
from py123d.datatypes.metadata.route_metadata import RouteMetadata
from py123d.datatypes.modalities.base_modality import BaseModalityMetadata
from py123d.datatypes.route.route_position import RoutePosition
from py123d.datatypes.time.timestamp import Timestamp


class ArrowRoutePositionReader(ArrowBaseModalityReader):
    """Stateless reader for route positions from Arrow tables."""

    @staticmethod
    def read_at_index(
        index: int,
        table: pa.Table,
        metadata: BaseModalityMetadata,
        dataset: str,
        **kwargs,
    ) -> Optional[RoutePosition]:
        assert isinstance(metadata, RouteMetadata)
        key = metadata.modality_key
        return RoutePosition(
            progress_m=table[f"{key}.progress_m"][index].as_py(),
            timestamp=Timestamp.from_us(table[f"{key}.timestamp_us"][index].as_py()),
            metadata=metadata,
        )

    @staticmethod
    def read_column_at_index(
        index: int,
        table: pa.Table,
        metadata: BaseModalityMetadata,
        column: str,
        dataset: str,
        deserialize: bool = False,
        **kwargs,
    ) -> Optional[Any]:
        full_column_name = f"{metadata.modality_key}.{column}"
        if full_column_name not in table.column_names:
            raise ValueError(
                f"Column '{full_column_name}' not found in Arrow table for modality '{metadata.modality_key}'"
            )
        value = table[full_column_name][index].as_py()
        if deserialize and column in ROUTE_POSITION_DESERIALIZE_FUNC:
            value = ROUTE_POSITION_DESERIALIZE_FUNC[column](value)
        return value


ROUTE_POSITION_DESERIALIZE_FUNC: Dict[str, Callable[[Any], Any]] = {
    "timestamp_us": Timestamp.from_us,
}
