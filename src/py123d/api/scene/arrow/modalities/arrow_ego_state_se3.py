from pathlib import Path
from typing import Literal, Optional

import pyarrow as pa

from py123d.api.scene.arrow.modalities.base_modality import BaseModalityWriter
from py123d.api.scene.arrow.modalities.utils import all_columns_in_schema, get_optional_array_mixin
from py123d.api.utils.arrow_metadata_utils import add_metadata_to_arrow_schema
from py123d.datatypes.time.timestamp import Timestamp
from py123d.datatypes.vehicle_state.dynamic_state import DynamicStateSE3, DynamicStateSE3Index
from py123d.datatypes.vehicle_state.ego_metadata import EgoStateSE3Metadata
from py123d.datatypes.vehicle_state.ego_state import EgoStateSE3
from py123d.geometry.geometry_index import PoseSE3Index
from py123d.geometry.pose import PoseSE3


class ArrowEgoStateSE3Writer(BaseModalityWriter):
    def __init__(
        self,
        log_dir: Path,
        metadata: EgoStateSE3Metadata,
        ipc_compression: Optional[Literal["lz4", "zstd"]] = None,
        ipc_compression_level: Optional[int] = None,
    ) -> None:
        assert isinstance(metadata, EgoStateSE3Metadata), f"Expected EgoStateSE3Metadata, got {type(metadata)}"

        self._modality_metadata = metadata
        self._modality_name = metadata.modality_name

        file_path = log_dir / f"{metadata.modality_name}.arrow"
        schema = pa.schema(
            [
                (f"{metadata.modality_name}.timestamp_us", pa.int64()),
                (f"{metadata.modality_name}.imu_se3", pa.list_(pa.float64(), len(PoseSE3Index))),
                (f"{metadata.modality_name}.dynamic_state_se3", pa.list_(pa.float64(), len(DynamicStateSE3Index))),
            ]
        )
        schema = add_metadata_to_arrow_schema(schema, metadata)
        super().__init__(
            file_path=file_path,
            schema=schema,
            ipc_compression=ipc_compression,
            ipc_compression_level=ipc_compression_level,
            max_batch_size=1000,
        )

    def write_modality(self, ego_state_se3: EgoStateSE3):
        assert isinstance(ego_state_se3, EgoStateSE3), f"Expected EgoStateSE3, got {type(ego_state_se3)}"
        self.write_batch(
            {
                f"{self._modality_name}.timestamp_us": [ego_state_se3.timestamp.time_us],
                f"{self._modality_name}.imu_se3": [ego_state_se3.imu_se3],
                f"{self._modality_name}.dynamic_state_se3": [ego_state_se3.dynamic_state_se3],
            }
        )


def get_ego_state_se3_from_arrow_table(
    modality_table: pa.Table,
    index: int,
    modality_metadata: Optional[EgoStateSE3Metadata],
) -> Optional[EgoStateSE3]:
    ego_columns = ["ego_state_se3.imu_se3", "ego_state_se3.dynamic_state_se3", "ego_state_se3.timestamp_us"]
    ego_state_se3: Optional[EgoStateSE3] = None
    if all_columns_in_schema(modality_table, ego_columns) and modality_metadata is not None:
        timestamp = Timestamp.from_us(modality_table["ego_state_se3.timestamp_us"][index].as_py())
        imu_se3 = PoseSE3.from_list(modality_table["ego_state_se3.imu_se3"][index].as_py())
        dynamic_state_se3 = get_optional_array_mixin(
            modality_table["ego_state_se3.dynamic_state_se3"][index].as_py(),
            DynamicStateSE3,
        )
        ego_state_se3 = EgoStateSE3.from_imu(
            imu_se3=imu_se3,
            ego_metadata=modality_metadata,
            dynamic_state_se3=dynamic_state_se3,  # type: ignore
            timestamp=timestamp,
        )
    return ego_state_se3
