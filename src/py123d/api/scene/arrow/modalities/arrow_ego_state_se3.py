from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional
from warnings import warn

import numpy as np
import pyarrow as pa

from py123d.api.scene.arrow.modalities.arrow_base import ArrowBaseModalityReader, ArrowBaseModalityWriter
from py123d.api.scene.arrow.modalities.utils import all_columns_in_schema, get_optional_array_mixin
from py123d.api.utils.arrow_metadata_utils import add_metadata_to_arrow_schema
from py123d.datatypes.modalities.base_modality import BaseModality, BaseModalityMetadata
from py123d.datatypes.time.timestamp import Timestamp
from py123d.datatypes.vehicle_state.dynamic_state import DynamicStateSE3, DynamicStateSE3Index
from py123d.datatypes.vehicle_state.ego_state import EgoStateSE3
from py123d.datatypes.vehicle_state.ego_state_metadata import EgoStateSE3Metadata
from py123d.datatypes.vehicle_state.ego_uncertainty import EgoQualityFlag, EgoUncertaintySE3, EgoUncertaintySE3Index
from py123d.geometry.geometry_index import PoseSE3Index
from py123d.geometry.pose import PoseSE3
from py123d.geometry.utils.kinematics_se3 import (
    angular_velocity_body,
    linear_acceleration_global,
    linear_velocity_global,
    rotate_to_body,
)
from py123d.geometry.vector import Vector3D

# Written only when EgoStateSE3Metadata.has_uncertainty is set, so logs whose poses carry no
# estimated uncertainty keep their original schema and still load.
_UNCERTAINTY_FIELDS: tuple = (
    ("uncertainty_se3", pa.list_(pa.float64(), len(EgoUncertaintySE3Index))),
    ("quality_flags", pa.int32()),
    ("fix_age_s", pa.float64()),
)

# ------------------------------------------------------------------------------------------------------------------
# Writer
# ------------------------------------------------------------------------------------------------------------------


class ArrowEgoStateSE3Writer(ArrowBaseModalityWriter):
    def __init__(
        self,
        log_dir: Path,
        metadata: BaseModalityMetadata,
        ipc_compression: Optional[Literal["lz4", "zstd"]] = None,
        ipc_compression_level: Optional[int] = None,
        infer_ego_dynamics: bool = False,
    ) -> None:
        assert isinstance(metadata, EgoStateSE3Metadata), f"Expected EgoStateSE3Metadata, got {type(metadata)}"

        self._metadata = metadata

        # Optional: For inferring ego dynamics from pose.
        self._infer_ego_dynamics = infer_ego_dynamics
        self._inference_window: List[EgoStateSE3] = []
        self._deferred_count: int = 0  # frames accepted but not yet emitted

        file_path = log_dir / f"{metadata.modality_key}.arrow"
        fields = [
            (f"{self._metadata.modality_key}.timestamp_us", pa.int64()),
            (f"{self._metadata.modality_key}.imu_se3", pa.list_(pa.float64(), len(PoseSE3Index))),
            (f"{self._metadata.modality_key}.dynamic_state_se3", pa.list_(pa.float64(), len(DynamicStateSE3Index))),
        ]
        if metadata.has_uncertainty:
            fields.extend((f"{self._metadata.modality_key}.{name}", dtype) for name, dtype in _UNCERTAINTY_FIELDS)
        schema = add_metadata_to_arrow_schema(pa.schema(fields), metadata)
        super().__init__(
            file_path=file_path,
            schema=schema,
            ipc_compression=ipc_compression,
            ipc_compression_level=ipc_compression_level,
            max_batch_size=1000,
        )

    @property
    def row_count(self) -> int:
        """Row count including frames accepted but not yet emitted."""
        return self._row_count + self._deferred_count

    def write_modality(self, modality: BaseModality) -> None:
        assert isinstance(modality, EgoStateSE3), f"Expected EgoStateSE3, got {type(modality)}"
        if not self._infer_ego_dynamics:
            self._emit(modality)
            return

        self._inference_window.append(modality)
        self._deferred_count += 1
        if len(self._inference_window) == 2:
            # First frame: emit with forward difference against the second frame. The buffer is kept
            # intact so the second frame will later be emitted with a centered difference once a third
            # frame arrives.
            first, second = self._inference_window
            self._emit(_with_dynamics(first, _infer_ego_dynamics(prev=None, curr=first, nxt=second)))
            self._deferred_count -= 1
        elif len(self._inference_window) == 3:
            # Middle frame: centered difference, then drop the oldest.
            prev, curr, nxt = self._inference_window
            self._emit(_with_dynamics(curr, _infer_ego_dynamics(prev=prev, curr=curr, nxt=nxt)))
            self._inference_window.pop(0)
            self._deferred_count -= 1

    def close(self) -> None:
        if self._infer_ego_dynamics and self._inference_window:
            if len(self._inference_window) == 1:
                only = self._inference_window[0]
                self._emit(_with_dynamics(only, _zero_dynamic_state_se3()))
            else:
                # The penultimate frame in the buffer has already been emitted; flush the tail with a
                # backward difference against it.
                prev = self._inference_window[0]
                last = self._inference_window[-1]
                self._emit(_with_dynamics(last, _infer_ego_dynamics(prev=prev, curr=last, nxt=None)))
            self._inference_window.clear()
        super().close()

    def _emit(self, modality: EgoStateSE3) -> None:
        key = self._metadata.modality_key
        row: Dict[str, Any] = {
            f"{key}.timestamp_us": [modality.timestamp.time_us],
            f"{key}.imu_se3": [modality.imu_se3],
            f"{key}.dynamic_state_se3": [modality.dynamic_state_se3],
        }
        if self._metadata.has_uncertainty:
            # inf is not representable in Arrow's float64 in a way consumers handle uniformly, and
            # "no fix yet" is genuinely absent rather than infinitely old, so it is written null.
            fix_age = modality.fix_age_s
            row[f"{key}.uncertainty_se3"] = [modality.uncertainty_se3]
            row[f"{key}.quality_flags"] = [None if modality.quality_flags is None else int(modality.quality_flags)]
            row[f"{key}.fix_age_s"] = [None if fix_age is None or not np.isfinite(fix_age) else float(fix_age)]
        elif modality.uncertainty_se3 is not None:
            warn(
                "Ego state carries uncertainty but EgoStateSE3Metadata.has_uncertainty is False; "
                "the values would be silently dropped.",
                category=UserWarning,
            )
        self.write_batch(row)


def _zero_dynamic_state_se3() -> DynamicStateSE3:
    return DynamicStateSE3(
        velocity=Vector3D(0.0, 0.0, 0.0),
        acceleration=Vector3D(0.0, 0.0, 0.0),
        angular_velocity=Vector3D(0.0, 0.0, 0.0),
    )


def _with_dynamics(state: EgoStateSE3, dynamic_state: DynamicStateSE3) -> EgoStateSE3:
    return EgoStateSE3.from_imu(
        imu_se3=state.imu_se3,
        metadata=state.metadata,
        timestamp=state.timestamp,
        dynamic_state_se3=dynamic_state,
        tire_steering_angle=state.tire_steering_angle if state.tire_steering_angle is not None else 0.0,
        uncertainty_se3=state.uncertainty_se3,
        quality_flags=state.quality_flags,
        fix_age_s=state.fix_age_s,
    )


def _timestamp_delta_seconds(t1: Timestamp, t2: Timestamp) -> float:
    return (t2.time_us - t1.time_us) / 1e6


def _infer_ego_dynamics(
    prev: Optional[EgoStateSE3],
    curr: EgoStateSE3,
    nxt: Optional[EgoStateSE3],
) -> DynamicStateSE3:
    if prev is None and nxt is None:
        return _zero_dynamic_state_se3()

    xyz_curr = np.asarray(curr.imu_se3.array[PoseSE3Index.XYZ], dtype=np.float64)
    rotation_curr = curr.imu_se3.rotation_matrix

    if prev is not None and nxt is not None:
        xyz_prev = np.asarray(prev.imu_se3.array[PoseSE3Index.XYZ], dtype=np.float64)
        xyz_next = np.asarray(nxt.imu_se3.array[PoseSE3Index.XYZ], dtype=np.float64)
        dt_prev = _timestamp_delta_seconds(prev.timestamp, curr.timestamp)
        dt_next = _timestamp_delta_seconds(curr.timestamp, nxt.timestamp)

        dt_total = dt_prev + dt_next
        if dt_prev <= 0.0 or dt_next <= 0.0:
            warn("Non-positive dt in ego dynamics inference; emitting zero dynamics.", category=UserWarning)
            return _zero_dynamic_state_se3()
        velocity_global = linear_velocity_global(xyz_prev, xyz_next, dt_total)
        acceleration_global = linear_acceleration_global(xyz_prev, xyz_curr, xyz_next, dt_prev, dt_next)
        angular_velocity = angular_velocity_body(prev.imu_se3.rotation_matrix, nxt.imu_se3.rotation_matrix, dt_total)
    elif nxt is not None:
        xyz_next = np.asarray(nxt.imu_se3.array[PoseSE3Index.XYZ], dtype=np.float64)
        dt = _timestamp_delta_seconds(curr.timestamp, nxt.timestamp)
        if dt <= 0.0:
            warn("Non-positive dt in ego dynamics inference; emitting zero dynamics.", category=UserWarning)
            return _zero_dynamic_state_se3()
        velocity_global = linear_velocity_global(xyz_curr, xyz_next, dt)
        acceleration_global = np.zeros(3, dtype=np.float64)
        angular_velocity = angular_velocity_body(rotation_curr, nxt.imu_se3.rotation_matrix, dt)
    else:
        assert prev is not None
        xyz_prev = np.asarray(prev.imu_se3.array[PoseSE3Index.XYZ], dtype=np.float64)
        dt = _timestamp_delta_seconds(prev.timestamp, curr.timestamp)
        if dt <= 0.0:
            warn("Non-positive dt in ego dynamics inference; emitting zero dynamics.", category=UserWarning)
            return _zero_dynamic_state_se3()
        velocity_global = linear_velocity_global(xyz_prev, xyz_curr, dt)
        acceleration_global = np.zeros(3, dtype=np.float64)
        angular_velocity = angular_velocity_body(prev.imu_se3.rotation_matrix, rotation_curr, dt)

    velocity_body = rotate_to_body(velocity_global, rotation_curr)
    acceleration_body = rotate_to_body(acceleration_global, rotation_curr)
    return DynamicStateSE3(
        velocity=Vector3D(float(velocity_body[0]), float(velocity_body[1]), float(velocity_body[2])),
        acceleration=Vector3D(float(acceleration_body[0]), float(acceleration_body[1]), float(acceleration_body[2])),
        angular_velocity=Vector3D(float(angular_velocity[0]), float(angular_velocity[1]), float(angular_velocity[2])),
    )


# ------------------------------------------------------------------------------------------------------------------
# Reader
# ------------------------------------------------------------------------------------------------------------------


class ArrowEgoStateSE3Reader(ArrowBaseModalityReader):
    """Stateless reader for ego state SE3 data from Arrow tables."""

    @staticmethod
    def read_at_index(
        index: int,
        table: pa.Table,
        metadata: BaseModalityMetadata,
        dataset: str,
        **kwargs,
    ) -> Optional[EgoStateSE3]:
        assert isinstance(metadata, EgoStateSE3Metadata)
        return _deserialize_ego_state_se3(table, index, metadata)

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
        """Return a single column value from the ego state Arrow table at a given row index.

        :param index: The row index in the Arrow table.
        :param table: The Arrow modality table.
        :param metadata: The modality metadata.
        :param column: The field name (e.g. ``"imu_se3"``, ``"timestamp_us"``).
        :param deserialize: If True, deserialize the value to its domain type.
        :return: The column value, or None if the column is not present.
        """
        full_column_name = f"{metadata.modality_key}.{column}"
        column_at_iteration: Optional[Any] = None
        if full_column_name in table.column_names:
            column_at_iteration = table[full_column_name][index].as_py()
            deserializers = {**EGO_STATE_SE3_DESERIALIZE_FUNC, **EGO_STATE_SE3_OPTIONAL_DESERIALIZE_FUNC}
            if deserialize and column in deserializers:
                column_at_iteration = deserializers[column](column_at_iteration)
        else:
            raise ValueError(
                f"Column '{full_column_name}' not found in Arrow table for modality '{metadata.modality_key}'"
            )

        return column_at_iteration


EGO_STATE_SE3_DESERIALIZE_FUNC: Dict[str, Callable[[Any], Any]] = {
    "imu_se3": PoseSE3.from_list,
    "dynamic_state_se3": lambda v: get_optional_array_mixin(data=v, cls=DynamicStateSE3),
    "timestamp_us": Timestamp.from_us,
}

# Deserializers for the optional columns. Kept apart from the required ones above, which decide
# whether a row can be read at all.
EGO_STATE_SE3_OPTIONAL_DESERIALIZE_FUNC: Dict[str, Callable[[Any], Any]] = {
    "uncertainty_se3": lambda v: get_optional_array_mixin(data=v, cls=EgoUncertaintySE3),
    "quality_flags": lambda v: None if v is None else EgoQualityFlag(int(v)),
    "fix_age_s": lambda v: None if v is None else float(v),
}


def _deserialize_ego_state_se3(
    modality_table: pa.Table,
    index: int,
    metadata: EgoStateSE3Metadata,
) -> Optional[EgoStateSE3]:
    """Deserialize an ego state from Arrow table columns at the given row index."""

    modality_key = metadata.modality_key
    ego_columns = [f"{modality_key}.{field}" for field in EGO_STATE_SE3_DESERIALIZE_FUNC.keys()]
    if not all_columns_in_schema(modality_table, ego_columns):
        return None

    timestamp = EGO_STATE_SE3_DESERIALIZE_FUNC["timestamp_us"](
        modality_table[f"{modality_key}.timestamp_us"][index].as_py()
    )
    imu_se3 = EGO_STATE_SE3_DESERIALIZE_FUNC["imu_se3"](modality_table[f"{modality_key}.imu_se3"][index].as_py())
    dynamic_state_se3 = EGO_STATE_SE3_DESERIALIZE_FUNC["dynamic_state_se3"](
        modality_table[f"{modality_key}.dynamic_state_se3"][index].as_py()
    )
    optional = {
        name: deserialize(modality_table[f"{modality_key}.{name}"][index].as_py())
        if f"{modality_key}.{name}" in modality_table.column_names
        else None
        for name, deserialize in EGO_STATE_SE3_OPTIONAL_DESERIALIZE_FUNC.items()
    }
    return EgoStateSE3.from_imu(
        imu_se3=imu_se3,
        metadata=metadata,
        dynamic_state_se3=dynamic_state_se3,  # type: ignore
        timestamp=timestamp,
        uncertainty_se3=optional["uncertainty_se3"],  # type: ignore
        quality_flags=optional["quality_flags"],  # type: ignore
        fix_age_s=optional["fix_age_s"],  # type: ignore
    )
