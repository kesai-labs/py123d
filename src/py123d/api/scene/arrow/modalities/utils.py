from typing import List, Optional, Type, Union

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from py123d.common.utils.mixin import ArrayMixin


def get_optional_array_mixin(data: Optional[Union[List, npt.NDArray]], cls: Type[ArrayMixin]) -> Optional[ArrayMixin]:
    """Builds an optional ArrayMixin if data is provided.

    :param data: The data to convert into an ArrayMixin.
    :param cls: The ArrayMixin class to instantiate.
    :raises ValueError: If the data type is unsupported.
    :return: The instantiated ArrayMixin, or None if data is None.
    """
    if data is None:
        return None
    if isinstance(data, list):
        return cls.from_list(data)
    elif isinstance(data, np.ndarray):
        return cls.from_array(data, copy=False)
    else:
        raise ValueError(f"Unsupported data type for ArrayMixin conversion: {type(data)}")


def all_columns_in_schema(arrow_table: pa.Table, columns: List[str]) -> bool:
    """Checks if all specified columns are present in the Arrow table schema.

    :param arrow_table: The Arrow table to check.
    :param columns: The list of column names to check for.
    :return: True if all columns are present, False otherwise.
    """
    return all(column in arrow_table.schema.names for column in columns)
