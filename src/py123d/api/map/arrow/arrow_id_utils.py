from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import pandas as pd


@dataclass
class IntIDMapping:
    """Class to map string IDs to integer IDs and vice versa."""

    str_to_int: Dict[str, int]

    def __post_init__(self):
        self.int_to_str = {v: k for k, v in self.str_to_int.items()}

    @classmethod
    def from_series(cls, series: pd.Series) -> IntIDMapping:
        """Creates an IntIDMapping from a pandas Series of string-like IDs."""

        # Drop NaN values and convert all to strings
        unique_ids = series.dropna().astype(str).unique()
        str_to_int = {str_id: idx for idx, str_id in enumerate(unique_ids)}
        return IntIDMapping(str_to_int)

    @classmethod
    def from_list(cls, list_: List[Any]) -> IntIDMapping:
        """Creates an IntIDMapping from a pandas Series of string-like IDs."""

        # Drop NaN values and convert all to strings
        ids = [str(v) for v in list_]
        series = pd.Series(ids)
        unique_ids = series.dropna().astype(str).unique()
        str_to_int = {str_id: idx for idx, str_id in enumerate(unique_ids)}
        return IntIDMapping(str_to_int)

    def map(self, str_like: Any) -> Optional[int]:
        """Maps a string-like ID to its corresponding integer ID."""

        # NOTE: We need to convert a string-like input to an integer ID
        if pd.isna(str_like) or str_like is None:
            return None

        if isinstance(str_like, float):
            key = str(int(str_like))  # Convert float to int first to avoid decimal point
        else:
            key = str(str_like)

        return self.str_to_int.get(key, None)

    def map_list(self, id_list: Optional[List[str]]) -> List[int]:
        """Maps a list of string-like IDs to their corresponding integer IDs."""
        if id_list is None:
            return []
        list_ = []
        for id_str in id_list:
            mapped_id = self.map(id_str)
            if mapped_id is not None:
                list_.append(mapped_id)
        return list_


@dataclass
class ToIntMapping:
    """Class to map string IDs to integer IDs and vice versa."""

    str_to_int: Dict[str, int]

    def __post_init__(self):
        self.int_to_id = {v: k for k, v in self.str_to_int.items()}

    @classmethod
    def from_list(cls, list_: List[Union[str, int]]) -> ToIntMapping:
        """Creates a ToIntMapping from a list of string or int IDs."""

        id_to_int = {}
        next_available_int = 0
        used_ints = set()

        # First pass: map IDs that can be parsed as integers
        ids_to_remap = []
        for id_ in list_:
            original_int_id = parse_int_id(id_)
            if original_int_id is not None:
                id_to_int[str(original_int_id)] = int(original_int_id)
                used_ints.add(original_int_id)
                next_available_int = max(next_available_int, original_int_id + 1)
            else:
                ids_to_remap.append(id_)

        # Second pass: map remaining IDs to available integers
        for id_ in ids_to_remap:
            while next_available_int in used_ints:
                next_available_int += 1
            id_to_int[str(id_)] = int(next_available_int)
            used_ints.add(next_available_int)
            next_available_int += 1

        return ToIntMapping(id_to_int)

    def map(self, str_like: Any) -> Optional[int]:
        """Maps a string-like ID to its corresponding integer ID."""

        if isinstance(str_like, float):
            key = str(int(str_like))  # Convert float to int first to avoid decimal point
        elif isinstance(str_like, int):
            key = str(str_like)
        else:
            key = str(str_like)

        return self.str_to_int.get(key, None)

    def map_list(self, id_list: Optional[List[str]]) -> List[int]:
        """Maps a list of string-like IDs to their corresponding integer IDs."""
        if id_list is None:
            return []
        list_ = []
        for id_str in id_list:
            mapped_id = self.map(id_str)
            if mapped_id is not None:
                list_.append(mapped_id)
        return list_


def parse_int_id(id_value: Union[int, str]) -> Optional[int]:
    """Convert id to int if it's a numeric string, otherwise return as-is or raise error."""
    if isinstance(id_value, float):
        return int(id_value)
    if isinstance(id_value, int):
        return id_value
    if isinstance(id_value, str):
        try:
            return int(id_value)
        except ValueError:
            return None
    raise TypeError(f"ID must be int or string, got {type(id_value)}")
