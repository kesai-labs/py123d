from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, Dict

from py123d.datatypes.metadata.abstract_metadata import AbstractMetadata
from py123d.datatypes.sensors.fisheye_mei_camera import FisheyeMEICameraID, FisheyeMEICameraMetadata
from py123d.datatypes.sensors.lidar import LidarID, LidarMetadata
from py123d.datatypes.sensors.pinhole_camera import PinholeCameraID, PinholeCameraMetadata


class PinholeCameraMetadatas(AbstractMetadata, Mapping[PinholeCameraID, PinholeCameraMetadata]):
    __slots__ = ("_data",)

    def __init__(self, pinhole_camera_metadata_dict: Dict[PinholeCameraID, PinholeCameraMetadata]):
        self._data = pinhole_camera_metadata_dict

    def __getitem__(self, key: PinholeCameraID) -> PinholeCameraMetadata:
        return self._data[key]

    def __iter__(self) -> Iterator[PinholeCameraID]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the metadata instance to a plain Python dictionary.

        :return: A dictionary representation using only default Python types.
        """
        return {str(int(lid)): meta.to_dict() for lid, meta in self._data.items()}

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> PinholeCameraMetadatas:
        """Construct a metadata instance from a plain Python dictionary.

        :param data_dict: A dictionary containing the metadata fields.
        :return: A metadata instance.
        """
        return PinholeCameraMetadatas(
            pinhole_camera_metadata_dict={
                PinholeCameraID(int(k)): PinholeCameraMetadata.from_dict(v) for k, v in data_dict.items()
            }
        )


class FisheyeMEICameraMetadatas(AbstractMetadata, Mapping[FisheyeMEICameraID, FisheyeMEICameraMetadata]):
    __slots__ = ("_data",)

    def __init__(self, fisheye_mei_camera_metadata_dict: Dict[FisheyeMEICameraID, FisheyeMEICameraMetadata]):
        self._data = fisheye_mei_camera_metadata_dict

    def __getitem__(self, key: FisheyeMEICameraID) -> FisheyeMEICameraMetadata:
        return self._data[key]

    def __iter__(self) -> Iterator[FisheyeMEICameraID]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the metadata instance to a plain Python dictionary.

        :return: A dictionary representation using only default Python types.
        """
        return {str(int(lid)): meta.to_dict() for lid, meta in self._data.items()}

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> FisheyeMEICameraMetadatas:
        """Construct a metadata instance from a plain Python dictionary.

        :param data_dict: A dictionary containing the metadata fields.
        :return: A metadata instance.
        """
        return FisheyeMEICameraMetadatas(
            fisheye_mei_camera_metadata_dict={
                FisheyeMEICameraID(int(k)): FisheyeMEICameraMetadata.from_dict(v) for k, v in data_dict.items()
            }
        )


class LidarMetadatas(AbstractMetadata, Mapping[LidarID, LidarMetadata]):
    __slots__ = ("_data",)

    def __init__(self, lidar_metadata_dict: Dict[LidarID, LidarMetadata]):
        self._data = lidar_metadata_dict

    def __getitem__(self, key: LidarID) -> LidarMetadata:
        return self._data[key]

    def __iter__(self) -> Iterator[LidarID]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the metadata instance to a plain Python dictionary.

        :return: A dictionary representation using only default Python types.
        """
        return {str(int(lid)): meta.to_dict() for lid, meta in self._data.items()}

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> LidarMetadatas:
        """Construct a metadata instance from a plain Python dictionary.

        :param data_dict: A dictionary containing the metadata fields.
        :return: A metadata instance.
        """
        return LidarMetadatas(
            lidar_metadata_dict={LidarID(int(k)): LidarMetadata.from_dict(v) for k, v in data_dict.items()}
        )
