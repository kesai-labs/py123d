import abc

from py123d.api.log_writer.abstract_log_writer import AbstractLogWriter
from py123d.api.map_writer.abstract_map_writer import AbstractMapWriter
from py123d.conversion.dataset_converter_config import DatasetConverterConfig


class AbstractDatasetConverter(abc.ABC):
    """Abstract base class for dataset converters.

    A dataset converter for implementing all dataset specific conversion logic.

    """

    def __init__(self, dataset_converter_config: DatasetConverterConfig) -> None:
        self.dataset_converter_config = dataset_converter_config

    @abc.abstractmethod
    def get_number_of_maps(self) -> int:
        """Returns the number of available raw data maps for conversion."""

    @abc.abstractmethod
    def get_number_of_logs(self) -> int:
        """Returns the number of available raw data logs for conversion."""

    @abc.abstractmethod
    def convert_map(self, map_index: int, map_writer: AbstractMapWriter) -> None:
        """
        Convert a single map in raw data format to the uniform 123D format.
        :param map_index: The index of the map to convert.
        :param map_writer: The map writer to use for writing the converted map.
        """

    @abc.abstractmethod
    def convert_log(self, log_index: int, log_writer: AbstractLogWriter) -> None:
        """
        Convert a single log in raw data format to the uniform 123D format.
        :param log_index: The index of the log to convert.
        :param log_writer: The log writer to use for writing the converted log.
        """
