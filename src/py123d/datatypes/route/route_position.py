from py123d.datatypes.metadata.route_metadata import RouteMetadata
from py123d.datatypes.modalities.base_modality import BaseModality
from py123d.datatypes.time.timestamp import Timestamp


class RoutePosition(BaseModality):
    """The ego's arc-length position on the log's route polyline at one frame.

    The position is the coordinate into ``route.arrow``'s arc-length parameterization:
    interpolating the polyline at ``progress_m + d`` yields the point d meters ahead
    along the route.
    """

    __slots__ = ("_progress_m", "_timestamp", "_metadata")

    def __init__(self, progress_m: float, timestamp: Timestamp, metadata: RouteMetadata):
        """Initializes the :class:`RoutePosition`.

        :param progress_m: Arc-length position on the route polyline in meters.
        :param timestamp: The timestamp of the frame.
        :param metadata: The route metadata.
        """
        self._progress_m = progress_m
        self._timestamp = timestamp
        self._metadata = metadata

    @property
    def progress_m(self) -> float:
        """Arc-length position on the route polyline in meters."""
        return self._progress_m

    @property
    def timestamp(self) -> Timestamp:
        """Inherited, see superclass."""
        return self._timestamp

    @property
    def metadata(self) -> RouteMetadata:
        """Inherited, see superclass."""
        return self._metadata
