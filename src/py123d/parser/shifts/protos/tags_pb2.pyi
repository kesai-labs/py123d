from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TrajectoryTag(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    kMoveLeft: _ClassVar[TrajectoryTag]
    kMoveRight: _ClassVar[TrajectoryTag]
    kMoveForward: _ClassVar[TrajectoryTag]
    kMoveBack: _ClassVar[TrajectoryTag]
    kAcceleration: _ClassVar[TrajectoryTag]
    kDeceleration: _ClassVar[TrajectoryTag]
    kUniform: _ClassVar[TrajectoryTag]
    kStopping: _ClassVar[TrajectoryTag]
    kStarting: _ClassVar[TrajectoryTag]
    kStationary: _ClassVar[TrajectoryTag]

class DayTimeTag(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    _kUnusedDayTime: _ClassVar[DayTimeTag]
    kNight: _ClassVar[DayTimeTag]
    kMorning: _ClassVar[DayTimeTag]
    kAfternoon: _ClassVar[DayTimeTag]
    kEvening: _ClassVar[DayTimeTag]

class SeasonTag(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    _kUnusedSeason: _ClassVar[SeasonTag]
    kWinter: _ClassVar[SeasonTag]
    kSpring: _ClassVar[SeasonTag]
    kSummer: _ClassVar[SeasonTag]
    kAutumn: _ClassVar[SeasonTag]

class SunPhaseTag(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    _kUnusedDaylight: _ClassVar[SunPhaseTag]
    kAstronomicalNight: _ClassVar[SunPhaseTag]
    kTwilight: _ClassVar[SunPhaseTag]
    kDaylight: _ClassVar[SunPhaseTag]

class Track(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    _kUnusedTrack: _ClassVar[Track]
    Moscow: _ClassVar[Track]
    Skolkovo: _ClassVar[Track]
    Innopolis: _ClassVar[Track]
    AnnArbor: _ClassVar[Track]
    Modiin: _ClassVar[Track]
    TelAviv: _ClassVar[Track]

class Precipitation(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    _kUnusedPrecipitation: _ClassVar[Precipitation]
    kNoPrecipitation: _ClassVar[Precipitation]
    kRain: _ClassVar[Precipitation]
    kSleet: _ClassVar[Precipitation]
    kSnow: _ClassVar[Precipitation]
kMoveLeft: TrajectoryTag
kMoveRight: TrajectoryTag
kMoveForward: TrajectoryTag
kMoveBack: TrajectoryTag
kAcceleration: TrajectoryTag
kDeceleration: TrajectoryTag
kUniform: TrajectoryTag
kStopping: TrajectoryTag
kStarting: TrajectoryTag
kStationary: TrajectoryTag
_kUnusedDayTime: DayTimeTag
kNight: DayTimeTag
kMorning: DayTimeTag
kAfternoon: DayTimeTag
kEvening: DayTimeTag
_kUnusedSeason: SeasonTag
kWinter: SeasonTag
kSpring: SeasonTag
kSummer: SeasonTag
kAutumn: SeasonTag
_kUnusedDaylight: SunPhaseTag
kAstronomicalNight: SunPhaseTag
kTwilight: SunPhaseTag
kDaylight: SunPhaseTag
_kUnusedTrack: Track
Moscow: Track
Skolkovo: Track
Innopolis: Track
AnnArbor: Track
Modiin: Track
TelAviv: Track
_kUnusedPrecipitation: Precipitation
kNoPrecipitation: Precipitation
kRain: Precipitation
kSleet: Precipitation
kSnow: Precipitation

class SceneTags(_message.Message):
    __slots__ = ("day_time", "season", "track", "sun_phase", "precipitation")
    DAY_TIME_FIELD_NUMBER: _ClassVar[int]
    SEASON_FIELD_NUMBER: _ClassVar[int]
    TRACK_FIELD_NUMBER: _ClassVar[int]
    SUN_PHASE_FIELD_NUMBER: _ClassVar[int]
    PRECIPITATION_FIELD_NUMBER: _ClassVar[int]
    day_time: DayTimeTag
    season: SeasonTag
    track: Track
    sun_phase: SunPhaseTag
    precipitation: Precipitation
    def __init__(self, day_time: _Optional[_Union[DayTimeTag, str]] = ..., season: _Optional[_Union[SeasonTag, str]] = ..., track: _Optional[_Union[Track, str]] = ..., sun_phase: _Optional[_Union[SunPhaseTag, str]] = ..., precipitation: _Optional[_Union[Precipitation, str]] = ...) -> None: ...
