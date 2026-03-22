def mps_to_kmph(mps: float) -> float:
    """
    Convert meters per second to kilometers per hour.
    :param mps: meter per second [m/s]
    :return: kilometers per hour [km/h]
    """
    return mps * 3.6


def kmph_to_mps(kmph: float) -> float:
    """
    Convert kilometers per hour to meters per second.
    :param kmph: kilometers per hour [km/h]
    :return: meters per second [m/s]
    """
    return kmph / 3.6


def mph_to_mps(mph: float) -> float:
    """
    Convert miles per hour to meters per second.
    :param mph: miles per hour [mi/h]
    :return: meters per second [m/s]
    """
    return mph * 0.44704


def mps_to_mph(mps: float) -> float:
    """
    Convert meters per second to miles per hour.
    :param mps: meters per second [m/s]
    :return: miles per hour [mi/h]
    """
    return mps / 0.44704
