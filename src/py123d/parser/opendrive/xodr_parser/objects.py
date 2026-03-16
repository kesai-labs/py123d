from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
from xml.etree.ElementTree import Element


@dataclass
class XODRObject:
    """
    https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/latest/specification/13_objects/13_01_introduction.html
    """

    id: int
    name: str
    s: float
    t: float
    z_offset: float
    hdg: float
    roll: float
    pitch: float
    orientation: str
    type: str
    width: Optional[float]
    length: Optional[float]

    outline: Optional[List[CornerLocal]]

    def __post_init__(self):
        # NOTE: added assertion/filtering to check for element type or consistency
        pass

    @classmethod
    def parse(cls, object_element: Optional[Element]) -> XODRObject:
        args = {}
        args["id"] = int(object_element.get("id"))
        args["name"] = object_element.get("name")
        args["s"] = float(object_element.get("s"))
        args["t"] = float(object_element.get("t"))
        args["z_offset"] = float(object_element.get("zOffset"))
        args["hdg"] = float(object_element.get("hdg"))
        args["roll"] = float(object_element.get("roll"))
        args["pitch"] = float(object_element.get("pitch"))
        args["orientation"] = object_element.get("orientation")
        args["type"] = object_element.get("type")
        args["width"] = float(object_element.get("width")) if object_element.get("width") is not None else None
        args["length"] = float(object_element.get("length")) if object_element.get("length") is not None else None

        outline: List[CornerLocal] = []
        if object_element.find("outline") is not None:
            for corner_element in object_element.find("outline").findall("cornerLocal"):
                outline.append(CornerLocal.parse(corner_element))
        args["outline"] = outline

        return XODRObject(**args)


@dataclass
class CornerLocal:
    """
    https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/latest/specification/13_objects/13_03_object_outline.html#sec-cc00c8a6-eea6-49e6-b90c-37b21524c748
    """

    u: float
    v: float
    z: float
    height: Optional[float] = None

    @classmethod
    def parse(cls, corner_element: Optional[Element]) -> CornerLocal:
        args = {}
        args["u"] = float(corner_element.get("u"))
        args["v"] = float(corner_element.get("v"))
        args["z"] = float(corner_element.get("z"))
        if corner_element.get("height") is not None:
            args["height"] = float(corner_element.get("height"))
        return CornerLocal(**args)
