from __future__ import annotations

import gzip
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional
from xml.etree.ElementTree import Element, parse

from py123d.parser.opendrive.xodr_parser.road import XODRRoad


@dataclass
class XODR:
    """
    https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/latest/specification/06_general_architecture/06_03_root_element.html
    """

    header: Header

    roads: List[XODRRoad]
    controllers: List[Controller]
    junctions: List[Junction]

    @classmethod
    def parse(cls, root_element: Element) -> XODR:
        args = {}
        args["header"] = Header.parse(root_element.find("header"))

        roads: List[XODRRoad] = []
        for road_element in root_element.findall("road"):
            try:
                roads.append(XODRRoad.parse(road_element))
            except Exception as e:
                print(
                    f"Error parsing road element with id/name {road_element.get('id')}/{road_element.get('name')}: {e}"
                )
                traceback.print_exc()
        args["roads"] = roads

        controllers: List[Controller] = []
        for controller_element in root_element.findall("controller"):
            controllers.append(Controller.parse(controller_element))
        args["controllers"] = controllers

        junctions: List[Junction] = []
        for junction_element in root_element.findall("junction"):
            junctions.append(Junction.parse(junction_element))
        args["junctions"] = junctions

        return XODR(**args)

    @classmethod
    def parse_from_file(cls, file_path: Path) -> XODR:
        if file_path.suffix == ".gz":
            with gzip.open(file_path, "rb") as f:
                tree = parse(f)
        else:
            tree = parse(file_path)
        return XODR.parse(tree.getroot())


@dataclass
class Header:
    """
    https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/latest/specification/06_general_architecture/06_04_header.html
    """

    rev_major: Optional[int] = None
    rev_minor: Optional[int] = None
    name: Optional[str] = None
    version: Optional[str] = None
    data: Optional[str] = None
    north: Optional[float] = None
    south: Optional[float] = None
    east: Optional[float] = None
    west: Optional[float] = None
    vendor: Optional[str] = None
    geo_reference: Optional[str] = None

    @classmethod
    def parse(cls, header_element: Optional[Element]) -> Header:
        """
        :param header_element: XML element containing the OpenDrive header.
        :return: instance of OpenDrive header dataclass.
        """
        args = {}
        if header_element is not None:
            args["rev_major"] = header_element.get("rev_major")
            args["rev_minor"] = header_element.get("rev_minor")
            args["name"] = header_element.get("name")
            args["version"] = header_element.get("version")
            args["data"] = header_element.get("data")
            args["north"] = float(header_element.get("north"))
            args["south"] = float(header_element.get("south"))
            args["east"] = float(header_element.get("east"))
            args["west"] = float(header_element.get("west"))
            args["vendor"] = header_element.get("vendor")
            if header_element.find("geoReference") is not None:
                args["geo_reference"] = header_element.find("geoReference").text

        return Header(**args)


@dataclass
class Controller:
    """
    https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/latest/specification/14_signals/14_06_controllers.html
    """

    name: str
    id: int
    sequence: int
    controls: List[Control]

    @classmethod
    def parse(cls, controller_element: Optional[Element]) -> Junction:
        args = {}
        args["name"] = controller_element.get("name")
        args["id"] = float(controller_element.get("id"))
        args["sequence"] = float(controller_element.get("sequence"))

        controls: List[Control] = []
        for control_element in controller_element.findall("control"):
            controls.append(Control.parse(control_element))
        args["controls"] = controls

        return Controller(**args)


@dataclass
class Control:
    """
    https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/latest/specification/14_signals/14_06_controllers.html
    """

    signal_id: str
    type: str

    @classmethod
    def parse(cls, control_element: Optional[Element]) -> Control:
        args = {}
        args["signal_id"] = control_element.get("signalId")
        args["type"] = control_element.get("type")
        return Control(**args)


@dataclass
class Junction:
    """
    https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/latest/specification/12_junctions/12_02_common_junctions.html
    """

    id: int
    name: str
    connections: List[Connection]
    controllers: List[JunctionController] = field(default_factory=list)

    @classmethod
    def parse(cls, junction_element: Optional[Element]) -> Junction:
        args = {}

        args["id"] = int(junction_element.get("id"))
        args["name"] = junction_element.get("name")

        connections: List[Connection] = []
        for connection_element in junction_element.findall("connection"):
            connections.append(Connection.parse(connection_element))
        args["connections"] = connections

        controllers: List[JunctionController] = []
        for controller_element in junction_element.findall("controller"):
            controllers.append(JunctionController.parse(controller_element))
        args["controllers"] = controllers

        return Junction(**args)


@dataclass
class JunctionController:
    """
    Controller reference inside a junction; sequence is the signal phase index of the junction's cycle.
    https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/latest/specification/12_junctions/12_02_common_junctions.html
    """

    id: int
    type: Optional[str]
    sequence: int

    @classmethod
    def parse(cls, controller_element: Element) -> JunctionController:
        return JunctionController(
            id=int(controller_element.get("id")),
            type=controller_element.get("type"),
            sequence=int(controller_element.get("sequence", 0)),
        )


@dataclass
class Connection:
    """
    https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/latest/specification/12_junctions/12_02_common_junctions.html
    """

    id: int
    incoming_road: int
    connecting_road: int
    contact_point: Literal["start", "end"]
    lane_links: List[LaneLink]

    def __post_init__(self):
        assert self.contact_point in ["start", "end"]

    @classmethod
    def parse(cls, connection_element: Optional[Element]) -> Connection:
        args = {}

        args["id"] = int(connection_element.get("id"))
        args["incoming_road"] = int(connection_element.get("incomingRoad"))
        args["connecting_road"] = int(connection_element.get("connectingRoad"))
        args["contact_point"] = connection_element.get("contactPoint")

        lane_links: List[LaneLink] = []
        for lane_link_element in connection_element.findall("laneLink"):
            lane_links.append(LaneLink.parse(lane_link_element))
        args["lane_links"] = lane_links

        return Connection(**args)


@dataclass
class LaneLink:
    """
    https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/1.8.0/specification/12_junctions/12_04_connecting_roads.html#top-3e9bb97e-f2ab-4751-906a-c25e9fb7ac4e
    """

    start: int  # NOTE: named "from" in xml
    end: int  # NOTE: named "to" in xml

    @classmethod
    def parse(cls, lane_link_element: Optional[Element]) -> LaneLink:
        args = {}
        args["start"] = int(lane_link_element.get("from"))
        args["end"] = int(lane_link_element.get("to"))
        return LaneLink(**args)
