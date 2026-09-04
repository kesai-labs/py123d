from xml.etree.ElementTree import fromstring

from py123d.parser.opendrive.utils.collection import _collect_signal_phases
from py123d.parser.opendrive.xodr_parser.opendrive import XODR, Control, Controller, Header, Junction, JunctionController


def _xodr(controllers, junctions):
    return XODR(header=Header(), roads=[], controllers=controllers, junctions=junctions)


def test_junction_controller_parse_reads_sequence():
    element = fromstring('<junction id="20" name="j"><controller id="480" type="0" sequence="2"/></junction>')
    junction = Junction.parse(element)
    assert junction.controllers == [JunctionController(id=480, type="0", sequence=2)]


def test_signal_phases_follow_junction_controller_sequence():
    controllers = [
        Controller(name="a", id=480, sequence=0, controls=[Control(signal_id="457", type=""), Control(signal_id="457", type="")]),
        Controller(name="b", id=481, sequence=1, controls=[Control(signal_id="456", type="")]),
        Controller(name="c", id=490, sequence=0, controls=[Control(signal_id="999", type="")]),
    ]
    junctions = [
        Junction(id=20, name="j20", connections=[], controllers=[JunctionController(480, "0", 0), JunctionController(481, "0", 1)]),
        Junction(id=76, name="j76", connections=[], controllers=[JunctionController(490, "0", 0)]),
    ]
    assert _collect_signal_phases(_xodr(controllers, junctions)) == {457: (20, 0), 456: (20, 1), 999: (76, 0)}


def test_signal_referenced_twice_keeps_first_phase():
    controllers = [
        Controller(name="a", id=1, sequence=0, controls=[Control(signal_id="5", type="")]),
        Controller(name="b", id=2, sequence=2, controls=[Control(signal_id="5", type="")]),
    ]
    junctions = [Junction(id=9, name="j", connections=[], controllers=[JunctionController(1, "0", 0), JunctionController(2, "0", 2)])]
    assert _collect_signal_phases(_xodr(controllers, junctions)) == {5: (9, 0)}
