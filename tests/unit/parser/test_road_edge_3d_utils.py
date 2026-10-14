import numpy as np
import shapely

from py123d.geometry.polyline import Polyline3D
from py123d.parser.utils.map_utils.road_edge.road_edge_3d_utils import (
    _split_at_vertical_seams,
    lift_road_edges_to_3d,
)


def test_lifted_ring_is_split_where_it_steps_between_vertical_layers():
    # A 2D union ring whose bottom run follows a road at z=0 and whose right run follows a bridge at z=10;
    # the other two runs have no boundary nearby and are dropped.
    ring = shapely.LinearRing(shapely.box(0.0, 0.0, 20.0, 10.0).exterior.segmentize(1.0).coords)
    road_boundary = Polyline3D.from_array(np.array([[-5.0, 0.0, 0.0], [25.0, 0.0, 0.0]]))
    bridge_boundary = Polyline3D.from_array(np.array([[20.0, -5.0, 10.0], [20.0, 15.0, 10.0]]))

    road_edges = lift_road_edges_to_3d([ring], [road_boundary, bridge_boundary])

    assert len(road_edges) == 2
    for road_edge in road_edges:
        z_values = road_edge.array[:, 2]
        assert np.ptp(z_values) < 1e-6, "an edge must stay on one vertical layer"
    lengths_by_level = {round(float(edge.array[0, 2])): edge.length for edge in road_edges}
    assert lengths_by_level[0] >= 19.0
    assert lengths_by_level[10] >= 9.0


def test_split_keeps_ramps_and_cuts_vertical_seams():
    ramp = Polyline3D.from_array(
        np.column_stack([np.linspace(0.0, 150.0, 31), np.zeros(31), np.linspace(0.0, 10.0, 31)])
    )
    seam = Polyline3D.from_array(np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.01, 0.0, 10.0], [10.0, 10.0, 10.0]]))

    assert len(_split_at_vertical_seams([ramp])) == 1
    pieces = _split_at_vertical_seams([seam])
    assert [len(piece.array) for piece in pieces] == [2, 2]
    assert pieces[0].array[-1, 2] == 0.0 and pieces[1].array[0, 2] == 10.0
