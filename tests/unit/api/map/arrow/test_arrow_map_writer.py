"""Stress tests for ArrowMapWriter and _map_ids_to_integer."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import shapely.geometry as geom

from py123d.api.map.arrow.arrow_map_writer import ArrowMapWriter, _map_ids_to_integer
from py123d.datatypes import MapMetadata
from py123d.datatypes.map_objects.base_map_objects import BaseMapObject
from py123d.datatypes.map_objects.map_layer_types import MapLayer
from py123d.geometry import Polyline2D, Polyline3D

from ..conftest import (
    make_carpark,
    make_crosswalk,
    make_generic_drivable,
    make_intersection,
    make_lane,
    make_lane_group,
    make_road_edge,
    make_road_line,
    make_stop_zone,
    make_walkway,
)


def _empty_map_data() -> Dict[MapLayer, Dict[str, Any]]:
    """Build an empty map_data dict with all 10 MapLayer keys."""
    return {layer: defaultdict(list) for layer in MapLayer}


def _per_log_metadata() -> MapMetadata:
    return MapMetadata(
        dataset="test",
        location="boston",
        map_has_z=True,
        map_is_per_log=True,
        split="test_train",
        log_name="log_001",
    )


def _global_metadata() -> MapMetadata:
    return MapMetadata(
        dataset="test",
        location="boston",
        map_has_z=True,
        map_is_per_log=False,
    )


def _make_writer(tmp_path: Path, force: bool = True) -> ArrowMapWriter:
    maps_root = tmp_path / "maps"
    logs_root = tmp_path / "logs"
    maps_root.mkdir(exist_ok=True)
    logs_root.mkdir(exist_ok=True)
    return ArrowMapWriter(force_map_conversion=force, maps_root=maps_root, logs_root=logs_root)


# ---------------------------------------------------------------------------
# ArrowMapWriter.reset()
# ---------------------------------------------------------------------------


class TestArrowMapWriterReset:
    """Tests for ArrowMapWriter.reset() path resolution and gating logic."""

    def test_per_log_writes_to_logs_root(self, tmp_path: Path) -> None:
        """Per-log metadata targets logs_root/split/log_name/map.arrow."""
        writer = _make_writer(tmp_path)
        metadata = _per_log_metadata()
        result = writer.reset(metadata)

        assert result is True

        # Write a minimal object and close to create the file
        writer.write_map_object(make_crosswalk(object_id="cw_1"))
        writer.close()

        expected_file = tmp_path / "logs" / "test_train" / "log_001" / "map.arrow"
        assert expected_file.exists()

    def test_global_writes_to_maps_root(self, tmp_path: Path) -> None:
        """Global metadata targets maps_root/dataset/dataset_location.arrow."""
        writer = _make_writer(tmp_path)
        metadata = _global_metadata()
        result = writer.reset(metadata)

        assert result is True

        writer.write_map_object(make_crosswalk(object_id="cw_1"))
        writer.close()

        expected_file = tmp_path / "maps" / "test" / "test_boston.arrow"
        assert expected_file.exists()

    def test_returns_true_if_file_does_not_exist(self, tmp_path: Path) -> None:
        """reset() returns True when the target file does not yet exist."""
        writer = _make_writer(tmp_path, force=False)
        result = writer.reset(_per_log_metadata())
        assert result is True

    def test_returns_false_if_file_exists_and_no_force(self, tmp_path: Path) -> None:
        """reset() returns False when file exists and force_map_conversion=False."""
        # First write to create the file
        writer = _make_writer(tmp_path, force=True)
        writer.reset(_per_log_metadata())
        writer.write_map_object(make_crosswalk(object_id="cw_1"))
        writer.close()

        # Second reset with force=False should return False
        writer2 = _make_writer(tmp_path, force=False)
        result = writer2.reset(_per_log_metadata())
        assert result is False

    def test_returns_true_if_file_exists_and_force(self, tmp_path: Path) -> None:
        """reset() returns True when file exists but force_map_conversion=True."""
        writer = _make_writer(tmp_path, force=True)
        writer.reset(_per_log_metadata())
        writer.write_map_object(make_crosswalk(object_id="cw_1"))
        writer.close()

        writer2 = _make_writer(tmp_path, force=True)
        result = writer2.reset(_per_log_metadata())
        assert result is True

    def test_per_log_asserts_on_missing_split(self, tmp_path: Path) -> None:
        """Per-log metadata with split=None triggers AssertionError.

        Note: MapMetadata.__init__ validates this, so the assertion happens
        during metadata construction, not during reset(). This validates that
        the invariant is enforced somewhere in the pipeline.
        """
        with pytest.raises(AssertionError, match="split must be provided"):
            MapMetadata(
                dataset="test",
                location="boston",
                map_has_z=True,
                map_is_per_log=True,
                split=None,
                log_name="log_001",
            )

    def test_per_log_asserts_on_missing_log_name(self, tmp_path: Path) -> None:
        """Per-log metadata with log_name=None triggers AssertionError."""
        with pytest.raises(AssertionError, match="log_name must be provided"):
            MapMetadata(
                dataset="test",
                location="boston",
                map_has_z=True,
                map_is_per_log=True,
                split="train",
                log_name=None,
            )

    def test_global_asserts_on_missing_location(self, tmp_path: Path) -> None:
        """Global metadata with location=None triggers AssertionError."""
        writer = _make_writer(tmp_path)
        metadata = MapMetadata(
            dataset="test",
            location=None,
            map_has_z=True,
            map_is_per_log=False,
        )
        with pytest.raises(AssertionError, match="location must be provided"):
            writer.reset(metadata)


# ---------------------------------------------------------------------------
# ArrowMapWriter.write_map_object()
# ---------------------------------------------------------------------------


class TestArrowMapWriterWriteMapObject:
    """Tests for write_map_object dispatch and error handling."""

    def test_dispatches_all_supported_types(self, tmp_path: Path) -> None:
        """All 10 known map object types are accepted without error."""
        writer = _make_writer(tmp_path)
        writer.reset(_per_log_metadata())

        objects = [
            make_lane(object_id="l_1", lane_group_id="lg_1"),
            make_lane_group(object_id="lg_1", lane_ids=["l_1"]),
            make_intersection(object_id="i_1", lane_group_ids=["lg_1"]),
            make_crosswalk(object_id="cw_1"),
            make_walkway(object_id="w_1"),
            make_carpark(object_id="cp_1"),
            make_generic_drivable(object_id="gd_1"),
            make_stop_zone(object_id="sz_1", lane_ids=["l_1"]),
            make_road_edge(object_id="re_1"),
            make_road_line(object_id="rl_1"),
        ]
        for obj in objects:
            writer.write_map_object(obj)

        writer.close()

    def test_raises_value_error_for_unsupported_type(self, tmp_path: Path) -> None:
        """An unknown BaseMapObject subclass raises ValueError."""

        class UnknownMapObject(BaseMapObject):
            @property
            def layer(self) -> MapLayer:
                return MapLayer.LANE

        writer = _make_writer(tmp_path)
        writer.reset(_per_log_metadata())

        with pytest.raises(ValueError, match="Unsupported map object type"):
            writer.write_map_object(UnknownMapObject(object_id="fake"))

    def test_assert_initialized_raises_without_reset(self, tmp_path: Path) -> None:
        """write_map_object before reset() raises AssertionError."""
        writer = _make_writer(tmp_path)

        with pytest.raises(AssertionError, match="Call reset"):
            writer.write_map_object(make_crosswalk(object_id="cw_1"))


# ---------------------------------------------------------------------------
# ArrowMapWriter._write_surface_layer() — outline handling
# ---------------------------------------------------------------------------


class TestWriteSurfaceLayer:
    """Tests for outline inference and zero-padding in _write_surface_layer."""

    def test_polyline3d_outline_stored_as_is(self, tmp_path: Path) -> None:
        """A surface object with Polyline3D outline stores the array unchanged."""
        writer = _make_writer(tmp_path)
        writer.reset(_per_log_metadata())

        crosswalk = make_crosswalk(object_id="cw_1")
        assert isinstance(crosswalk.outline, Polyline3D)

        writer.write_map_object(crosswalk)

        stored_outline = writer._map_data[MapLayer.CROSSWALK]["outline"][0]
        np.testing.assert_array_equal(stored_outline, crosswalk.outline.array)

    def test_polyline2d_outline_zero_padded(self, tmp_path: Path) -> None:
        """A surface object with Polyline2D outline gets zero-padded to 3 columns."""
        coords_2d = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=float)
        outline_2d = Polyline2D.from_array(coords_2d)
        polygon = geom.Polygon(coords_2d)

        crosswalk = make_crosswalk(object_id="cw_2d")
        # Override outline to be 2D
        crosswalk._outline = outline_2d
        crosswalk._shapely_polygon = polygon

        writer = _make_writer(tmp_path)
        writer.reset(_per_log_metadata())
        writer.write_map_object(crosswalk)

        stored = writer._map_data[MapLayer.CROSSWALK]["outline"][0]
        assert stored.shape[1] == 3
        np.testing.assert_array_equal(stored[:, 2], np.zeros(len(coords_2d)))
        np.testing.assert_array_equal(stored[:, :2], coords_2d)

    def test_shapely_polygon_2d_inferred_and_zero_padded(self, tmp_path: Path) -> None:
        """When outline is None but shapely_polygon is 2D, exterior is zero-padded to 3 columns."""
        polygon_2d = geom.Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

        crosswalk = make_crosswalk(object_id="cw_infer")
        # Set outline to None and shapely_polygon to a 2D polygon to trigger the else branch
        crosswalk._outline = None
        crosswalk._shapely_polygon = polygon_2d

        writer = _make_writer(tmp_path)
        writer.reset(_per_log_metadata())
        writer.write_map_object(crosswalk)

        stored = writer._map_data[MapLayer.CROSSWALK]["outline"][0]
        assert stored.shape[1] == 3
        np.testing.assert_array_equal(stored[:, 2], np.zeros(len(stored)))

    def test_shapely_polygon_3d_inferred_as_is(self, tmp_path: Path) -> None:
        """When outline is None but shapely_polygon has 3D exterior, coords are stored directly."""
        polygon_3d = geom.Polygon([(0, 0, 1), (10, 0, 2), (10, 10, 3), (0, 10, 4)])

        crosswalk = make_crosswalk(object_id="cw_3d_infer")
        crosswalk._outline = None
        crosswalk._shapely_polygon = polygon_3d

        writer = _make_writer(tmp_path)
        writer.reset(_per_log_metadata())
        writer.write_map_object(crosswalk)

        stored = writer._map_data[MapLayer.CROSSWALK]["outline"][0]
        assert stored.shape[1] == 3
        # Z values should be non-zero for the non-closing vertices
        assert np.any(stored[:, 2] != 0)


# ---------------------------------------------------------------------------
# ArrowMapWriter.close()
# ---------------------------------------------------------------------------


class TestArrowMapWriterClose:
    """Tests for close() file creation and state reset."""

    def test_creates_file_and_parent_dirs(self, tmp_path: Path) -> None:
        """close() creates the arrow file and any missing parent directories."""
        writer = _make_writer(tmp_path)
        writer.reset(_per_log_metadata())
        writer.write_map_object(make_crosswalk(object_id="cw_1"))
        writer.close()

        expected = tmp_path / "logs" / "test_train" / "log_001" / "map.arrow"
        assert expected.exists()

    def test_resets_internal_state_after_close(self, tmp_path: Path) -> None:
        """After close(), internal state is reset so a new cycle can begin."""
        writer = _make_writer(tmp_path)
        writer.reset(_per_log_metadata())
        writer.write_map_object(make_crosswalk(object_id="cw_1"))
        writer.close()

        assert writer._map_file is None
        assert writer._map_data == {}
        assert writer._map_metadata is None

    def test_double_close_does_not_crash(self, tmp_path: Path) -> None:
        """Calling close() twice does not raise an exception."""
        writer = _make_writer(tmp_path)
        writer.reset(_per_log_metadata())
        writer.write_map_object(make_crosswalk(object_id="cw_1"))
        writer.close()
        # Second close should be a no-op
        writer.close()

    def test_close_without_objects_does_not_crash(self, tmp_path: Path) -> None:
        """close() after reset() with no objects written does not crash."""
        writer = _make_writer(tmp_path)
        writer.reset(_per_log_metadata())
        # Close without writing any objects -- the file may or may not be created
        # but it should not raise
        writer.close()


# ---------------------------------------------------------------------------
# _map_ids_to_integer()
# ---------------------------------------------------------------------------


class TestMapIdsToInteger:
    """Stress tests for the standalone _map_ids_to_integer function."""

    def test_basic_cross_layer_remapping(self) -> None:
        """IDs are remapped to integers across related layers."""
        map_data = _empty_map_data()

        # Lane referencing lane_group "lg_A"
        map_data[MapLayer.LANE]["id"].append("lane_A")
        map_data[MapLayer.LANE]["lane_group_id"].append("lg_A")
        map_data[MapLayer.LANE]["left_lane_id"].append(None)
        map_data[MapLayer.LANE]["right_lane_id"].append(None)
        map_data[MapLayer.LANE]["predecessor_ids"].append([])
        map_data[MapLayer.LANE]["successor_ids"].append([])

        # Lane group "lg_A" with lane_ids ["lane_A"], referencing intersection "int_X"
        map_data[MapLayer.LANE_GROUP]["id"].append("lg_A")
        map_data[MapLayer.LANE_GROUP]["lane_ids"].append(["lane_A"])
        map_data[MapLayer.LANE_GROUP]["intersection_id"].append("int_X")
        map_data[MapLayer.LANE_GROUP]["predecessor_ids"].append([])
        map_data[MapLayer.LANE_GROUP]["successor_ids"].append([])

        # Intersection "int_X" with lane_group_ids ["lg_A"]
        map_data[MapLayer.INTERSECTION]["id"].append("int_X")
        map_data[MapLayer.INTERSECTION]["lane_group_ids"].append(["lg_A"])

        _map_ids_to_integer(map_data)

        # All IDs should now be integers
        lane_id = map_data[MapLayer.LANE]["id"][0]
        lg_id = map_data[MapLayer.LANE_GROUP]["id"][0]
        int_id = map_data[MapLayer.INTERSECTION]["id"][0]

        assert isinstance(lane_id, int)
        assert isinstance(lg_id, int)
        assert isinstance(int_id, int)

        # Cross-references should match
        assert map_data[MapLayer.LANE]["lane_group_id"][0] == lg_id
        assert map_data[MapLayer.LANE_GROUP]["lane_ids"][0] == [lane_id]
        assert map_data[MapLayer.LANE_GROUP]["intersection_id"][0] == int_id
        assert map_data[MapLayer.INTERSECTION]["lane_group_ids"][0] == [lg_id]

    def test_stop_zone_lane_ids_remapped(self) -> None:
        """Stop zone lane_ids are remapped using the lane ID mapping."""
        map_data = _empty_map_data()

        map_data[MapLayer.LANE]["id"].append("L1")
        map_data[MapLayer.LANE]["lane_group_id"].append(None)
        map_data[MapLayer.LANE]["left_lane_id"].append(None)
        map_data[MapLayer.LANE]["right_lane_id"].append(None)
        map_data[MapLayer.LANE]["predecessor_ids"].append([])
        map_data[MapLayer.LANE]["successor_ids"].append([])

        map_data[MapLayer.STOP_ZONE]["id"].append("sz_1")
        map_data[MapLayer.STOP_ZONE]["lane_ids"].append(["L1"])

        _map_ids_to_integer(map_data)

        lane_int_id = map_data[MapLayer.LANE]["id"][0]
        assert map_data[MapLayer.STOP_ZONE]["lane_ids"][0] == [lane_int_id]

    def test_empty_layers(self) -> None:
        """All layers empty -- function completes without error."""
        map_data = _empty_map_data()
        _map_ids_to_integer(map_data)

        # All ID lists should remain empty
        for layer in MapLayer:
            assert len(map_data[layer]["id"]) == 0

    def test_all_empty_map(self) -> None:
        """Completely empty map data with no keys besides defaultdict defaults."""
        map_data = _empty_map_data()
        _map_ids_to_integer(map_data)
        # Should not raise

    def test_lane_references_nonexistent_lane_group_becomes_none(self) -> None:
        """When a lane's lane_group_id doesn't exist in LANE_GROUP, it maps to None."""
        map_data = _empty_map_data()

        map_data[MapLayer.LANE]["id"].append("lane_orphan")
        map_data[MapLayer.LANE]["lane_group_id"].append("nonexistent_lg")
        map_data[MapLayer.LANE]["left_lane_id"].append(None)
        map_data[MapLayer.LANE]["right_lane_id"].append(None)
        map_data[MapLayer.LANE]["predecessor_ids"].append([])
        map_data[MapLayer.LANE]["successor_ids"].append([])

        _map_ids_to_integer(map_data)

        # The nonexistent lane_group_id should become None
        assert map_data[MapLayer.LANE]["lane_group_id"][0] is None

    def test_lane_group_references_nonexistent_intersection_becomes_none(self) -> None:
        """When a lane_group's intersection_id doesn't exist in INTERSECTION, it maps to None."""
        map_data = _empty_map_data()

        map_data[MapLayer.LANE_GROUP]["id"].append("lg_orphan")
        map_data[MapLayer.LANE_GROUP]["lane_ids"].append([])
        map_data[MapLayer.LANE_GROUP]["intersection_id"].append("nonexistent_int")
        map_data[MapLayer.LANE_GROUP]["predecessor_ids"].append([])
        map_data[MapLayer.LANE_GROUP]["successor_ids"].append([])

        _map_ids_to_integer(map_data)

        assert map_data[MapLayer.LANE_GROUP]["intersection_id"][0] is None

    def test_stop_zone_references_nonexistent_lanes_list_becomes_empty(self) -> None:
        """When a stop_zone's lane_ids reference nonexistent lanes, the list becomes empty."""
        map_data = _empty_map_data()

        map_data[MapLayer.STOP_ZONE]["id"].append("sz_orphan")
        map_data[MapLayer.STOP_ZONE]["lane_ids"].append(["no_such_lane_1", "no_such_lane_2"])

        _map_ids_to_integer(map_data)

        assert map_data[MapLayer.STOP_ZONE]["lane_ids"][0] == []

    def test_predecessor_successor_nonexistent_lanes_shrink(self) -> None:
        """predecessor_ids/successor_ids referencing nonexistent lanes are filtered out."""
        map_data = _empty_map_data()

        map_data[MapLayer.LANE]["id"].append("lane_real")
        map_data[MapLayer.LANE]["lane_group_id"].append(None)
        map_data[MapLayer.LANE]["left_lane_id"].append(None)
        map_data[MapLayer.LANE]["right_lane_id"].append(None)
        map_data[MapLayer.LANE]["predecessor_ids"].append(["lane_real", "lane_ghost"])
        map_data[MapLayer.LANE]["successor_ids"].append(["lane_phantom"])

        _map_ids_to_integer(map_data)

        lane_int = map_data[MapLayer.LANE]["id"][0]
        # Only "lane_real" should survive in predecessor_ids
        assert map_data[MapLayer.LANE]["predecessor_ids"][0] == [lane_int]
        # "lane_phantom" doesn't exist, so successor_ids should be empty
        assert map_data[MapLayer.LANE]["successor_ids"][0] == []

    def test_none_values_in_id_fields_preserved(self) -> None:
        """None values in left_lane_id and right_lane_id remain None after remapping."""
        map_data = _empty_map_data()

        map_data[MapLayer.LANE]["id"].append("lane_solo")
        map_data[MapLayer.LANE]["lane_group_id"].append(None)
        map_data[MapLayer.LANE]["left_lane_id"].append(None)
        map_data[MapLayer.LANE]["right_lane_id"].append(None)
        map_data[MapLayer.LANE]["predecessor_ids"].append([])
        map_data[MapLayer.LANE]["successor_ids"].append([])

        _map_ids_to_integer(map_data)

        assert map_data[MapLayer.LANE]["left_lane_id"][0] is None
        assert map_data[MapLayer.LANE]["right_lane_id"][0] is None
        assert map_data[MapLayer.LANE]["lane_group_id"][0] is None

    def test_left_right_lane_ids_remapped(self) -> None:
        """left_lane_id and right_lane_id are correctly remapped to integer IDs."""
        map_data = _empty_map_data()

        map_data[MapLayer.LANE]["id"].extend(["lane_L", "lane_C", "lane_R"])
        map_data[MapLayer.LANE]["lane_group_id"].extend([None, None, None])
        map_data[MapLayer.LANE]["left_lane_id"].extend([None, "lane_L", "lane_C"])
        map_data[MapLayer.LANE]["right_lane_id"].extend(["lane_C", "lane_R", None])
        map_data[MapLayer.LANE]["predecessor_ids"].extend([[], [], []])
        map_data[MapLayer.LANE]["successor_ids"].extend([[], [], []])

        _map_ids_to_integer(map_data)

        ids = map_data[MapLayer.LANE]["id"]
        left_ids = map_data[MapLayer.LANE]["left_lane_id"]
        right_ids = map_data[MapLayer.LANE]["right_lane_id"]

        # lane_C is left neighbor of lane_L => left_lane_id[1] == ids[0]
        assert left_ids[1] == ids[0]
        # lane_R is right neighbor of lane_C => right_lane_id[1] == ids[2]
        assert right_ids[1] == ids[2]
        # Nones stay None
        assert left_ids[0] is None
        assert right_ids[2] is None

    def test_lane_group_predecessor_successor_remapped(self) -> None:
        """Lane group predecessor/successor IDs are remapped via the lane_group mapping."""
        map_data = _empty_map_data()

        map_data[MapLayer.LANE_GROUP]["id"].extend(["lgA", "lgB"])
        map_data[MapLayer.LANE_GROUP]["lane_ids"].extend([[], []])
        map_data[MapLayer.LANE_GROUP]["intersection_id"].extend([None, None])
        map_data[MapLayer.LANE_GROUP]["predecessor_ids"].extend([[], ["lgA"]])
        map_data[MapLayer.LANE_GROUP]["successor_ids"].extend([["lgB"], []])

        _map_ids_to_integer(map_data)

        lgA_int = map_data[MapLayer.LANE_GROUP]["id"][0]
        lgB_int = map_data[MapLayer.LANE_GROUP]["id"][1]

        assert map_data[MapLayer.LANE_GROUP]["predecessor_ids"][1] == [lgA_int]
        assert map_data[MapLayer.LANE_GROUP]["successor_ids"][0] == [lgB_int]

    def test_intersection_lane_group_ids_with_nonexistent_shrink(self) -> None:
        """Intersection lane_group_ids referencing nonexistent groups are filtered out."""
        map_data = _empty_map_data()

        map_data[MapLayer.LANE_GROUP]["id"].append("lg_real")
        map_data[MapLayer.LANE_GROUP]["lane_ids"].append([])
        map_data[MapLayer.LANE_GROUP]["intersection_id"].append(None)
        map_data[MapLayer.LANE_GROUP]["predecessor_ids"].append([])
        map_data[MapLayer.LANE_GROUP]["successor_ids"].append([])

        map_data[MapLayer.INTERSECTION]["id"].append("int_1")
        map_data[MapLayer.INTERSECTION]["lane_group_ids"].append(["lg_real", "lg_ghost"])

        _map_ids_to_integer(map_data)

        lg_int = map_data[MapLayer.LANE_GROUP]["id"][0]
        # Only lg_real should survive
        assert map_data[MapLayer.INTERSECTION]["lane_group_ids"][0] == [lg_int]

    def test_simple_layers_remapped_to_integers(self) -> None:
        """Standalone layers (crosswalk, walkway, carpark, etc.) get integer IDs."""
        map_data = _empty_map_data()

        map_data[MapLayer.CROSSWALK]["id"].append("cw_alpha")
        map_data[MapLayer.WALKWAY]["id"].append("wk_beta")
        map_data[MapLayer.CARPARK]["id"].append("cp_gamma")
        map_data[MapLayer.GENERIC_DRIVABLE]["id"].append("gd_delta")
        map_data[MapLayer.ROAD_LINE]["id"].append("rl_epsilon")
        map_data[MapLayer.ROAD_EDGE]["id"].append("re_zeta")

        _map_ids_to_integer(map_data)

        for layer in [
            MapLayer.CROSSWALK,
            MapLayer.WALKWAY,
            MapLayer.CARPARK,
            MapLayer.GENERIC_DRIVABLE,
            MapLayer.ROAD_LINE,
            MapLayer.ROAD_EDGE,
        ]:
            assert isinstance(map_data[layer]["id"][0], int)

    def test_multiple_objects_per_layer(self) -> None:
        """Multiple objects in a single layer all receive unique integer IDs."""
        map_data = _empty_map_data()

        for i in range(5):
            map_data[MapLayer.CROSSWALK]["id"].append(f"cw_{i}")

        _map_ids_to_integer(map_data)

        int_ids = map_data[MapLayer.CROSSWALK]["id"]
        assert len(int_ids) == 5
        assert len(set(int_ids)) == 5  # all unique
        for id_ in int_ids:
            assert isinstance(id_, int)

    def test_integer_string_ids_preserve_value(self) -> None:
        """IDs that are already integers (or parseable as int) keep their numeric value."""
        map_data = _empty_map_data()

        map_data[MapLayer.CROSSWALK]["id"].extend([42, "7", 100])

        _map_ids_to_integer(map_data)

        ids = map_data[MapLayer.CROSSWALK]["id"]
        assert 42 in ids
        assert 7 in ids
        assert 100 in ids
