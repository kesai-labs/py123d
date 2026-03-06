from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import cached_property
from typing import Final, List, Optional, Union
from xml.etree.ElementTree import Element

import numpy as np
import numpy.typing as npt

from py123d.geometry import Point3DIndex, PoseSE2Index
from py123d.parser.opendrive.xodr_parser.elevation import XODRElevation
from py123d.parser.opendrive.xodr_parser.geometry import XODRArc, XODRGeometry, XODRLine, XODRSpiral
from py123d.parser.opendrive.xodr_parser.lane import XODRLaneOffset, XODRWidth
from py123d.parser.opendrive.xodr_parser.polynomial import XODRPolynomial

TOLERANCE: Final[float] = 1e-3


@dataclass
class XODRPlanView:
    geometries: List[XODRGeometry]

    def __post_init__(self):
        # Ensure geometries are sorted by their starting position 's'
        self.geometries.sort(key=lambda x: x.s)

    @classmethod
    def parse(cls, plan_view_element: Optional[Element]) -> XODRPlanView:
        geometries: List[XODRGeometry] = []
        for geometry_element in plan_view_element.findall("geometry"):
            if geometry_element.find("line") is not None:
                geometry = XODRLine.parse(geometry_element)
            elif geometry_element.find("arc") is not None:
                geometry = XODRArc.parse(geometry_element)
            elif geometry_element.find("spiral") is not None:
                geometry = XODRSpiral.parse(geometry_element)
            else:
                geometry_str = ET.tostring(geometry_element, encoding="unicode")
                raise NotImplementedError(f"Geometry not implemented: {geometry_str}")
            geometries.append(geometry)
        return XODRPlanView(geometries=geometries)

    @cached_property
    def geometry_lengths(self) -> npt.NDArray[np.float64]:
        return np.cumsum([0.0] + [geo.length for geo in self.geometries], dtype=np.float64)

    @property
    def length(self) -> float:
        return float(self.geometry_lengths[-1])

    def interpolate_se2(self, s: float, t: float = 0.0, lane_section_end: bool = False) -> npt.NDArray[np.float64]:
        """
        Interpolates the SE2 state at a given longitudinal position s along the plan view.
        """
        if s > self.length:
            if np.isclose(s, self.length, atol=TOLERANCE):
                s = self.length
            else:
                raise ValueError(
                    f"PlanView: s={s} is beyond the end of the plan view (length={self.length}) with tolerance={TOLERANCE}."
                )

        # Find the geometry segment containing s
        geo_idx = np.searchsorted(self.geometry_lengths, s, side="right") - 1
        geo_idx = int(np.clip(geo_idx, 0, len(self.geometries) - 1))

        return self.geometries[geo_idx].interpolate_se2(s - self.geometry_lengths[geo_idx], t)

    def interpolate_se2_batch(
        self,
        s_arr: npt.NDArray[np.float64],
        t_arr: npt.NDArray[np.float64],
        lane_section_end_arr: npt.NDArray[np.bool_],
    ) -> npt.NDArray[np.float64]:
        s_clamped = s_arr.copy()
        over = s_clamped > self.length
        if np.any(over):
            close = np.isclose(s_clamped, self.length, atol=TOLERANCE)
            if np.any(over & ~close):
                bad = s_clamped[over & ~close][0]
                raise ValueError(
                    f"PlanView: s={bad} is beyond the end of the plan view (length={self.length}) with tolerance={TOLERANCE}."
                )
            s_clamped[over] = self.length

        geo_indices = np.searchsorted(self.geometry_lengths, s_clamped, side="right") - 1
        geo_indices = np.clip(geo_indices, 0, len(self.geometries) - 1)

        result = np.empty((len(s_arr), len(PoseSE2Index)), dtype=np.float64)
        for geo_idx in np.unique(geo_indices):
            mask = geo_indices == geo_idx
            local_s = s_clamped[mask] - self.geometry_lengths[geo_idx]
            local_t = t_arr[mask]
            result[mask] = self.geometries[geo_idx].interpolate_se2_batch(local_s, local_t)
        return result


@dataclass
class XODRReferenceLine:
    reference_line: Union[XODRReferenceLine, XODRPlanView]
    width_polynomials: List[XODRPolynomial]
    elevations: List[XODRElevation]
    s_offset: float

    @property
    def length(self) -> float:
        return float(self.reference_line.length)

    @classmethod
    def from_plan_view(
        cls,
        plan_view: XODRPlanView,
        lane_offsets: List[XODRLaneOffset],
        elevations: List[XODRElevation],
    ) -> XODRReferenceLine:
        args = {}
        args["reference_line"] = plan_view
        args["width_polynomials"] = lane_offsets
        args["elevations"] = elevations
        args["s_offset"] = 0.0
        return XODRReferenceLine(**args)

    @classmethod
    def from_reference_line(
        cls,
        reference_line: XODRReferenceLine,
        widths: List[XODRWidth],
        s_offset: float = 0.0,
        t_sign: float = 1.0,
    ) -> XODRReferenceLine:
        assert t_sign in [1.0, -1.0], "t_sign must be either 1.0 or -1.0"

        args = {}
        args["reference_line"] = reference_line
        width_polynomials: List[XODRPolynomial] = []
        for width in widths:
            width_polynomials.append(width.get_polynomial(t_sign=t_sign))
        args["width_polynomials"] = width_polynomials
        args["s_offset"] = s_offset
        args["elevations"] = reference_line.elevations

        return XODRReferenceLine(**args)

    @staticmethod
    def _find_polynomial(s: float, polynomials: List[XODRPolynomial], lane_section_end: bool = False) -> XODRPolynomial:
        out_polynomial = polynomials[-1]
        for polynomial in polynomials[::-1]:
            if lane_section_end:
                if polynomial.s < s:
                    out_polynomial = polynomial
                    break
            elif polynomial.s <= s:
                out_polynomial = polynomial
                break

        # s_values = np.array([poly.s for poly in polynomials])
        # side = "left" if lane_section_end else "right"
        # poly_idx = np.searchsorted(s_values, s, side=side) - 1
        # poly_idx = int(np.clip(poly_idx, 0, len(polynomials) - 1))
        # return polynomials[poly_idx]
        return out_polynomial

    def interpolate_se2(self, s: float, t: float = 0.0, lane_section_end: bool = False) -> npt.NDArray[np.float64]:
        width_polynomial = self._find_polynomial(s, self.width_polynomials, lane_section_end=lane_section_end)
        t_offset = width_polynomial.get_value(s - width_polynomial.s)
        se2 = self.reference_line.interpolate_se2(self.s_offset + s, t=t_offset + t, lane_section_end=lane_section_end)

        return se2

    def interpolate_3d(self, s: float, t: float = 0.0, lane_section_end: bool = False) -> npt.NDArray[np.float64]:
        se2 = self.interpolate_se2(s, t, lane_section_end=lane_section_end)

        elevation_polynomial = self._find_polynomial(s, self.elevations, lane_section_end=lane_section_end)
        point_3d = np.zeros(len(Point3DIndex), dtype=np.float64)
        point_3d[Point3DIndex.XY] = se2[PoseSE2Index.XY]
        point_3d[Point3DIndex.Z] = elevation_polynomial.get_value(s - elevation_polynomial.s)

        return point_3d

    @staticmethod
    def _eval_single_polynomial_batch(
        s_arr: npt.NDArray[np.float64],
        poly: XODRPolynomial,
    ) -> npt.NDArray[np.float64]:
        ds = s_arr - poly.s
        return poly.a + poly.b * ds + poly.c * ds**2 + poly.d * ds**3

    @staticmethod
    def _find_polynomial_indices_batch(
        s_arr: npt.NDArray[np.float64],
        polynomials: List[XODRPolynomial],
        lane_section_end_arr: npt.NDArray[np.bool_],
    ) -> npt.NDArray[np.int64]:
        if len(polynomials) == 1:
            return np.zeros(len(s_arr), dtype=np.int64)
        poly_s = np.array([p.s for p in polynomials], dtype=np.float64)
        indices = np.searchsorted(poly_s, s_arr, side="right") - 1
        # Fix last point (lane_section_end) if needed
        if lane_section_end_arr[-1]:
            indices[-1] = np.searchsorted(poly_s, s_arr[-1], side="left") - 1
        return np.clip(indices, 0, len(polynomials) - 1)

    @staticmethod
    def _eval_polynomials_batch(
        s_arr: npt.NDArray[np.float64],
        polynomials: List[XODRPolynomial],
        indices: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.float64]:
        if len(polynomials) == 1:
            poly = polynomials[0]
            ds = s_arr - poly.s
            return poly.a + poly.b * ds + poly.c * ds**2 + poly.d * ds**3
        result = np.empty(len(s_arr), dtype=np.float64)
        for poly_idx in np.unique(indices):
            mask = indices == poly_idx
            poly = polynomials[poly_idx]
            ds = s_arr[mask] - poly.s
            result[mask] = poly.a + poly.b * ds + poly.c * ds**2 + poly.d * ds**3
        return result

    def interpolate_se2_batch(
        self,
        s_arr: npt.NDArray[np.float64],
        t_arr: npt.NDArray[np.float64],
        lane_section_end_arr: npt.NDArray[np.bool_],
    ) -> npt.NDArray[np.float64]:
        # Fast path for single polynomial (most common case: constant width)
        if len(self.width_polynomials) == 1:
            t_offsets = self._eval_single_polynomial_batch(s_arr, self.width_polynomials[0])
        else:
            width_indices = self._find_polynomial_indices_batch(s_arr, self.width_polynomials, lane_section_end_arr)
            t_offsets = self._eval_polynomials_batch(s_arr, self.width_polynomials, width_indices)
        return self.reference_line.interpolate_se2_batch(self.s_offset + s_arr, t_offsets + t_arr, lane_section_end_arr)

    def interpolate_3d_batch(
        self,
        s_arr: npt.NDArray[np.float64],
        t_arr: npt.NDArray[np.float64],
        lane_section_end_arr: npt.NDArray[np.bool_],
    ) -> npt.NDArray[np.float64]:
        se2_arr = self.interpolate_se2_batch(s_arr, t_arr, lane_section_end_arr)
        # Fast path for single elevation (common case)
        if len(self.elevations) == 1:
            z_values = self._eval_single_polynomial_batch(s_arr, self.elevations[0])
        else:
            elev_indices = self._find_polynomial_indices_batch(s_arr, self.elevations, lane_section_end_arr)
            z_values = self._eval_polynomials_batch(s_arr, self.elevations, elev_indices)
        result = np.zeros((len(s_arr), len(Point3DIndex)), dtype=np.float64)
        result[:, Point3DIndex.XY] = se2_arr[:, PoseSE2Index.XY]
        result[:, Point3DIndex.Z] = z_values
        return result
