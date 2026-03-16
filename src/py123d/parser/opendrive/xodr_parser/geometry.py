from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
from xml.etree.ElementTree import Element

import numpy as np
import numpy.typing as npt
from scipy.special import fresnel

from py123d.geometry import PoseSE2Index


@dataclass
class XODRGeometry:
    """
    https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/latest/specification/09_geometries/09_02_road_reference_line.html
    """

    s: float
    x: float
    y: float
    hdg: float
    length: float

    @property
    def start_se2(self) -> npt.NDArray[np.float64]:
        start_se2 = np.zeros(len(PoseSE2Index), dtype=np.float64)
        start_se2[PoseSE2Index.X] = self.x
        start_se2[PoseSE2Index.Y] = self.y
        start_se2[PoseSE2Index.YAW] = self.hdg
        return start_se2

    def interpolate_se2(self, s: float, t: float = 0.0) -> npt.NDArray[np.float64]:
        raise NotImplementedError

    def interpolate_se2_batch(self, s: npt.NDArray[np.float64], t: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        raise NotImplementedError


@dataclass
class XODRLine(XODRGeometry):
    """
    https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/latest/specification/09_geometries/09_03_straight_line.html
    """

    @classmethod
    def parse(cls, geometry_element: Element) -> XODRGeometry:
        args = {key: float(geometry_element.get(key)) for key in ["s", "x", "y", "hdg", "length"]}
        return cls(**args)

    def interpolate_se2(self, s: float, t: float = 0.0) -> npt.NDArray[np.float64]:
        interpolated_se2 = self.start_se2.copy()
        interpolated_se2[PoseSE2Index.X] += s * np.cos(self.hdg)
        interpolated_se2[PoseSE2Index.Y] += s * np.sin(self.hdg)

        if t != 0.0:
            interpolated_se2[PoseSE2Index.X] += t * np.cos(interpolated_se2[PoseSE2Index.YAW] + np.pi / 2)
            interpolated_se2[PoseSE2Index.Y] += t * np.sin(interpolated_se2[PoseSE2Index.YAW] + np.pi / 2)

        return interpolated_se2

    def interpolate_se2_batch(self, s: npt.NDArray[np.float64], t: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        n = len(s)
        cos_hdg = np.cos(self.hdg)
        sin_hdg = np.sin(self.hdg)
        result = np.empty((n, 3), dtype=np.float64)
        result[:, 0] = self.x + s * cos_hdg
        result[:, 1] = self.y + s * sin_hdg
        result[:, 2] = self.hdg
        mask = t != 0.0
        if np.any(mask):
            yaw_half_pi = self.hdg + np.pi / 2
            result[mask, 0] += t[mask] * np.cos(yaw_half_pi)
            result[mask, 1] += t[mask] * np.sin(yaw_half_pi)
        return result


@dataclass
class XODRArc(XODRGeometry):
    """
    https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/latest/specification/09_geometries/09_05_arc.html
    """

    curvature: float

    def __post_init__(self):
        if self.curvature == 0.0:
            raise ValueError("Curvature cannot be zero for Arc geometry.")

    @classmethod
    def parse(cls, geometry_element: Element) -> XODRGeometry:
        args = {key: float(geometry_element.get(key)) for key in ["s", "x", "y", "hdg", "length"]}
        args["curvature"] = float(geometry_element.find("arc").get("curvature"))
        return cls(**args)

    def interpolate_se2(self, s: float, t: float = 0.0) -> npt.NDArray[np.float64]:
        kappa = self.curvature
        radius = 1.0 / kappa if kappa != 0 else float("inf")

        initial_heading = self.hdg
        final_heading = initial_heading + s * kappa

        # Parametric circle equations
        dx = radius * (np.sin(final_heading) - np.sin(initial_heading))
        dy = -radius * (np.cos(final_heading) - np.cos(initial_heading))

        interpolated_se2 = self.start_se2.copy()
        interpolated_se2[PoseSE2Index.X] += dx
        interpolated_se2[PoseSE2Index.Y] += dy
        interpolated_se2[PoseSE2Index.YAW] = final_heading

        if t != 0.0:
            interpolated_se2[PoseSE2Index.X] += t * np.cos(interpolated_se2[PoseSE2Index.YAW] + np.pi / 2)
            interpolated_se2[PoseSE2Index.Y] += t * np.sin(interpolated_se2[PoseSE2Index.YAW] + np.pi / 2)

        return interpolated_se2

    def interpolate_se2_batch(self, s: npt.NDArray[np.float64], t: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        kappa = self.curvature
        radius = 1.0 / kappa if kappa != 0 else float("inf")
        final_heading = self.hdg + s * kappa
        sin_initial = np.sin(self.hdg)
        cos_initial = np.cos(self.hdg)
        n = len(s)
        result = np.empty((n, 3), dtype=np.float64)
        result[:, 0] = self.x + radius * (np.sin(final_heading) - sin_initial)
        result[:, 1] = self.y - radius * (np.cos(final_heading) - cos_initial)
        result[:, 2] = final_heading
        mask = t != 0.0
        if np.any(mask):
            yaw_half_pi = result[mask, 2] + np.pi / 2
            result[mask, 0] += t[mask] * np.cos(yaw_half_pi)
            result[mask, 1] += t[mask] * np.sin(yaw_half_pi)
        return result


@dataclass
class XODRSpiral(XODRGeometry):
    """
    https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/latest/specification/09_geometries/09_04_spiral.html
    https://en.wikipedia.org/wiki/Euler_spiral
    """

    curvature_start: float
    curvature_end: float

    def __post_init__(self):
        gamma = (self.curvature_end - self.curvature_start) / self.length
        if abs(gamma) < 1e-10:
            raise ValueError("Curvature change is too small, cannot define a valid spiral.")

    @classmethod
    def parse(cls, geometry_element: Element) -> XODRGeometry:
        args = {key: float(geometry_element.get(key)) for key in ["s", "x", "y", "hdg", "length"]}
        spiral_element = geometry_element.find("spiral")
        args["curvature_start"] = float(spiral_element.get("curvStart"))
        args["curvature_end"] = float(spiral_element.get("curvEnd"))
        return cls(**args)

    def interpolate_se2(self, s: float, t: float = 0.0) -> npt.NDArray[np.float64]:
        interpolated_se2 = self.start_se2.copy()

        gamma = (self.curvature_end - self.curvature_start) / self.length

        dx, dy = self._compute_spiral_position(s, gamma)

        interpolated_se2[PoseSE2Index.X] += dx
        interpolated_se2[PoseSE2Index.Y] += dy
        interpolated_se2[PoseSE2Index.YAW] += gamma * s**2 / 2 + self.curvature_start * s

        if t != 0.0:
            interpolated_se2[PoseSE2Index.X] += t * np.cos(interpolated_se2[PoseSE2Index.YAW] + np.pi / 2)
            interpolated_se2[PoseSE2Index.Y] += t * np.sin(interpolated_se2[PoseSE2Index.YAW] + np.pi / 2)

        return interpolated_se2

    def interpolate_se2_batch(self, s: npt.NDArray[np.float64], t: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        gamma = (self.curvature_end - self.curvature_start) / self.length
        dx, dy = self._compute_spiral_position_batch(s, gamma)
        n = len(s)
        result = np.empty((n, 3), dtype=np.float64)
        result[:, 0] = self.x + dx
        result[:, 1] = self.y + dy
        result[:, 2] = self.hdg + gamma * s**2 / 2 + self.curvature_start * s
        mask = t != 0.0
        if np.any(mask):
            yaw_half_pi = result[mask, 2] + np.pi / 2
            result[mask, 0] += t[mask] * np.cos(yaw_half_pi)
            result[mask, 1] += t[mask] * np.sin(yaw_half_pi)
        return result

    def _compute_spiral_position_batch(
        self, s: npt.NDArray[np.float64], gamma: float
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        abs_gamma = abs(gamma)
        sqrt_gamma_pi = np.sqrt(abs_gamma / np.pi)
        scale = np.sqrt(np.pi / abs_gamma)

        v_start = self.curvature_start / gamma
        v_end = s + v_start

        w_start = v_start * sqrt_gamma_pi
        w_end = v_end * sqrt_gamma_pi

        S_start, C_start = fresnel(w_start)
        S_end, C_end = fresnel(w_end)
        dC = C_end - C_start
        dS = S_end - S_start

        phase = -(self.curvature_start**2) / (2 * gamma)
        cos_p = np.cos(phase)
        sin_p = np.sin(phase)

        if gamma > 0:
            dx_local = scale * (cos_p * dC - sin_p * dS)
            dy_local = scale * (cos_p * dS + sin_p * dC)
        else:
            dx_local = scale * (cos_p * dC + sin_p * dS)
            dy_local = scale * (-cos_p * dS + sin_p * dC)

        cos_hdg = np.cos(self.hdg)
        sin_hdg = np.sin(self.hdg)
        dx = cos_hdg * dx_local - sin_hdg * dy_local
        dy = sin_hdg * dx_local + cos_hdg * dy_local
        return dx, dy

    def _compute_spiral_position(self, s: float, gamma: float) -> Tuple[float, float]:
        abs_gamma = abs(gamma)
        sqrt_gamma_pi = np.sqrt(abs_gamma / np.pi)
        scale = np.sqrt(np.pi / abs_gamma)

        v_start = self.curvature_start / gamma
        v_end = s + v_start

        w_start = v_start * sqrt_gamma_pi
        w_end = v_end * sqrt_gamma_pi

        S_start, C_start = fresnel(w_start)
        S_end, C_end = fresnel(w_end)
        dC = C_end - C_start
        dS = S_end - S_start

        phase = -(self.curvature_start**2) / (2 * gamma)
        cos_p = np.cos(phase)
        sin_p = np.sin(phase)

        if gamma > 0:
            dx_local = scale * (cos_p * dC - sin_p * dS)
            dy_local = scale * (cos_p * dS + sin_p * dC)
        else:
            dx_local = scale * (cos_p * dC + sin_p * dS)
            dy_local = scale * (-cos_p * dS + sin_p * dC)

        cos_hdg = np.cos(self.hdg)
        sin_hdg = np.sin(self.hdg)
        dx = cos_hdg * dx_local - sin_hdg * dy_local
        dy = sin_hdg * dx_local + cos_hdg * dy_local
        return dx, dy
