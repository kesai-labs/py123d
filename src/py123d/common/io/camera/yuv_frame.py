"""8-bit YUV 4:2:0 images: the input and output of the AV1 camera codec.

A :class:`YUV420Frame` holds the luma plane (H, W) and the two chroma planes (H/2, W/2), together with the color
range (full: 0-255, as in JPEG; limited: 16-235 luma, 16-240 chroma, as in most videos) and the matrix (BT.601 or
BT.709) that relate them to RGB. Images from JPEG files keep their decoded planes exactly (JPEG stores YUV 4:2:0 with
the full-range BT.601 matrix of JFIF), so no color conversion is applied before the video encoder; RGB images are
converted with the JFIF matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np
import numpy.typing as npt

MATRICES = ("bt601", "bt709")

# (Kr, Kb) of the luma definition of each matrix.
_LUMA_WEIGHTS: Dict[str, Tuple[float, float]] = {"bt601": (0.299, 0.114), "bt709": (0.2126, 0.0722)}

# Color tags of FFmpeg (AVColorSpace) for the matrices.
AV_COLORSPACE = {"bt601": 5, "bt709": 1}
_FROM_AV_COLORSPACE = {5: "bt601", 6: "bt601", 1: "bt709"}


def _ycbcr_from_rgb_matrix(matrix: str, full_range: bool) -> npt.NDArray[np.float32]:
    """3x4 matrix that maps (R, G, B, 1) to (Y, Cb, Cr) in 8-bit code values."""
    kr, kb = _LUMA_WEIGHTS[matrix]
    kg = 1.0 - kr - kb
    rgb_to_ycbcr = np.array(
        [
            [kr, kg, kb],
            [-kr / (2 * (1 - kb)), -kg / (2 * (1 - kb)), 0.5],
            [0.5, -kg / (2 * (1 - kr)), -kb / (2 * (1 - kr))],
        ]
    )
    if full_range:
        scale, offset = np.array([1.0, 1.0, 1.0]), np.array([0.0, 128.0, 128.0])
    else:
        scale, offset = np.array([219.0, 224.0, 224.0]) / 255.0, np.array([16.0, 128.0, 128.0])
    m = rgb_to_ycbcr * scale[:, None]
    return np.hstack([m, offset[:, None]]).astype(np.float32)


def _rgb_from_ycbcr_matrix(matrix: str, full_range: bool) -> npt.NDArray[np.float32]:
    """3x4 matrix that maps (Y, Cb, Cr, 1) in 8-bit code values to (R, G, B)."""
    forward = _ycbcr_from_rgb_matrix(matrix, full_range).astype(np.float64)
    inverse = np.linalg.inv(forward[:, :3])
    return np.hstack([inverse, (-inverse @ forward[:, 3])[:, None]]).astype(np.float32)


@dataclass(frozen=True)
class YUV420Frame:
    """An 8-bit YUV 4:2:0 image with even width and height.

    :param y: Luma plane, (H, W) uint8.
    :param u: Blue-difference chroma plane (Cb), (H/2, W/2) uint8.
    :param v: Red-difference chroma plane (Cr), (H/2, W/2) uint8.
    :param full_range: True for full-range code values (JPEG), False for limited range (most videos).
    :param matrix: ``"bt601"`` or ``"bt709"``.
    """

    y: npt.NDArray[np.uint8]
    u: npt.NDArray[np.uint8]
    v: npt.NDArray[np.uint8]
    full_range: bool = True
    matrix: str = "bt601"

    def __post_init__(self) -> None:
        if self.matrix not in MATRICES:
            raise ValueError(f"Unknown YUV matrix {self.matrix!r}, expected one of {MATRICES}.")
        height, width = self.y.shape[:2] if self.y.ndim == 2 else (-1, -1)
        if height <= 0 or height % 2 or width % 2:
            raise ValueError(f"A YUV 4:2:0 frame requires a 2-D luma plane of even size, got {self.y.shape}.")
        for name in ("y", "u", "v"):
            if getattr(self, name).dtype != np.uint8:
                raise ValueError(f"YUV planes must be uint8, plane {name} is {getattr(self, name).dtype}.")
        if self.u.shape != (height // 2, width // 2) or self.v.shape != (height // 2, width // 2):
            raise ValueError(f"Chroma planes must have shape {(height // 2, width // 2)}, got {self.u.shape}.")

    @property
    def height(self) -> int:
        return int(self.y.shape[0])

    @property
    def width(self) -> int:
        return int(self.y.shape[1])

    def to_i420(self) -> npt.NDArray[np.uint8]:
        """The planes in I420 layout, shape (H * 3 / 2, W), as PyAV's ``VideoFrame.from_ndarray`` takes them."""
        return np.concatenate([self.y.ravel(), self.u.ravel(), self.v.ravel()]).reshape(
            self.height * 3 // 2, self.width
        )

    @classmethod
    def from_i420(cls, data: npt.NDArray[np.uint8], full_range: bool = True, matrix: str = "bt601") -> "YUV420Frame":
        """The inverse of :meth:`to_i420`."""
        height = data.shape[0] * 2 // 3
        width = data.shape[1]
        flat = np.ascontiguousarray(data).reshape(-1)
        luma = height * width
        chroma = luma // 4
        return cls(
            y=flat[:luma].reshape(height, width),
            u=flat[luma : luma + chroma].reshape(height // 2, width // 2),
            v=flat[luma + chroma : luma + 2 * chroma].reshape(height // 2, width // 2),
            full_range=full_range,
            matrix=matrix,
        )

    def to_rgb(self) -> npt.NDArray[np.uint8]:
        """The image as (H, W, 3) uint8 RGB; the chroma planes are upsampled bilinearly."""
        size = (self.width, self.height)
        u = cv2.resize(self.u, size, interpolation=cv2.INTER_LINEAR)
        v = cv2.resize(self.v, size, interpolation=cv2.INTER_LINEAR)
        return cv2.transform(cv2.merge([self.y, u, v]), _rgb_from_ycbcr_matrix(self.matrix, self.full_range))

    @classmethod
    def from_rgb(cls, rgb: npt.NDArray[np.uint8], full_range: bool = True, matrix: str = "bt601") -> "YUV420Frame":
        """Convert an (H, W, 3) uint8 RGB image of even size; the chroma planes are 2 x 2 means.

        The defaults (full range, BT.601) are the JFIF conversion of JPEG.
        """
        rgb = np.asarray(rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError(f"Expected an (H, W, 3) uint8 RGB image, got {rgb.dtype} {rgb.shape}.")
        if rgb.shape[0] % 2 or rgb.shape[1] % 2:
            raise ValueError(f"A YUV 4:2:0 frame requires an even image size, got {rgb.shape[:2]}.")
        ycbcr = cv2.transform(rgb, _ycbcr_from_rgb_matrix(matrix, full_range))
        half = (rgb.shape[1] // 2, rgb.shape[0] // 2)
        u = cv2.resize(ycbcr[..., 1], half, interpolation=cv2.INTER_AREA)
        v = cv2.resize(ycbcr[..., 2], half, interpolation=cv2.INTER_AREA)
        return cls(np.ascontiguousarray(ycbcr[..., 0]), u, v, full_range=full_range, matrix=matrix)

    @classmethod
    def from_av_frame(cls, frame) -> "YUV420Frame":
        """The planes of a decoded PyAV video frame; 4:2:0 frames keep their planes, others are converted.

        The color range and matrix are taken from the frame's tags (unspecified: limited range; BT.709 from 720
        lines up, else BT.601).
        """
        full_range = int(frame.color_range) == 2 or frame.format.name.startswith("yuvj")
        matrix = _FROM_AV_COLORSPACE.get(int(frame.colorspace), "bt709" if frame.height >= 720 else "bt601")
        if frame.format.name in ("yuv420p", "yuvj420p") and frame.width % 2 == 0 and frame.height % 2 == 0:
            return cls.from_i420(frame.to_ndarray(), full_range=full_range, matrix=matrix)
        converted = frame.reformat(format="yuv420p")
        return cls.from_i420(converted.to_ndarray(), full_range=full_range, matrix=matrix)

    @classmethod
    def from_jpeg(cls, jpeg_binary: bytes) -> "YUV420Frame":
        """Decode a JPEG image; a 4:2:0 JPEG keeps its decoded planes exactly, other JPEGs are converted via RGB.

        :raises ValueError: If the image cannot be decoded or its width or height is odd.
        """
        frame = _decode_jpeg_frame(jpeg_binary)
        if frame.format.name in ("yuvj420p", "yuv420p") and frame.width % 2 == 0 and frame.height % 2 == 0:
            return cls.from_i420(frame.to_ndarray(), full_range=True, matrix="bt601")
        if frame.format.name == "gray":
            rgb = np.repeat(frame.to_ndarray()[..., None], 3, axis=2)
        else:
            rgb = frame.to_ndarray(format="rgb24")
        return cls.from_rgb(rgb)


def _decode_jpeg_frame(jpeg_binary: bytes):
    """Decode one JPEG image with FFmpeg's JPEG decoder (one decoder per call; the decoder is not shared)."""
    import av

    context = av.CodecContext.create("mjpeg", "r")
    frames = context.decode(av.Packet(jpeg_binary))
    if not frames:
        frames = context.decode(None)
    if not frames:
        raise ValueError("The JPEG image could not be decoded.")
    return frames[0]
