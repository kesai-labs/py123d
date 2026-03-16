from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import numpy.typing as npt


def is_png_binary(png_binary: bytes) -> bool:
    """Check if the given binary data represents a PNG image.

    :param png_binary: The binary data to check.
    :return: True if the binary data is a PNG image, False otherwise.
    """
    PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"  # PNG file signature

    return png_binary.startswith(PNG_SIGNATURE)


def encode_image_as_png_binary(image: npt.NDArray[np.uint8]) -> bytes:
    """Encodes a numpy RGB image as PNG binary."""
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # type: ignore
    _, encoded_img = cv2.imencode(".png", image)
    png_binary = encoded_img.tobytes()
    return png_binary


def decode_image_from_png_binary(png_binary: bytes, scale: Optional[int] = None) -> npt.NDArray[np.uint8]:
    """Decodes a numpy image from PNG binary.

    :param png_binary: The PNG binary data to decode.
    :param scale: Optional downscale denominator, e.g. 2 for half size, 4 for quarter size.
    """
    image = cv2.imdecode(np.frombuffer(png_binary, np.uint8), cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if scale is not None and scale > 1:
        new_h = image.shape[0] // scale
        new_w = image.shape[1] // scale
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image  # type: ignore[return-value]


def load_png_binary_from_png_file(png_path: Path) -> bytes:
    """Loads PNG binary data from a PNG file."""
    with open(png_path, "rb") as f:
        png_binary = f.read()
    return png_binary


def load_image_from_png_file(png_path: Path) -> npt.NDArray[np.uint8]:
    """Loads a numpy image from a PNG file."""
    image = cv2.imread(str(png_path), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image  # type: ignore[return-value]
