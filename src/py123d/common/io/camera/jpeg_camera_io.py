import io
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image


def is_jpeg_binary(jpeg_binary: bytes) -> bool:
    """Check if the given binary data represents a JPEG image.

    :param jpeg_binary: The binary data to check.
    :return: True if the binary data is a JPEG image, False otherwise.
    """
    SOI_MARKER = b"\xff\xd8"  # Start Of Image
    EOI_MARKER = b"\xff\xd9"  # End Of Image

    return jpeg_binary.startswith(SOI_MARKER) and jpeg_binary.endswith(EOI_MARKER)


def encode_image_as_jpeg_binary(image: npt.NDArray[np.uint8]) -> bytes:
    """Encodes a numpy RGB image as JPEG binary."""
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, encoded_img = cv2.imencode(".jpg", image)
    jpeg_binary = encoded_img.tobytes()
    return jpeg_binary


def decode_image_from_jpeg_binary(
    jpeg_binary: bytes,
    scale: Optional[int] = None,
) -> npt.NDArray[np.uint8]:
    """Decodes a numpy RGB image from JPEG binary.

    :param jpeg_binary: The JPEG binary data to decode.
    :param scale: Optional downscale denominator, e.g. 2 for half size, 4 for quarter size.
        For JPEG, uses Pillow's DCT-level scaling (supported factors: 2, 4, 8).
    """
    if scale is not None and scale > 1:
        img = Image.open(io.BytesIO(jpeg_binary))
        w, h = img.size
        img.draft("RGB", (w // scale, h // scale))
        img.load()
        return np.array(img)

    image = cv2.imdecode(np.frombuffer(jpeg_binary, np.uint8), cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_jpeg_binary_from_jpeg_file(jpeg_path: Path) -> bytes:
    """Loads JPEG binary data from a JPEG file."""
    with open(jpeg_path, "rb") as f:
        jpeg_binary = f.read()
    return jpeg_binary


def load_image_from_jpeg_file(
    jpeg_path: Path,
    scale: Optional[int] = None,
) -> npt.NDArray[np.uint8]:
    """Loads a numpy RGB image from a JPEG file.

    :param jpeg_path: Path to the JPEG file.
    :param scale: Optional downscale denominator, e.g. 2 for half size, 4 for quarter size.
        For JPEG, uses Pillow's DCT-level scaling (supported factors: 2, 4, 8).
    """
    if scale is not None and scale > 1:
        img = Image.open(jpeg_path)
        w, h = img.size
        img.draft("RGB", (w // scale, h // scale))
        img.load()
        return np.array(img)

    image = cv2.imread(str(jpeg_path), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
