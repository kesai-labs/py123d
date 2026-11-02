"""The schema-metadata entry that names the codec of a camera or lidar Arrow file (key ``codec``).

The value is UTF-8 JSON with at least ``codec`` (the codec name), ``version`` (the format version) and the writer's
settings, so that a log records how its sensor data were encoded.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Optional

CODEC_METADATA_KEY = b"codec"


def encode_codec_metadata(description: Mapping[str, Any]) -> bytes:
    """The schema-metadata value of a codec description (UTF-8 JSON)."""
    return json.dumps(dict(description), sort_keys=True).encode("utf-8")


def decode_codec_metadata(value: Optional[bytes]) -> Optional[Dict[str, Any]]:
    """The codec description of a schema-metadata value, or None if there is none."""
    return None if value is None else json.loads(value.decode("utf-8"))
