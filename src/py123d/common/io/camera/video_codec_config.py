"""Settings of the AV1 camera codec (camera store option ``"av1"``)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

VIDEO_BACKENDS = ("nvenc", "svt", "auto")


@dataclass(frozen=True)
class VideoCodecConfig:
    """Encoder settings of the AV1 camera codec.

    The defaults are the kedec settings: NVIDIA's AV1 encoder (NVENC) at constant quality 38, one key frame every
    10 images, at most 3 B-frames, rate caps of 200 Mbit/s (buffer 400 Mbit). ``backend: "svt"`` encodes on the
    CPU with SVT-AV1; ``"auto"`` uses NVENC when a GPU encoder opens and SVT-AV1 otherwise.
    """

    # "nvenc" (GPU, default), "svt" (CPU) or "auto".
    backend: str = "nvenc"
    # Nominal image rate of the stream, images per second. The stored time stamps of each image are in the
    # camera's Arrow file; the video only counts images.
    frame_rate: int = 10
    # One key frame every gop_size images (random reads decode from the previous key frame).
    gop_size: int = 10

    # --- NVENC ---
    # Constant quality (0 to 63; higher is smaller). The report "camera-lidar-compression" measured cq 38.
    nvenc_cq: int = 38
    nvenc_preset: str = "p7"
    nvenc_tune: str = "hq"
    nvenc_max_rate: str = "200M"
    nvenc_buffer_size: str = "400M"
    nvenc_b_frames: int = 3
    # GPU index, -1 for any.
    nvenc_gpu: int = -1

    # --- SVT-AV1 ---
    # Constant rate factor (0 to 63) and preset (0 slowest to 13 fastest).
    svt_crf: int = 38
    svt_preset: int = 6
    # Logical processors per encoder, 0 for SVT-AV1's choice (all).
    svt_threads: int = 0

    def __post_init__(self) -> None:
        assert self.backend in VIDEO_BACKENDS, f"backend must be one of {VIDEO_BACKENDS}, got {self.backend}."
        assert self.frame_rate > 0, f"frame_rate must be positive, got {self.frame_rate}."
        assert self.gop_size >= 1, f"gop_size must be >= 1, got {self.gop_size}."
        assert 0 <= self.nvenc_cq <= 63, f"nvenc_cq must be in [0, 63], got {self.nvenc_cq}."
        assert 0 <= self.svt_crf <= 63, f"svt_crf must be in [0, 63], got {self.svt_crf}."
        assert 0 <= self.svt_preset <= 13, f"svt_preset must be in [0, 13], got {self.svt_preset}."
        assert self.nvenc_b_frames >= 0 and self.svt_threads >= 0

    def nvenc_options(self) -> Dict[str, str]:
        """The FFmpeg options of the av1_nvenc encoder."""
        options = {
            "preset": self.nvenc_preset,
            "tune": self.nvenc_tune,
            "rc": "vbr",
            "cq": str(self.nvenc_cq),
            "b": "0",
            "maxrate": self.nvenc_max_rate,
            "bufsize": self.nvenc_buffer_size,
            "g": str(self.gop_size),
            "bf": str(self.nvenc_b_frames),
            "tier": "1",
        }
        if self.nvenc_gpu >= 0:
            options["gpu"] = str(self.nvenc_gpu)
        return options

    def svt_options(self) -> Dict[str, str]:
        """The FFmpeg options of the libsvtav1 encoder."""
        options = {"crf": str(self.svt_crf), "preset": str(self.svt_preset), "g": str(self.gop_size)}
        if self.svt_threads > 0:
            options["svtav1-params"] = f"lp={self.svt_threads}"
        return options

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
