from dataclasses import dataclass
from typing import Tuple

KEDEC_MODES = ("auto", "spherical", "grid")
KEDEC_FRAMES = ("position", "pose")
KEDEC_INT_CHOICES = ("estimate", "exact", "raw", "diff")


@dataclass(frozen=True)
class PointCloudCodecConfig:
    """Encoder settings for the LAZ, Draco and kedec point-cloud codecs, shared by the Lidar and Radar writers.

    These settle how the point cloud is quantized inside its binary blob. They are unrelated to the
    ``ipc_compression`` of the enclosing Arrow file, which compresses the record batches that carry
    the blob.

    The defaults reproduce the historical hard-coded encoder behaviour, so logs written without an
    explicit config are byte-for-byte unchanged.
    """

    # --- LAZ ---
    # LAS point format. Format 0 carries only xyz + the base attributes; formats 1/3 additionally
    # reserve room for GPS time and RGB, which cost bytes even when left unset.
    laz_point_format: int = 3
    # Quantization step of the LAS integer grid, in meters, per axis. This is the dominant driver of
    # the encoded size: LASzip stores xyz as int32 multiples of the scale, so a coarser scale
    # directly shortens the arithmetic-coded residuals.
    laz_scales: Tuple[float, float, float] = (0.01, 0.01, 0.01)
    # Origin of the LAS integer grid, in meters, per axis.
    laz_offsets: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # --- Draco ---
    draco_quantization_bits: int = 16
    # Range: 0 (fastest) to 10 (slowest, best compression).
    draco_compression_level: int = 7
    # -1 uses Draco's default range.
    draco_quantization_range: int = -1
    draco_preserve_order: bool = True

    # --- kedec (lidar only; see py123d.common.io.lidar.kedec_lidar_io) ---
    # "grid" codes beams and columns against per-scan tables and needs both a beam index (feature "channel") and a
    # column index (feature "column"); "spherical" works for every lidar; "auto" uses "grid" only when both
    # indices are stored.
    kedec_mode: str = "auto"
    # Range step in meters; the range error is at most half a step.
    kedec_range_step_m: float = 0.02
    # Spherical mode: azimuth and elevation step in degrees. 0.008 degrees keeps the position error within 2 cm up
    # to about 175 m (5.04 cm at 500 m) together with the default range step.
    kedec_angle_step_deg: float = 0.008
    # Spherical mode: height of the elevation rows that order the points of a scan without a ring index, degrees.
    kedec_row_step_deg: float = 0.1
    # Spherical mode: "position" codes the points around the sensor position (the translation of the sensor pose),
    # "pose" in the sensor frame (translation and rotation). The grid mode always uses the sensor frame.
    kedec_frame: str = "position"
    # Grid mode: step of the sideways deviations from the beam and column tables, in meters.
    kedec_side_step_m: float = 0.02
    # Zstandard level of all parts (1 to 22).
    kedec_zstd_level: int = 19
    # Choice between raw values and differences per integer channel ("estimate" decides at Zstandard level 1).
    kedec_int_choice: str = "estimate"

    def __post_init__(self) -> None:
        assert self.laz_point_format in {0, 1, 2, 3, 6, 7, 8}, (
            f"Unsupported LAS point format, got {self.laz_point_format}."
        )
        assert len(self.laz_scales) == 3 and all(scale > 0.0 for scale in self.laz_scales), (
            f"LAZ scales must be three positive values, got {self.laz_scales}."
        )
        assert len(self.laz_offsets) == 3, f"LAZ offsets must be three values, got {self.laz_offsets}."
        assert 1 <= self.draco_quantization_bits <= 30, (
            f"Draco quantization bits must be in [1, 30], got {self.draco_quantization_bits}."
        )
        assert 0 <= self.draco_compression_level <= 10, (
            f"Draco compression level must be in [0, 10], got {self.draco_compression_level}."
        )
        assert self.kedec_mode in KEDEC_MODES, f"kedec mode must be one of {KEDEC_MODES}, got {self.kedec_mode}."
        assert self.kedec_frame in KEDEC_FRAMES, f"kedec frame must be one of {KEDEC_FRAMES}, got {self.kedec_frame}."
        assert self.kedec_int_choice in KEDEC_INT_CHOICES, (
            f"kedec integer choice must be one of {KEDEC_INT_CHOICES}, got {self.kedec_int_choice}."
        )
        for name in ("kedec_range_step_m", "kedec_angle_step_deg", "kedec_row_step_deg", "kedec_side_step_m"):
            assert getattr(self, name) > 0.0, f"{name} must be positive, got {getattr(self, name)}."
        assert 1 <= self.kedec_zstd_level <= 22, (
            f"kedec Zstandard level must be in [1, 22], got {self.kedec_zstd_level}."
        )
