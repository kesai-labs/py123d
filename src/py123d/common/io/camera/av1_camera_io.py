"""AV1 video files of camera images (camera store option ``"av1"``).

:class:`AV1Writer` encodes the images of one camera into ``{key}.av1.mp4`` with NVIDIA's encoder (NVENC) or
SVT-AV1, and returns each image's number, which the camera's Arrow file stores. :class:`AV1Reader` returns the
image of a number: sequential reads decode the next image, other reads seek to the previous key frame. The
conversion to RGB uses the color range and matrix that the file declares (full-range BT.601 for JPEG sources).
"""

from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import numpy.typing as npt

from py123d.common.io.camera.video_codec_config import VideoCodecConfig
from py123d.common.io.camera.yuv_frame import AV_COLORSPACE, YUV420Frame

logger = logging.getLogger(__name__)

AV1_FILE_SUFFIX = ".av1.mp4"
AV1_FORMAT_VERSION = 1
_ENCODERS = {"nvenc": "av1_nvenc", "svt": "libsvtav1"}
_COLOR_RANGE_JPEG, _COLOR_RANGE_MPEG = 2, 1


def av1_file_name(modality_key: str) -> str:
    """The file name of a camera's AV1 video: ``{modality_key}.av1.mp4``."""
    return f"{modality_key}{AV1_FILE_SUFFIX}"


@lru_cache(maxsize=None)
def nvenc_av1_available(gpu: int = -1) -> bool:
    """Whether an NVENC AV1 encoder opens in this process (a GPU with AV1 encoding and its driver).

    :param gpu: GPU index, -1 for any.
    """
    try:
        import av

        context = av.CodecContext.create("av1_nvenc", "w")
        context.width, context.height, context.pix_fmt = 256, 256, "yuv420p"
        context.time_base = Fraction(1, 10)
        context.framerate = Fraction(10, 1)
        if gpu >= 0:
            context.options = {"gpu": str(gpu)}
        context.open()
        del context  # frees the encoder session
        return True
    except Exception as error:  # noqa: BLE001 - any failure means the encoder is not usable
        logger.debug("NVENC AV1 encoder not available: %s", error)
        return False


def resolve_av1_encoder(config: VideoCodecConfig) -> str:
    """The FFmpeg encoder of a configuration: ``"av1_nvenc"`` or ``"libsvtav1"``.

    :raises RuntimeError: If the backend is ``"nvenc"`` and no NVENC AV1 encoder opens.
    """
    if config.backend == "svt":
        return _ENCODERS["svt"]
    if nvenc_av1_available(config.nvenc_gpu):
        return _ENCODERS["nvenc"]
    if config.backend == "nvenc":
        raise RuntimeError(
            "No NVENC AV1 encoder is available (a GPU with AV1 encoding is required, e.g. RTX 40 series or later); "
            "use the video backend 'svt' or 'auto' to encode on the CPU."
        )
    logger.warning("No NVENC AV1 encoder is available; encoding AV1 on the CPU with SVT-AV1.")
    return _ENCODERS["svt"]


def av1_codec_metadata(config: VideoCodecConfig, encoder: str) -> Dict[str, Any]:
    """The codec description written into the schema metadata of a camera's Arrow file (key ``codec``)."""
    from py123d import __version__

    options = config.nvenc_options() if encoder == _ENCODERS["nvenc"] else config.svt_options()
    return {
        "codec": "av1",
        "version": AV1_FORMAT_VERSION,
        "encoder": encoder,
        "writer": f"py123d {__version__}",
        "frame_rate": config.frame_rate,
        "options": options,
    }


class AV1Writer:
    """Writes the images of one camera into an AV1 video (MP4 container).

    The encoder opens with the first image, whose size and color tags all later images must share. :meth:`close`
    flushes the encoder and closes the file; it is safe to call more than once and releases the encoder also after
    an error.
    """

    def __init__(
        self,
        output_path: Union[str, Path],
        config: Optional[VideoCodecConfig] = None,
        encoder: Optional[str] = None,
    ) -> None:
        """
        :param output_path: The path of the video file (``{key}.av1.mp4``).
        :param config: Encoder settings; defaults to :class:`VideoCodecConfig`.
        :param encoder: ``"av1_nvenc"`` or ``"libsvtav1"``; resolved from the configuration when None.
        """
        self.output_path = Path(output_path)
        self.config = config if config is not None else VideoCodecConfig()
        self.encoder = encoder if encoder is not None else resolve_av1_encoder(self.config)
        if self.encoder not in _ENCODERS.values():
            raise ValueError(f"Unknown AV1 encoder {self.encoder!r}.")
        self.frame_count = 0
        self._container: Any = None
        self._stream: Any = None
        self._format: Optional[Tuple[int, int, bool, str]] = None
        self._closed = False

    def __enter__(self) -> "AV1Writer":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def _open(self, frame: YUV420Frame) -> None:
        import av

        if self.encoder == _ENCODERS["svt"]:
            os.environ.setdefault("SVT_LOG", "2")  # SVT-AV1 prints its configuration at level 3 and up
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        container = av.open(str(self.output_path), "w", format="mp4")
        try:
            stream = container.add_stream(self.encoder, rate=self.config.frame_rate)
            stream.width, stream.height, stream.pix_fmt = frame.width, frame.height, "yuv420p"
            stream.time_base = Fraction(1, self.config.frame_rate)
            context = stream.codec_context
            context.color_range = _COLOR_RANGE_JPEG if frame.full_range else _COLOR_RANGE_MPEG
            context.colorspace = AV_COLORSPACE[frame.matrix]
            if frame.matrix == "bt709":
                context.color_primaries = 1
                context.color_trc = 1
            options = self.config.nvenc_options() if self.encoder == _ENCODERS["nvenc"] else self.config.svt_options()
            stream.options = options
        except Exception:
            container.close()
            raise
        self._container, self._stream = container, stream
        self._format = (frame.width, frame.height, frame.full_range, frame.matrix)

    def write_frame(self, frame: Union[YUV420Frame, npt.NDArray[np.uint8]]) -> int:
        """Encode one image.

        :param frame: A :class:`YUV420Frame`, or an (H, W, 3) uint8 RGB image (converted with the JFIF matrix).
        :return: The number of the image in the video (0, 1, 2, ...).
        """
        import av

        if self._closed:
            raise RuntimeError(f"The AV1 writer of {self.output_path} is closed.")
        yuv = frame if isinstance(frame, YUV420Frame) else YUV420Frame.from_rgb(frame)
        if self._container is None:
            self._open(yuv)
        if (yuv.width, yuv.height, yuv.full_range, yuv.matrix) != self._format:
            raise ValueError(
                f"Image {(yuv.width, yuv.height, yuv.full_range, yuv.matrix)} differs from the video {self._format}."
            )
        video_frame = av.VideoFrame.from_ndarray(yuv.to_i420(), format="yuv420p")
        video_frame.pts = self.frame_count
        video_frame.time_base = Fraction(1, self.config.frame_rate)
        video_frame.color_range = _COLOR_RANGE_JPEG if yuv.full_range else _COLOR_RANGE_MPEG
        video_frame.colorspace = AV_COLORSPACE[yuv.matrix]
        for packet in self._stream.encode(video_frame):
            self._container.mux(packet)
        index = self.frame_count
        self.frame_count += 1
        return index

    def close(self) -> None:
        """Flush the encoder and close the file (no effect when already closed)."""
        if self._closed:
            return
        self._closed = True
        if self._container is None:
            return
        try:
            for packet in self._stream.encode(None):
                self._container.mux(packet)
        finally:
            self._container.close()
            self._container = None
            self._stream = None


class AV1Reader:
    """Reads images from an AV1 video written by :class:`AV1Writer`.

    Reads of the next image decode one image; other reads seek to the key frame at or before the image (one per
    ``gop_size`` images) and decode from there. The last image is kept. An instance is not thread-safe; the readers of
    :func:`get_av1_reader` come with a lock for that.
    """

    def __init__(self, video_path: Union[str, Path], decoder: str = "libdav1d", threads: int = 0) -> None:
        """
        :param video_path: The path of the video file.
        :param decoder: ``"libdav1d"`` (CPU) or ``"av1_cuvid"`` (NVIDIA GPU decoder).
        :param threads: Decoder threads, 0 for the decoder's choice.
        :raises FileNotFoundError: If the file does not exist.
        """
        import av

        self.video_path = str(video_path)
        if not Path(self.video_path).exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
        self.lock = threading.Lock()
        self._container = av.open(self.video_path)
        self._stream = self._container.streams.video[0]
        source = self._stream.codec_context
        self.full_range = int(source.color_range) == _COLOR_RANGE_JPEG
        self.matrix = "bt709" if int(source.colorspace) == AV_COLORSPACE["bt709"] else "bt601"
        self.width, self.height = int(source.width), int(source.height)
        rate = self._stream.average_rate or self._stream.guessed_rate or 10
        self._pts_step = max(1, int(round(1 / (float(self._stream.time_base) * float(rate)))))
        self.frame_count = int(self._stream.frames)
        self._decoder_name = decoder
        self._decoder: Any = None
        if decoder == "av1_cuvid":
            self._decoder = av.CodecContext.create("av1_cuvid", "r")
            self._decoder.extradata = source.extradata
        elif decoder != "libdav1d":
            raise ValueError(f"Unknown AV1 decoder {decoder!r}, expected 'libdav1d' or 'av1_cuvid'.")
        elif threads > 0:
            source.thread_count = threads
        self._frames: Any = None  # iterator of decoded frames
        self._next_index: Optional[int] = None
        self._cached: Optional[Tuple[int, YUV420Frame]] = None

    def _decode(self):
        """Decoded frames from the current position of the demuxer."""
        if self._decoder is None:
            yield from self._container.decode(self._stream)
            return
        for packet in self._container.demux(self._stream):
            for frame in self._decoder.decode(packet if packet.size else None):
                yield frame

    def _seek(self, index: int) -> None:
        import av

        self._container.seek(index * self._pts_step, stream=self._stream, backward=True, any_frame=False)
        if self._decoder is not None:  # a drained or used GPU decoder takes no packets after a seek; use a new one
            extradata = self._decoder.extradata
            self._decoder = av.CodecContext.create(self._decoder_name, "r")
            self._decoder.extradata = extradata
        self._frames = self._decode()
        self._next_index = None

    def _to_yuv(self, frame) -> YUV420Frame:
        if frame.format.name == "nv12":
            data = frame.to_ndarray()
            chroma = data[self.height :].reshape(self.height // 2, self.width // 2, 2)
            return YUV420Frame(
                np.ascontiguousarray(data[: self.height]),
                np.ascontiguousarray(chroma[..., 0]),
                np.ascontiguousarray(chroma[..., 1]),
                full_range=self.full_range,
                matrix=self.matrix,
            )
        if frame.format.name not in ("yuv420p", "yuvj420p"):
            frame = frame.reformat(format="yuv420p")
        return YUV420Frame.from_i420(frame.to_ndarray(), full_range=self.full_range, matrix=self.matrix)

    def get_yuv(self, frame_index: int) -> YUV420Frame:
        """The planes of an image.

        :raises IndexError: If the number lies outside the video.
        """
        if not 0 <= frame_index < self.frame_count:
            raise IndexError(f"Frame index {frame_index} out of range [0, {self.frame_count}) in {self.video_path}")
        if self._cached is not None and self._cached[0] == frame_index:
            return self._cached[1]
        if self._next_index is None or not 0 <= frame_index - self._next_index <= 30:
            self._seek(frame_index)
        assert self._frames is not None
        for frame in self._frames:
            index = int(round(frame.pts / self._pts_step))
            self._next_index = index + 1
            if index == frame_index:
                yuv = self._to_yuv(frame)
                self._cached = (frame_index, yuv)
                return yuv
            if index > frame_index:
                break
        self._next_index = None
        raise IndexError(f"Frame {frame_index} was not found in {self.video_path}.")

    def get_frame(self, frame_index: int, scale: Optional[int] = None) -> npt.NDArray[np.uint8]:
        """An image as (H, W, 3) uint8 RGB, optionally reduced by an integer factor.

        :param frame_index: The number of the image.
        :param scale: Optional reduction factor (2 for half width and height).
        """
        image = self.get_yuv(frame_index).to_rgb()
        if scale is not None and scale > 1:
            image = cv2.resize(image, (self.width // scale, self.height // scale), interpolation=cv2.INTER_AREA)
        return image

    def close(self) -> None:
        if self._container is not None:
            self._container.close()
            self._container = None


_READERS: "OrderedDict[Tuple[int, str, str], AV1Reader]" = OrderedDict()
_READERS_LOCK = threading.Lock()
_MAX_READERS = 64


def get_av1_reader(video_path: Union[str, Path], decoder: str = "libdav1d") -> AV1Reader:
    """A cached reader per process, file and decoder (at most 64; the least recently used one is dropped).

    The cache key includes the process id, so a forked process opens its own readers. Lock the reader's ``lock``
    while reading from several threads (:func:`read_av1_frame` does).
    """
    key = (os.getpid(), str(video_path), decoder)
    with _READERS_LOCK:
        reader = _READERS.get(key)
        if reader is not None:
            _READERS.move_to_end(key)
            return reader
    reader = AV1Reader(video_path, decoder=decoder)
    with _READERS_LOCK:
        existing = _READERS.get(key)
        if existing is not None:
            return existing
        _READERS[key] = reader
        while len(_READERS) > _MAX_READERS:
            _READERS.popitem(last=False)  # closed by garbage collection, after any running read
    return reader


def read_av1_frame(
    video_path: Union[str, Path], frame_index: int, scale: Optional[int] = None, decoder: str = "libdav1d"
) -> npt.NDArray[np.uint8]:
    """One RGB image of an AV1 video, through the cached reader of this process (thread-safe)."""
    reader = get_av1_reader(video_path, decoder=decoder)
    with reader.lock:
        yuv = reader.get_yuv(frame_index)
    image = yuv.to_rgb()
    if scale is not None and scale > 1:
        image = cv2.resize(image, (reader.width // scale, reader.height // scale), interpolation=cv2.INTER_AREA)
    return image
