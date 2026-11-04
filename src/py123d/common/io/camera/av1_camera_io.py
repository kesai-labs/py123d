"""AV1 video files of camera images (camera store option ``"av1"``).

:class:`AV1Writer` encodes the images of one camera into ``{key}.av1.mp4`` with NVIDIA's encoder (NVENC) or
SVT-AV1, and returns each image's number, which the camera's Arrow file stores. :class:`AV1Reader` returns the
image of a number: sequential reads decode the next image, other reads seek to the previous key frame. The
conversion to RGB uses the color range and matrix that the file declares (full-range BT.601 for JPEG sources).
"""

from __future__ import annotations

import logging
import os
import queue
import threading
from collections import OrderedDict
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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


# Options of a reused NVENC session: zero latency (each encode call returns the packet of its image; the bitstream is
# the same as with the default latency) and IDR key frames, so each video starts with a key frame and its sequence
# header.
_REUSE_OPTIONS = {"delay": "0", "zerolatency": "1", "forced-idr": "1"}


def _reuses_sessions(config: VideoCodecConfig, encoder: str) -> bool:
    return encoder == _ENCODERS["nvenc"] and config.nvenc_reuse_sessions


def _encoder_options(config: VideoCodecConfig, encoder: str) -> Dict[str, str]:
    if encoder != _ENCODERS["nvenc"]:
        return config.svt_options()
    options = config.nvenc_options()
    return {**options, **_REUSE_OPTIONS} if config.nvenc_reuse_sessions else options


def av1_codec_metadata(config: VideoCodecConfig, encoder: str) -> Dict[str, Any]:
    """The codec description written into the schema metadata of a camera's Arrow file (key ``codec``)."""
    from py123d import __version__

    description = {
        "codec": "av1",
        "version": AV1_FORMAT_VERSION,
        "encoder": encoder,
        "writer": f"py123d {__version__}",
        "frame_rate": config.frame_rate,
        "options": _encoder_options(config, encoder),
    }
    if _reuses_sessions(config, encoder):
        description["session"] = "reused"
    return description


def _set_color_tags(context: Any, frame: YUV420Frame) -> None:
    context.color_range = _COLOR_RANGE_JPEG if frame.full_range else _COLOR_RANGE_MPEG
    context.colorspace = AV_COLORSPACE[frame.matrix]
    if frame.matrix == "bt709":
        context.color_primaries = 1
        context.color_trc = 1


class _PooledEncoder:
    """An open NVENC session that writes one video after another (``nvenc_reuse_sessions``)."""

    def __init__(self, stream: Any) -> None:
        self.stream = stream  # the stream of the first video; its codec context is the encoder
        self.template: Any = None  # the first video's stream, opened for reading: the template of later videos
        self.template_container: Any = None
        self.next_pts = 0
        self.videos = 0  # videos started with this session


# Idle reused encoders of this process, per (width, height, full range, matrix, options, frame rate).
_POOL: Dict[Tuple[Any, ...], List[_PooledEncoder]] = {}
_POOL_LOCK = threading.Lock()
_POOL_PID = os.getpid()


def _pool() -> Dict[Tuple[Any, ...], List[_PooledEncoder]]:
    global _POOL_PID
    if os.getpid() != _POOL_PID:  # a forked process: the sessions of the parent process are not usable here
        _POOL.clear()
        _POOL_PID = os.getpid()
    return _POOL


class AV1Writer:
    """Writes the images of one camera into an AV1 video (MP4 container).

    The encoder opens with the first image, whose size and color tags all later images must share. :meth:`close`
    flushes the encoder and closes the file; it is safe to call more than once and releases the encoder also after
    an error.

    With ``nvenc_reuse_sessions`` in the configuration, the writer takes an idle NVENC session of its process for
    the same image size, color tags and settings (or opens one) and returns it on :meth:`close` instead of closing
    it. The video stream then is a mux-only copy of the session's first video stream; each video starts with a key
    frame, and :meth:`close` checks that the video received one packet per image. A session that failed is not
    reused. The rate control keeps its state from the previous video: on PhysicalAI clips (also night after day)
    the mean PSNR of a video differed by at most 0.05 dB from a fresh session and the size by at most 0.7 %; during
    the first 2 s of a video the PSNR differed by up to 0.4 dB.

    With ``threaded`` (the default), the images are encoded by one background thread per writer through a queue of
    ``queue_size`` images: an encode call to NVENC returns only after the image is processed (about 0.1 s per image
    with many sessions per GPU), so a parser that writes several cameras would otherwise wait for each image in turn.
    The images of one writer keep their order, so the video is the same. An error of the background thread is raised
    by the next :meth:`write_frame` or by :meth:`close`.
    """

    def __init__(
        self,
        output_path: Union[str, Path],
        config: Optional[VideoCodecConfig] = None,
        encoder: Optional[str] = None,
        threaded: bool = True,
        queue_size: int = 8,
    ) -> None:
        """
        :param output_path: The path of the video file (``{key}.av1.mp4``).
        :param config: Encoder settings; defaults to :class:`VideoCodecConfig`.
        :param encoder: ``"av1_nvenc"`` or ``"libsvtav1"``; resolved from the configuration when None.
        :param threaded: Encode in a background thread (see the class documentation).
        :param queue_size: Images waiting for the background thread at most.
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
        self._threaded = bool(threaded)
        self._queue: Any = queue.Queue(maxsize=max(1, int(queue_size))) if self._threaded else None
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[BaseException] = None
        self._reuse = _reuses_sessions(self.config, self.encoder)
        self._pooled: Optional[_PooledEncoder] = None
        self._pool_key: Optional[Tuple[Any, ...]] = None
        self._packets = 0

    def __enter__(self) -> "AV1Writer":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def _open(self, frame: YUV420Frame) -> None:
        if self._reuse:
            self._open_pooled(frame)
            return
        self._open_fresh(frame)

    def _open_pooled(self, frame: YUV420Frame) -> None:
        import av

        options = _encoder_options(self.config, self.encoder)
        key = (
            frame.width,
            frame.height,
            frame.full_range,
            frame.matrix,
            tuple(sorted(options.items())),
            self.config.frame_rate,
        )
        with _POOL_LOCK:
            idle = _pool().setdefault(key, [])
            pooled = idle.pop() if idle else None
        if pooled is None:
            self._open_fresh(frame)  # a new session; its first video is written through its own stream
            self._pooled = _PooledEncoder(self._stream)
        else:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            container = av.open(str(self.output_path), "w", format="mp4")
            try:
                stream = container.add_stream_from_template(pooled.template, opaque=True)
                stream.time_base = Fraction(1, self.config.frame_rate)
                _set_color_tags(stream.codec_context, frame)
            except Exception:
                container.close()
                with _POOL_LOCK:
                    _pool().setdefault(key, []).append(pooled)
                raise
            self._container, self._stream, self._pooled = container, stream, pooled
            self._format = (frame.width, frame.height, frame.full_range, frame.matrix)
        self._pool_key = key

    def _open_fresh(self, frame: YUV420Frame) -> None:
        import av

        if self.encoder == _ENCODERS["svt"]:
            os.environ.setdefault("SVT_LOG", "2")  # SVT-AV1 prints its configuration at level 3 and up
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        container = av.open(str(self.output_path), "w", format="mp4")
        try:
            stream = container.add_stream(self.encoder, rate=self.config.frame_rate)
            stream.width, stream.height, stream.pix_fmt = frame.width, frame.height, "yuv420p"
            stream.time_base = Fraction(1, self.config.frame_rate)
            _set_color_tags(stream.codec_context, frame)
            stream.options = _encoder_options(self.config, self.encoder)
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
        if self._closed:
            raise RuntimeError(f"The AV1 writer of {self.output_path} is closed.")
        self._raise_background_error()
        yuv = frame if isinstance(frame, YUV420Frame) else YUV420Frame.from_rgb(frame)
        if self._container is None:
            self._open(yuv)
        if (yuv.width, yuv.height, yuv.full_range, yuv.matrix) != self._format:
            raise ValueError(
                f"Image {(yuv.width, yuv.height, yuv.full_range, yuv.matrix)} differs from the video {self._format}."
            )
        index = self.frame_count
        self.frame_count += 1
        if not self._threaded:
            self._encode(index, yuv)
            return index
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name=f"av1-{self.output_path.name}", daemon=True)
            self._thread.start()
        self._queue.put((index, yuv))
        return index

    def _encode(self, index: int, yuv: YUV420Frame) -> None:
        import av

        video_frame = av.VideoFrame.from_ndarray(yuv.to_i420(), format="yuv420p")
        time_base = Fraction(1, self.config.frame_rate)
        video_frame.time_base = time_base
        video_frame.color_range = _COLOR_RANGE_JPEG if yuv.full_range else _COLOR_RANGE_MPEG
        video_frame.colorspace = AV_COLORSPACE[yuv.matrix]
        if self._pooled is None:
            video_frame.pts = index
            for packet in self._stream.encode(video_frame):
                self._container.mux(packet)
            return
        # A reused session: its own increasing time stamps, a key frame at the video start, and per video the packet
        # time stamps 0, 1, 2, ... (one packet per image at once: zero latency, no B-frames).
        video_frame.pts = self._pooled.next_pts
        self._pooled.next_pts += 1
        encoder = self._pooled.stream.codec_context
        if index == 0:
            video_frame.pict_type = av.video.frame.PictureType.I
            self._pooled.videos += 1
        for packet in encoder.encode(video_frame):
            packet.stream = self._stream
            packet.time_base = time_base
            packet.pts = packet.dts = self._packets
            packet.duration = 1
            self._container.mux(packet)
            self._packets += 1

    def _run(self) -> None:
        """The background thread: encodes the queued images in order; after an error it only empties the queue."""
        while True:
            item = self._queue.get()
            if item is None:
                return
            if self._error is None:
                try:
                    self._encode(*item)
                except BaseException as e:  # noqa: BLE001 - raised again by write_frame or close
                    self._error = e

    def _raise_background_error(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"Encoding {self.output_path} failed: {self._error}") from self._error

    def _stop_thread(self) -> None:
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join()
            self._thread = None

    def close(self) -> None:
        """Flush the encoder and close the file (no effect when already closed)."""
        if self._closed:
            return
        self._closed = True
        try:
            self._stop_thread()
        finally:
            if self._container is not None:
                if self._pooled is None:
                    self._finish()
                else:
                    self._finish_pooled()

    def _finish_pooled(self) -> None:
        """Close the video and return the session to the pool (not after an error or a missing packet)."""
        import av

        pooled, key = self._pooled, self._pool_key
        reusable = False
        try:
            self._raise_background_error()
            if self._packets != self.frame_count:
                raise RuntimeError(
                    f"{self.output_path}: {self._packets} packets for {self.frame_count} images (reused NVENC session)."
                )
            reusable = True
        finally:
            self._container.close()
            self._container = None
            self._stream = None
            self._pooled = None
            if reusable and pooled is not None:
                if pooled.template is None:
                    pooled.template_container = av.open(str(self.output_path))
                    pooled.template = pooled.template_container.streams.video[0]
                with _POOL_LOCK:
                    _pool().setdefault(key, []).append(pooled)

    def _finish(self) -> None:
        try:
            self._raise_background_error()
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
