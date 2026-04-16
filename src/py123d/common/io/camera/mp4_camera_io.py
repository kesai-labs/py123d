from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np


class MP4Writer:
    """Simple implementation of an MP4 video writer, based on OpenCV."""

    def __init__(self, output_path: Union[str, Path], fps: float = 30.0, codec: str = "mp4v"):
        """
        Initialize MP4 writer.

        :param output_path: The output path for the MP4 file.
        :param fps: Frames per second, defaults to 30.0
        :param codec: Video codec, defaults to "mp4v"
        """
        self.output_path = Path(output_path)
        self.fps = fps
        self.codec = codec
        self.writer = None
        self.frame_size = None
        self.frame_count = 0

    def write_frame(self, frame: np.ndarray) -> int:
        """Write a single frame to the video.

        :param frame: Image as numpy array (RGB format)
        :raises ValueError: If frame size does not match video size
        :return: Index of the written frame
        """
        frame_idx = int(self.frame_count)
        if self.writer is None:
            # Initialize writer with first frame's dimensions
            h, w = frame.shape[:2]
            self.frame_size = (w, h)
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = cv2.VideoWriter(str(self.output_path), fourcc, self.fps, self.frame_size)

        if frame.shape[:2][::-1] != self.frame_size:
            raise ValueError(f"Frame size {frame.shape[:2][::-1]} doesn't match video size {self.frame_size}")

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.writer.write(frame)
        self.frame_count += 1
        return frame_idx

    def close(self):
        """Close the video writer and finalize the MP4 file."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None


class MP4Reader:
    """Simple implementation of an MP4 video reader, based on OpenCV."""

    def __init__(self, video_path: Union[str, Path], read_all: bool = False):
        """Initializes the MP4Reader.

        :param video_path: Path to the MP4 video file.
        :param read_all: Whether to read all frames into memory, defaults to False
        :raises FileNotFoundError: If the video file does not exist
        :raises ValueError: If the video file cannot be opened
        """
        self.video_path = video_path
        self.cap = None
        self.read_all = read_all
        self.frames = []

        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        # Get video properties
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if read_all:
            for _ in range(self.frame_count):
                ret, frame = self.cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frames.append(frame)
            self.cap.release()
            self.cap = None

    def get_frame(self, frame_index: int) -> Optional[np.ndarray]:
        """Get a specific frame, an RBG image as numpy array, by its index.

        :param frame_index: Index of the frame to retrieve.
        :raises IndexError: If the frame index is out of range.
        :return: The frame as a numpy array (RGB format) or None if the frame could not be read.
        """

        if frame_index < 0 or frame_index >= self.frame_count:
            raise IndexError(f"Frame index {frame_index} out of range [0, {self.frame_count})")

        if self.read_all:
            return self.frames[frame_index]

        # Set the frame position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        # Convert BGR to RGB for sane convention
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return frame if ret else None

    def __del__(self):
        """Destructor to release the video capture."""
        if self.cap is not None:
            self.cap.release()
        if self.read_all:
            self.frames = []


@lru_cache(maxsize=64)
def get_mp4_reader_from_path(mp4_path: str) -> MP4Reader:
    return MP4Reader(mp4_path, read_all=False)
