"""Video I/O utilities for frame extraction and file discovery."""

import os
import logging
from contextlib import contextmanager

import av
import numpy as np

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv", ".webm", ".mov")


@contextmanager
def _open_video(video_path: str):
    """Context manager for safe PyAV container lifecycle."""
    container = av.open(video_path)
    try:
        yield container
    finally:
        container.close()


def read_video_pyav(video_path: str, num_frames: int) -> list[np.ndarray]:
    """
    Uniformly sample frames from a video using seek-based decoding.

    O(num_frames) complexity via seeking rather than sequential decode.

    Returns:
        List of (H, W, 3) uint8 numpy arrays in RGB24 format.

    Raises:
        ValueError: If the video has no determinable duration or no decodable frames.
    """
    with _open_video(video_path) as container:
        duration_us = container.duration
        if not duration_us or duration_us <= 0:
            raise ValueError(f"Cannot determine duration of {video_path}")

        sample_timestamps = np.linspace(
            0, duration_us, num_frames, endpoint=False
        ).astype(int)

        frames = []
        for timestamp in sample_timestamps:
            container.seek(int(timestamp))
            for frame in container.decode(video=0):
                frames.append(frame.to_ndarray(format="rgb24"))
                break

    if not frames:
        raise ValueError(f"No frames decoded from {video_path}")

    # Pad with last frame if some seeks failed to decode
    while len(frames) < num_frames:
        frames.append(frames[-1])

    return frames


def scan_video_directory(video_dir: str | list[str]) -> list[str]:
    """Recursively find all video files in one or more directories."""
    if isinstance(video_dir, str):
        video_dir = [video_dir]

    paths = []
    for directory in video_dir:
        for root, _, files in os.walk(directory):
            for filename in sorted(files):
                if filename.lower().endswith(VIDEO_EXTENSIONS):
                    paths.append(os.path.join(root, filename))

    paths.sort()
    return paths
