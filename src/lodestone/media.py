"""Decode the opening of audio and video files with PyAV, as far as the duration settings ask for."""

from __future__ import annotations

import logging
from typing import Iterator

import av
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000  # The audio encoder expects mono at this rate.
# Past this gap, seeking to the next keyframe is cheaper than decoding every frame in between.
_SEEK_GAP_SECONDS = 2.0
DecodeError = av.FFmpegError


def read_audio(path: str, max_seconds: float) -> np.ndarray | None:
    """The opening of the first audio stream as 16 kHz mono floats, or None if there isn't one."""
    wanted = int(max_seconds * SAMPLE_RATE)
    chunks: list[np.ndarray] = []
    have = 0

    def collect(frames) -> None:
        nonlocal have
        for frame in frames:
            samples = frame.to_ndarray().reshape(-1)
            chunks.append(samples)
            have += samples.size

    with av.open(path) as container:
        if not container.streams.audio:
            return None
        resampler = av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)
        try:
            for frame in container.decode(container.streams.audio[0]):
                collect(resampler.resample(frame))
                if have >= wanted:
                    break
            else:
                collect(resampler.resample(None))  # Whatever the resampler still holds.
        except av.FFmpegError as error:
            # Damaged partway through: what came before it is still worth searching.
            if not chunks:
                raise
            log.info("kept the readable start of %s: %s", path, error)
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks)[:wanted].astype(np.float32)


def read_frames(
    path: str, max_seconds: float, max_frames: int, frames_per_second: float, long_side: int, blank_std: float
) -> list[Image.Image]:
    """Frames spread evenly across the opening of a video, skipping blank ones."""
    with av.open(path) as container:
        stream = _picture_stream(container)
        if stream is None:
            return []
        stream.thread_type = "AUTO"
        frames: list[Image.Image] = []
        try:
            times = _sample_times(container, stream, max_seconds, max_frames, frames_per_second)
            for frame in _frames_at(container, stream, times):
                image = _to_image(frame, long_side)
                if float(np.asarray(image).std()) >= blank_std:
                    frames.append(image)
        except av.FFmpegError as error:
            if not frames:
                raise
            log.info("kept the readable frames of %s: %s", path, error)
    return frames


def _picture_stream(container):
    """Skips cover art, which is stored as a one-frame video stream."""
    attached = av.stream.Disposition.attached_pic
    return next((s for s in container.streams.video if not s.disposition & attached), None)


def _duration(container, stream) -> float | None:
    if stream.duration is not None and stream.time_base is not None:
        return float(stream.duration * stream.time_base)
    if container.duration is not None:
        return container.duration / av.time_base
    return None


def _sample_times(
    container, stream, max_seconds: float, max_frames: int, frames_per_second: float
) -> list[float]:
    start = float(stream.start_time * stream.time_base) if stream.start_time is not None else 0.0
    duration = _duration(container, stream)
    # Some files don't record their length. Times past the end simply find no frame.
    window = min(duration, max_seconds) if duration and duration > 0 else max_seconds
    count = min(max_frames, max(1, int(window * frames_per_second)))
    # The middle of each equal slice, which also avoids the black frames videos often start with.
    return [start + (i + 0.5) * window / count for i in range(count)]


def _frames_at(container, stream, times: list[float]) -> Iterator[av.VideoFrame]:
    decoder = container.decode(stream)
    position: float | None = None
    for target in times:
        if (position if position is not None else 0.0) + _SEEK_GAP_SECONDS < target:
            container.seek(int(target / stream.time_base), stream=stream)
            decoder = container.decode(stream)
        for frame in decoder:
            position = frame.time
            if position is None or position >= target:
                yield frame
                break


def _to_image(frame: av.VideoFrame, long_side: int) -> Image.Image:
    scale = min(1.0, long_side / max(frame.width, frame.height))
    image = frame.reformat(
        width=max(1, round(frame.width * scale)),
        height=max(1, round(frame.height * scale)),
        format="rgb24",
    ).to_image()
    # Phones store portrait video sideways, with a note saying how to turn it.
    return image.rotate(frame.rotation, expand=True) if frame.rotation else image
