"""Shared test helpers: a small deterministic embedder, so tests need no GPU or downloads."""

from __future__ import annotations

import hashlib
import re
import wave
from pathlib import Path
from typing import Callable

import av
import numpy as np
import pytest
from PIL import Image

from lodestone.config import EMBEDDING_DIM, EmbeddingModel, Paths, Settings
from lodestone.ocr import OcrResult


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-9)


def word_vector(text: str) -> np.ndarray:
    """Bag of hashed words: texts that share words end up similar, like a real embedding."""
    vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    for word in re.findall(r"\w+", text.lower()):
        slot = int(hashlib.md5(word.encode()).hexdigest(), 16) % EMBEDDING_DIM
        vector[slot] += 1.0
    if not vector.any():
        vector[0] = 1.0
    return _unit(vector)


def _colour_label(image: Image.Image) -> str:
    """Pictures are labelled by their average colour, which tests can search for."""
    r, g, b = np.asarray(image.convert("RGB")).reshape(-1, 3).mean(axis=0)
    return "red" if r > g and r > b else "green" if g > b else "blue"


class FakeEmbedder:
    def __init__(self) -> None:
        self.image_calls = 0
        self.document_calls = 0
        self.video_lengths: list[int] = []
        self.audio_lengths: list[int] = []
        self.released = 0
        self.search_releases = 0
        self.gpu_settings: list[tuple[str, bool]] = []
        self.warm_ups = 0
        self.search_model_loaded = True
        self.fail_on_size: tuple[int, int] | None = None
        self.model = EmbeddingModel()
        self.inspected: EmbeddingModel | Exception | None = None

    def supports(self, kind: str) -> bool:
        return kind in self.model.kinds

    def set_model(self, model: EmbeddingModel) -> None:
        self.model = model

    def inspect_model(self, model_id: str) -> EmbeddingModel:
        if isinstance(self.inspected, Exception):
            raise self.inspected
        return self.inspected or EmbeddingModel(id=model_id)

    def encode_query(self, text: str) -> np.ndarray:
        return word_vector(text)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        self.document_calls += 1
        return np.stack([word_vector(t) for t in texts])

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        self.image_calls += 1
        if self.fail_on_size and any(i.size == self.fail_on_size for i in images):
            raise RuntimeError("bad image")
        return np.stack([word_vector(_colour_label(image)) for image in images])

    def encode_video(self, frames: list[Image.Image]) -> np.ndarray:
        self.video_lengths.append(len(frames))
        return np.stack([word_vector(_colour_label(frames[0]))])

    def encode_audio(self, samples: np.ndarray) -> np.ndarray:
        self.audio_lengths.append(len(samples))
        return np.stack([word_vector("humming tone")])

    def release_index_model(self) -> None:
        self.released += 1

    def release_search_model_if_idle(self) -> None:
        self.search_releases += 1

    def set_gpu_for_search(self, use_gpu: bool) -> None:
        self.gpu_settings.append(("search", use_gpu))

    def set_gpu_for_indexing(self, use_gpu: bool) -> None:
        self.gpu_settings.append(("indexing", use_gpu))

    def warm_up(self) -> None:
        self.warm_ups += 1


class FakeOcr:
    def __init__(self, text: str = "") -> None:
        self.text = text
        self.calls = 0

    def read(self, image: Image.Image) -> OcrResult:
        self.calls += 1
        return OcrResult(self.text, 0.9 if self.text else None)

    read_page = read

    def release(self) -> None:
        self.released = True


def textured(size: tuple[int, int], colour: tuple[int, int, int], seed: int = 0) -> Image.Image:
    """A sharp, non-blank image with a dominant colour."""
    rng = np.random.default_rng(seed)
    noise = rng.integers(-40, 40, (size[1], size[0], 3))
    pixels = np.clip(np.array(colour) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(pixels)


def write_wav(path: Path, seconds: float, amplitude: float = 0.3, rate: int = 44_100, channels: int = 2) -> None:
    """A real sound file at a rate and layout that the app has to convert."""
    count = int(seconds * rate)
    tone = amplitude * np.sin(2 * np.pi * 220 * np.arange(count) / rate)
    samples = np.repeat((tone * 32767).astype(np.int16), channels)
    with wave.open(str(path), "wb") as file:
        file.setnchannels(channels)
        file.setsampwidth(2)
        file.setframerate(rate)
        file.writeframes(samples.tobytes())


def write_video(
    path: Path,
    seconds: int,
    colour_at: Callable[[float], tuple[int, int, int]],
    with_sound: bool = False,
    fps: int = 10,
) -> None:
    """A real video file. It has noise in it, because blank frames get skipped."""
    rng = np.random.default_rng(0)
    with av.open(str(path), "w") as container:
        video = container.add_stream("mpeg4", rate=fps)
        video.width, video.height, video.pix_fmt = 320, 240, "yuv420p"
        # A keyframe only every 5 seconds, so samples between them have to be decoded to.
        video.gop_size = fps * 5
        audio = container.add_stream("aac", rate=44_100) if with_sound else None
        for index in range(seconds * fps):
            noise = rng.integers(-30, 30, (240, 320, 3))
            pixels = np.clip(np.array(colour_at(index / fps)) + noise, 0, 255).astype(np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = index
            container.mux(video.encode(frame))
        container.mux(video.encode())
        if audio is not None:
            count = 44_100 * seconds
            tone = 0.3 * np.sin(2 * np.pi * 220 * np.arange(count) / 44_100).astype(np.float32)
            for start in range(0, count, 1024):
                part = tone[start : start + 1024].reshape(1, -1).astype(np.float32)
                frame = av.AudioFrame.from_ndarray(part, format="fltp", layout="mono")
                frame.sample_rate = 44_100
                frame.pts = start
                container.mux(audio.encode(frame))
            container.mux(audio.encode())


@pytest.fixture
def paths(tmp_path: Path) -> Paths:
    root = tmp_path / "data"
    root.mkdir()
    return Paths(root)


@pytest.fixture
def library(tmp_path: Path) -> Path:
    folder = tmp_path / "library"
    folder.mkdir()
    return folder


@pytest.fixture
def fast_settings() -> Settings:
    return Settings(debounce_seconds=0.0, delete_delay_seconds=0.0, reindex_cooldown_seconds=0.0)
