"""Shared test helpers: a small deterministic embedder, so tests need no GPU or downloads."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from lodestone.config import EMBEDDING_DIM, Paths, Settings
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


class FakeEmbedder:
    def __init__(self) -> None:
        self.image_calls = 0
        self.document_calls = 0
        self.released = 0
        self.fail_on_size: tuple[int, int] | None = None

    def encode_query(self, text: str) -> np.ndarray:
        return word_vector(text)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        self.document_calls += 1
        return np.stack([word_vector(t) for t in texts])

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        self.image_calls += 1
        if self.fail_on_size and any(i.size == self.fail_on_size for i in images):
            raise RuntimeError("bad image")
        vectors = []
        for image in images:
            # Images are labelled by their average colour, which tests can search for.
            r, g, b = np.asarray(image.convert("RGB")).reshape(-1, 3).mean(axis=0)
            label = "red" if r > g and r > b else "green" if g > b else "blue"
            vectors.append(word_vector(label))
        return np.stack(vectors)

    def release_index_model(self) -> None:
        self.released += 1


class FakeOcr:
    def __init__(self, text: str = "") -> None:
        self.text = text
        self.calls = 0

    def read(self, image: Image.Image) -> OcrResult:
        self.calls += 1
        return OcrResult(self.text, 0.9 if self.text else None)

    read_page = read


def textured(size: tuple[int, int], colour: tuple[int, int, int], seed: int = 0) -> Image.Image:
    """A sharp, non-blank image with a dominant colour."""
    rng = np.random.default_rng(seed)
    noise = rng.integers(-40, 40, (size[1], size[0], 3))
    pixels = np.clip(np.array(colour) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(pixels)


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
