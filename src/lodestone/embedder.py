"""EmbeddingGemma 2, loaded as two instances so search stays light and is never blocked.

Search keeps only the text encoder loaded. Indexing loads text and vision while it has work, then
frees the GPU. Both produce vectors in the same space, so they can be used interchangeably.
"""

from __future__ import annotations

import ctypes
import gc
import logging
import os
import threading
from contextlib import contextmanager
from typing import Iterator

import numpy as np
from PIL import Image

from .config import MODEL_ID, SETTINGS, Settings

log = logging.getLogger(__name__)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def _unit(vectors: np.ndarray) -> np.ndarray:
    """Normalize in float32. Doing it in the model's bfloat16 leaves lengths visibly off 1.0."""
    values = np.asarray(vectors, dtype=np.float32)
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norm, 1e-9)


def _return_unused_ram() -> None:
    """Let Windows take back pages the app isn't using, such as ones read while loading a model."""
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        ctypes.windll.psapi.EmptyWorkingSet.argtypes = [ctypes.c_void_p]
        ctypes.windll.psapi.EmptyWorkingSet(kernel32.GetCurrentProcess())
    except (AttributeError, OSError):
        pass


class _LoadGate:
    """Model work can overlap with other model work, but never with a model loading."""

    def __init__(self) -> None:
        self._changed = threading.Condition()
        self._workers = 0
        self._is_loading = False

    @contextmanager
    def working(self) -> Iterator[None]:
        with self._changed:
            while self._is_loading:
                self._changed.wait()
            self._workers += 1
        try:
            yield
        finally:
            with self._changed:
                self._workers -= 1
                self._changed.notify_all()

    @contextmanager
    def loading(self) -> Iterator[None]:
        with self._changed:
            while self._is_loading or self._workers:
                self._changed.wait()
            self._is_loading = True
        try:
            yield
        finally:
            with self._changed:
                self._is_loading = False
                self._changed.notify_all()


class GemmaEmbedder:
    def __init__(self, settings: Settings = SETTINGS) -> None:
        self._settings = settings
        self._query_model = None
        self._index_model = None
        self._query_lock = threading.Lock()
        self._index_lock = threading.Lock()
        self._gate = _LoadGate()

    def _load(self, with_vision: bool):
        # Imported here so the UI opens instantly, before these heavy libraries load.
        import torch
        from sentence_transformers import SentenceTransformer

        config = {"audio_config": None}
        if not with_vision:
            config["vision_config"] = None
        cuda = torch.cuda.is_available()
        device = "cuda" if cuda else "cpu"
        if cuda:
            torch.cuda.set_per_process_memory_fraction(self._settings.max_vram_fraction)
        # Google warns that float16 can silently degrade embeddings, so never use it. Loading
        # straight into bfloat16 used over 1 GB less RAM than loading as float32 and converting.
        dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else torch.float32
        arguments = dict(device=device, config_kwargs=config, model_kwargs={"dtype": dtype})
        # Loading changes PyTorch's default dtype for the whole process, which once broke every
        # image by giving a model mismatched parts. The gate keeps all other model work out.
        with self._gate.loading():
            try:
                # After the first download this is a local tool, so don't contact the network.
                model = SentenceTransformer(MODEL_ID, local_files_only=True, **arguments)
            except OSError:
                log.info("downloading %s", MODEL_ID)
                model = SentenceTransformer(MODEL_ID, **arguments)
        model.max_seq_length = self._settings.max_text_tokens
        return model

    def warm_up(self) -> None:
        """Load the query model ahead of time, so the first search does not pay for it."""
        self.encode_query("warm up")
        _return_unused_ram()

    def encode_query(self, text: str) -> np.ndarray:
        with self._query_lock:
            if self._query_model is None:
                self._query_model = self._load(with_vision=False)
            with self._gate.working():
                return _unit(
                    self._query_model.encode(text, prompt_name="SearchQuery", show_progress_bar=False)
                )

    def _indexing_model(self):
        if self._index_model is None:
            self._index_model = self._load(with_vision=True)
        return self._index_model

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        import torch

        with self._index_lock:
            model = self._indexing_model()
            batch_size = self._settings.text_batch_size
            while True:
                try:
                    with self._gate.working():
                        vectors = model.encode(
                            texts, prompt_name="Document", batch_size=batch_size, show_progress_bar=False
                        )
                    break
                except torch.cuda.OutOfMemoryError:
                    # Something else may be using the GPU, so smaller batches can still fit.
                    torch.cuda.empty_cache()
                    if batch_size == 1:
                        raise
                    batch_size //= 2
            self._return_spare_vram()
            return _unit(vectors)

    def _return_spare_vram(self) -> None:
        """PyTorch keeps freed memory for itself, so hand it back after unusually big batches."""
        import torch

        if torch.cuda.is_available() and torch.cuda.memory_reserved() > 3 * 1024**3:
            torch.cuda.empty_cache()

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        with self._index_lock:
            model = self._indexing_model()
            with self._gate.working():
                return _unit(
                    model.encode(
                        [{"image": image} for image in images],
                        batch_size=self._settings.image_batch_size,
                        processing_kwargs={"image": {"max_soft_tokens": self._settings.image_tokens}},
                        show_progress_bar=False,
                    )
                )

    def release_index_model(self) -> None:
        with self._index_lock:
            if self._index_model is None:
                return
            self._index_model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        _return_unused_ram()
        log.info("released the indexing model")
