"""The embedding model, loaded as two instances so search stays light and is never blocked.

With EmbeddingGemma 2, the default, search only needs the text encoder. Indexing loads text and
vision while it has work, and adds the audio encoder only once it meets a file with sound. All of
them produce vectors in the same space, and all unload when idle. Everything runs on the CPU unless
the user turns on the GPU, because search is about as fast either way and the GPU option costs VRAM.

Other models can be chosen in Settings. They load whole, and inspect_model finds out what they can do.
"""

from __future__ import annotations

import ctypes
import gc
import logging
import os
import threading
import time
from contextlib import contextmanager
from typing import Generator

import numpy as np
from PIL import Image

from .config import DEFAULT_MODEL, MODEL_ID, SETTINGS, EmbeddingModel, Settings
from .media import SAMPLE_RATE

log = logging.getLogger(__name__)
_MODEL_MAX_TOKENS = 8192  # What one embedding can hold, across everything in it.
# The names models commonly give their prompts, best first.
_QUERY_PROMPTS = ("SearchQuery", "query", "search_query")
_DOCUMENT_PROMPTS = ("Document", "document", "passage", "search_document")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def _unit(vectors: np.ndarray) -> np.ndarray:
    """Normalize in float32. Doing it in the model's bfloat16 leaves lengths visibly off 1.0."""
    values = np.asarray(vectors, dtype=np.float32)
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norm, 1e-9)


def _gives_vector(encode, dim: int) -> bool:
    try:
        vectors = encode()
    except Exception as error:
        log.info("this model can't be used for that content: %s", error)
        return False
    return vectors.shape == (1, dim) and bool(np.isfinite(vectors).all())


def _return_unused_ram() -> None:
    """Let Windows take back pages the app isn't using, such as ones read while loading a model."""
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        ctypes.windll.psapi.EmptyWorkingSet.argtypes = [ctypes.c_void_p]
        ctypes.windll.psapi.EmptyWorkingSet(kernel32.GetCurrentProcess())
    except (AttributeError, OSError):
        pass


def _free_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    _return_unused_ram()


class _LoadGate:
    """Model work can overlap with other model work, but never with a model loading."""

    def __init__(self) -> None:
        self._changed = threading.Condition()
        self._workers = 0
        self._is_loading = False

    @contextmanager
    def working(self) -> Generator[None, None, None]:
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
    def loading(self) -> Generator[None, None, None]:
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
    def __init__(
        self,
        settings: Settings = SETTINGS,
        gpu_for_search: bool = False,
        gpu_for_indexing: bool = False,
        model: EmbeddingModel = DEFAULT_MODEL,
    ) -> None:
        self._settings = settings
        self._gpu_for_search = gpu_for_search
        self._gpu_for_indexing = gpu_for_indexing
        self._model = model
        self._query_model = None
        self._index_model = None
        self._index_has_audio = False
        self._last_query = 0.0
        self._query_lock = threading.Lock()
        self._index_lock = threading.Lock()
        self._gate = _LoadGate()

    def _load(self, with_vision: bool, with_audio: bool = False, use_gpu: bool = False):
        # Imported here so the UI opens instantly, before these heavy libraries load.
        import torch
        from sentence_transformers import SentenceTransformer

        model_id = self._model.id
        # These savings were measured on Gemma, and other models may not have these parts to leave out.
        is_gemma = model_id == MODEL_ID
        config = {}
        if is_gemma and not with_audio:
            config["audio_config"] = None
        if is_gemma and not with_vision:
            config["vision_config"] = None
        cuda = use_gpu and torch.cuda.is_available()
        # Google warns that float16 can silently degrade embeddings, so never use it.
        if cuda:
            torch.cuda.set_per_process_memory_fraction(self._settings.max_vram_fraction)
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        else:
            # Half the cores cost little speed and leave the laptop usable. Without bfloat16
            # hardware, it was 5x slower for indexing, so only search uses it, to save RAM.
            torch.set_num_threads(self._settings.cpu_threads)
            dtype = torch.bfloat16 if is_gemma and not with_vision else torch.float32
        arguments = dict(
            device="cuda" if cuda else "cpu", config_kwargs=config, model_kwargs={"dtype": dtype}
        )
        # Loading changes PyTorch's default dtype for the whole process, which once broke every
        # image by giving a model mismatched parts. The gate keeps all other model work out.
        with self._gate.loading():
            try:
                # After the first download this is a local tool, so don't contact the network.
                model = SentenceTransformer(model_id, local_files_only=True, **arguments)
            except OSError:
                log.info("downloading %s", model_id)
                model = SentenceTransformer(model_id, **arguments)
        # A model's own limit can be lower than ours, and going past it is an error.
        model.max_seq_length = min(model.max_seq_length or self._settings.max_text_tokens, self._settings.max_text_tokens)
        return model

    # ---- settings and lifecycle ---------------------------------------------------------------

    @property
    def search_model_loaded(self) -> bool:
        return self._query_model is not None

    def set_gpu_for_search(self, use_gpu: bool) -> None:
        """Drop the loaded model, so it reloads on the right device when next needed."""
        with self._query_lock:
            if use_gpu == self._gpu_for_search:
                return
            self._gpu_for_search = use_gpu
            self._query_model = None
        _free_memory()

    def set_gpu_for_indexing(self, use_gpu: bool) -> None:
        """Drop the loaded model, so it reloads on the right device when next needed."""
        with self._index_lock:
            if use_gpu == self._gpu_for_indexing:
                return
            self._gpu_for_indexing = use_gpu
            self._index_model = None
            self._index_has_audio = False
        _free_memory()

    def supports(self, kind: str) -> bool:
        return kind in self._model.kinds

    def set_model(self, model: EmbeddingModel) -> None:
        """Drop the loaded models, so the new one loads when next needed."""
        with self._query_lock, self._index_lock:
            self._model = model
            self._query_model = None
            self._index_model = None
            self._index_has_audio = False
        _free_memory()

    def inspect_model(self, model_id: str) -> EmbeddingModel:
        """Download a model if needed, then try each kind of content through the real code paths.

        A model can list a kind that it only accepts in some other form, so listing isn't enough.
        Never runs code from the model's repository, so models that need that fail here.
        """
        probe = GemmaEmbedder(self._settings, model=EmbeddingModel(id=model_id))
        probe._gate = self._gate  # Loading affects the whole process, so it has to wait its turn.
        try:
            loaded = probe._indexing_model()
        except Exception as error:
            raise ValueError(f"Couldn't load {model_id}: {error}") from error
        try:
            if "text" not in loaded.modalities:
                raise ValueError(f"{model_id} can't read text, which search needs.")
            prompts = loaded.prompts or {}
            probe._model = EmbeddingModel(
                id=model_id,
                query_prompt=next((p for p in _QUERY_PROMPTS if p in prompts), None),
                document_prompt=next((p for p in _DOCUMENT_PROMPTS if p in prompts), None),
            )
            try:
                vectors = probe.encode_documents(["A short sentence to test the model."])
            except Exception as error:
                raise ValueError(f"{model_id} couldn't embed text: {error}") from error
            dim = int(vectors.shape[-1])
            picture = Image.new("RGB", (256, 256), (128, 128, 128))
            sound = np.random.default_rng(0).normal(0, 0.1, SAMPLE_RATE).astype(np.float32)
            tries = {
                "image": lambda: probe.encode_images([picture]),
                "audio": lambda: probe.encode_audio(sound),
                "video": lambda: probe.encode_video([picture] * 2),
            }
            kinds = ["text"]
            for kind, encode in tries.items():
                if kind in loaded.modalities and _gives_vector(encode, dim):
                    kinds.append(kind)
            return EmbeddingModel(
                model_id, dim, tuple(kinds), probe._model.query_prompt, probe._model.document_prompt
            )
        finally:
            del loaded
            probe._index_model = None
            _free_memory()

    def warm_up(self) -> None:
        """Load the search model ahead of time, so the next search does not pay for it."""
        if self.search_model_loaded:
            self._last_query = time.monotonic()
            return
        self.encode_query("warm up")
        _return_unused_ram()

    def release_search_model_if_idle(self) -> None:
        if (
            self._query_model is None
            or time.monotonic() - self._last_query <= self._settings.search_idle_unload_seconds
        ):
            return
        # If a search is running right now, then the model isn't idle.
        if not self._query_lock.acquire(blocking=False):
            return
        try:
            self._query_model = None
        finally:
            self._query_lock.release()
        _free_memory()
        log.info("released the search model")

    def release_index_model(self) -> None:
        with self._index_lock:
            if self._index_model is None:
                return
            self._index_model = None
            self._index_has_audio = False
        _free_memory()
        log.info("released the indexing model")

    # ---- embedding ------------------------------------------------------------------------------

    def encode_query(self, text: str) -> np.ndarray:
        with self._query_lock:
            if self._query_model is None:
                self._query_model = self._load(with_vision=False, use_gpu=self._gpu_for_search)
            with self._gate.working():
                vector = _unit(
                    self._query_model.encode(
                        text, prompt_name=self._model.query_prompt, show_progress_bar=False
                    )
                )
            # Idle time starts when the search finishes. It can't unload during one, as this holds the lock.
            self._last_query = time.monotonic()
            return vector

    def _indexing_model(self, with_audio: bool = False):
        if with_audio and not self._index_has_audio:
            # The audio encoder is the largest part, so most runs shouldn't pay for it.
            self._index_model = None
            _free_memory()
            self._index_model = self._load(True, True, self._gpu_for_indexing)
            self._index_has_audio = True
        elif self._index_model is None:
            self._index_model = self._load(True, False, self._gpu_for_indexing)
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
                            texts,
                            prompt_name=self._model.document_prompt,
                            batch_size=batch_size,
                            show_progress_bar=False,
                        )
                    break
                except torch.cuda.OutOfMemoryError:
                    # Something else may be using the GPU, so smaller batches can still fit.
                    torch.cuda.empty_cache()
                    if batch_size == 1:
                        raise
                    batch_size //= 2
            # PyTorch keeps freed VRAM for itself, so hand it back after unusually big batches.
            if torch.cuda.is_initialized() and torch.cuda.memory_reserved() > 3 * 1024**3:
                torch.cuda.empty_cache()
            return _unit(vectors)

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

    def encode_video(self, frames: list[Image.Image]) -> np.ndarray:
        """One vector for all of the frames, as an array holding one row."""
        with self._index_lock:
            model = self._indexing_model()
            # When memory runs short, every other frame still covers the same stretch of video.
            return self._encode_media(
                model,
                frames,
                lambda part: (
                    [{"video": part}],
                    {
                        "video": {
                            "max_soft_tokens": self._settings.video_frame_tokens,
                            "max_frames": len(part),
                            "do_sample_frames": False,  # media.py already chose these frames.
                        }
                    },
                ),
                lambda part: part[::2],
            )

    def encode_audio(self, samples: np.ndarray) -> np.ndarray:
        """One vector for a clip of 16 kHz mono samples, as an array holding one row."""
        with self._index_lock:
            model = self._indexing_model(with_audio=True)
            return self._encode_media(
                model,
                samples,
                lambda part: (
                    [{"audio": {"array": part, "sampling_rate": SAMPLE_RATE}}],
                    # Otherwise anything past 30 seconds is silently cut off.
                    {"audio": {"max_length": len(part)}},
                ),
                lambda part: part[: len(part) // 2],
            )

    def _encode_media(self, model, content, prepare, shrink) -> np.ndarray:
        import torch

        while True:
            inputs, kwargs = prepare(content)
            try:
                with self._gate.working():
                    vectors = model.encode(
                        inputs,
                        # The usual limit is for text, and going over it here is an error.
                        processing_kwargs={**kwargs, "text": {"max_length": _MODEL_MAX_TOKENS}},
                        show_progress_bar=False,
                    )
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if len(content) <= 1:
                    raise
                content = shrink(content)
        if torch.cuda.is_initialized() and torch.cuda.memory_reserved() > 3 * 1024**3:
            torch.cuda.empty_cache()
        return _unit(vectors)
