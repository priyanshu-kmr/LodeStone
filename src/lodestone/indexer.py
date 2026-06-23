"""The indexing worker. Each queued path is only a hint: it checks the file's real state first."""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
from PIL import Image

from . import quality
from .catalog import PRIORITY_SCAN, Catalog, FileRow
from .config import SETTINGS, Settings, is_under, norm_path
from .extract import OcrReader, chunk_text, extract, within_embed_limit
from .office import close_shared_worker
from .scanner import is_cloud_only, kind_of, path_allowed
from .vectors import IMAGE, TEXT, VectorStore

log = logging.getLogger(__name__)

MAX_LOCK_RETRIES = 5
OFFICE_EXTS = {".docx", ".pptx", ".xlsx", ".html", ".htm"}


class IndexEmbedder(Protocol):
    def encode_documents(self, texts: list[str]) -> np.ndarray: ...
    def encode_images(self, images: list[Image.Image]) -> np.ndarray: ...
    def release_index_model(self) -> None: ...
    def release_search_model_if_idle(self) -> None: ...


def hash_file(path: str) -> str:
    digest = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class _ImageJob:
    path: str
    file_id: int
    image: Image.Image
    sharpness: float | None
    ocr_score: float | None


class Indexer:
    """Owns all writes to the index, which keeps the two stores simple to keep in sync."""

    def __init__(
        self,
        catalog: Catalog,
        vectors: VectorStore,
        embedder: IndexEmbedder,
        ocr: OcrReader,
        settings: Settings = SETTINGS,
        on_progress: Callable[[], None] | None = None,
    ) -> None:
        self._catalog = catalog
        self._vectors = vectors
        self._embedder = embedder
        self._ocr = ocr
        self._settings = settings
        self._on_progress = on_progress or (lambda: None)
        self._retries: dict[str, int] = {}
        self._dirty = False
        self._since_maintain = 0
        self._last_work = 0.0
        self._model_loaded = False

    # ---- public entry points --------------------------------------------------------------

    def run_batch(self) -> int:
        """Process one batch of ready paths. Returns how many it handled."""
        paths = self._catalog.next_tasks(self._settings.image_batch_size * 2)
        if not paths:
            return 0
        jobs: list[_ImageJob] = []
        for path in paths:
            job = self._prepare(path)
            if job is not None:
                jobs.append(job)
            if len(jobs) >= self._settings.image_batch_size:
                self._flush_images(jobs)
                jobs = []
        self._flush_images(jobs)
        self._since_maintain += len(paths)
        if self._since_maintain >= 1000:
            self.maintain()
        self._on_progress()
        return len(paths)

    def maintain(self) -> None:
        if not self._dirty:
            return
        self._vectors.maintain()
        self._dirty = False
        self._since_maintain = 0

    def idle(self) -> None:
        """Called when nothing is ready: tidy the index, then free whatever has gone unused."""
        self.maintain()
        self._embedder.release_search_model_if_idle()
        if time.monotonic() - self._last_work <= self._settings.vision_idle_unload_seconds:
            return
        close_shared_worker()
        self._ocr.release()
        if self._model_loaded:
            self._embedder.release_index_model()
            self._model_loaded = False

    def remove_paths(self, paths: list[str]) -> None:
        ids = []
        for path in paths:
            row = self._catalog.get_file(path)
            if row:
                ids.append(row.id)
        self.remove_ids(ids)

    def remove_ids(self, ids: list[int]) -> None:
        if not ids:
            return
        # Vectors first: if a crash lands in between, the catalog still knows to retry.
        self._vectors.delete_files(ids)
        for file_id in ids:
            self._catalog.delete_file(file_id)
        self._dirty = True

    # ---- one path ----------------------------------------------------------------------------

    def _prepare(self, path: str) -> _ImageJob | None:
        """Work out what this path needs. Images come back for batched GPU embedding."""
        row = self._catalog.get_file(path)
        try:
            return self._handle(path, row)
        except (PermissionError, BlockingIOError) as error:
            # Usually a file another program still has open. Try again shortly.
            tries = self._retries.get(path, 0) + 1
            if tries <= MAX_LOCK_RETRIES:
                self._retries[path] = tries
                self._catalog.enqueue(path, 1, delay=self._settings.debounce_seconds * tries)
                return None
            self._fail(path, row, error)
        except Exception as error:  # One bad file must never halt a long indexing run.
            log.warning("could not index %s: %s", path, error)
            self._fail(path, row, error)
        return None

    def _fail(self, path: str, row: FileRow | None, error: Exception) -> None:
        current = self._catalog.get_file(path)
        if current is None:
            try:
                info = os.stat(path)
            except OSError:
                self._catalog.dequeue(path)
                return
            kind = kind_of(path) or "document"
            file_id = self._catalog.start_file(path, kind, info.st_size, info.st_mtime_ns, None)
        else:
            file_id = current.id
        self._catalog.fail_file(file_id, f"{type(error).__name__}: {error}")
        self._retries.pop(path, None)
        self._catalog.dequeue(path)

    def _handle(self, path: str, row: FileRow | None) -> _ImageJob | None:
        catalog = self._catalog
        try:
            info = os.stat(path)
        except (FileNotFoundError, NotADirectoryError):
            if row:
                self.remove_ids([row.id])
            catalog.dequeue(path)
            return None

        folders = catalog.folders()
        if any(not f.enabled and is_under(path, f.path) for f in folders):
            # Turned off: keep what is indexed, so turning it back on does not start over.
            catalog.dequeue(path)
            return None
        enabled = [f.path for f in folders if f.enabled]
        kind = kind_of(path)
        pending = [p.path for p in catalog.dataset_prompts("pending")]
        blocked = catalog.excludes() + pending
        if (
            os.path.isdir(path)
            or kind is None
            or not path_allowed(path, enabled, blocked)
            or is_cloud_only(getattr(info, "st_file_attributes", 0))
        ):
            if row:
                self.remove_ids([row.id])
            catalog.dequeue(path)
            return None

        if (
            row
            and row.status in ("indexed", "error")
            and row.size == info.st_size
            and row.mtime_ns == info.st_mtime_ns
            and not (row.force_full and row.index_level != "full")
        ):
            catalog.dequeue(path)
            return None

        if row and row.status == "indexed" and row.indexed_at and not row.force_full:
            remaining = row.indexed_at + self._settings.reindex_cooldown_seconds - time.time()
            if remaining > 0:
                catalog.enqueue(path, PRIORITY_SCAN, delay=remaining)
                return None

        too_large = self._too_large(kind, info.st_size, path)
        content_hash = None if too_large else hash_file(path)

        if content_hash and row and row.status == "indexed" and row.content_hash == content_hash \
                and not (row.force_full and row.index_level != "full"):
            # Only the timestamp changed.
            catalog.touch_file(row.id, info.st_size, info.st_mtime_ns)
            catalog.dequeue(path)
            return None

        if content_hash and self._try_move(path, row, content_hash, info):
            catalog.dequeue(path)
            return None

        force = bool(row and row.force_full)
        if row:
            self._vectors.delete_files([row.id])
            self._dirty = True
        file_id = catalog.start_file(path, _stored_kind(kind), info.st_size, info.st_mtime_ns, content_hash)
        if force:
            catalog.set_force_full(file_id, True)
        self._on_progress()

        if too_large or kind == "name_only":
            catalog.finish_file(file_id, "name_only", "too_large" if too_large else "unsupported")
            catalog.dequeue(path)
            return None
        if kind == "image":
            return self._prepare_image(path, file_id, force)
        self._index_document(path, file_id)
        catalog.dequeue(path)
        self._retries.pop(path, None)
        return None

    def _too_large(self, kind: str, size: int, path: str) -> bool:
        if kind == "image":
            return size > self._settings.max_image_bytes
        if os.path.splitext(path)[1].lower() in OFFICE_EXTS:
            return size > self._settings.max_office_bytes
        return size > self._settings.max_document_bytes

    def _try_move(self, path: str, row: FileRow | None, content_hash: str, info: os.stat_result) -> bool:
        """Same content, and the old path is gone: it moved, so keep the expensive vectors."""
        for other in self._catalog.files_with_hash(content_hash, info.st_size):
            if norm_path(other.path) == norm_path(path) or os.path.exists(other.path):
                continue
            if row:
                self.remove_ids([row.id])
            self._catalog.move_file(other.id, path, info.st_size, info.st_mtime_ns)
            return True
        return False

    # ---- documents ---------------------------------------------------------------------------

    def _index_document(self, path: str, file_id: int) -> None:
        settings = self._settings
        self._last_work = time.monotonic()  # Keeps the Office converter open between documents.
        extracted = extract(path, self._ocr, settings)
        body = chunk_text(extracted.body, settings)
        ocr_chunks = chunk_text(extracted.ocr, settings)
        if not body and not ocr_chunks:
            self._catalog.finish_file(file_id, "name_only", "no_text")
            return
        ids = self._catalog.add_chunks(file_id, body, "body")
        ids += self._catalog.add_chunks(file_id, ocr_chunks, "ocr", start_ordinal=len(body) + 1)
        self._embed_chunks(file_id, ids, body + ocr_chunks)
        self._catalog.finish_file(file_id, "full", ocr_score=extracted.ocr_score)

    def _embed_chunks(self, file_id: int, ids: list[int], texts: list[str]) -> None:
        """Embed the opening of a file, if it has enough text. All chunks stay in keyword search."""
        minimum = self._settings.min_embed_chars
        pairs = [
            (chunk_id, text)
            for chunk_id, text in zip(ids, within_embed_limit(texts, self._settings))
            if sum(c.isalnum() for c in text) >= minimum
        ]
        if not pairs:
            return
        self._model_loaded = True
        self._last_work = time.monotonic()
        vectors = self._embedder.encode_documents([text for _, text in pairs])
        self._vectors.add(TEXT, [chunk_id for chunk_id, _ in pairs], [file_id] * len(pairs), vectors)
        self._dirty = True

    # ---- images ------------------------------------------------------------------------------

    def _prepare_image(self, path: str, file_id: int, force: bool) -> _ImageJob | None:
        result = quality.assess(path, self._settings, force=force)
        if not result.passed:
            self._catalog.finish_file(file_id, "name_only", result.reason, sharpness=result.sharpness)
            self._catalog.dequeue(path)
            return None
        ocr = self._ocr.read(result.image)
        chunks = chunk_text(ocr.text, self._settings)
        if chunks:
            self._embed_chunks(file_id, self._catalog.add_chunks(file_id, chunks, "ocr"), chunks)
        return _ImageJob(path, file_id, result.image, result.sharpness, ocr.score)

    def _flush_images(self, jobs: list[_ImageJob]) -> None:
        if not jobs:
            return
        self._model_loaded = True
        self._last_work = time.monotonic()
        try:
            vectors = self._embedder.encode_images([j.image for j in jobs])
        except Exception as error:
            if len(jobs) > 1:
                # One image can break a whole batch, so split until it is isolated.
                middle = len(jobs) // 2
                self._flush_images(jobs[:middle])
                self._flush_images(jobs[middle:])
                return
            log.warning("could not embed %s: %s", jobs[0].path, error)
            self._catalog.fail_file(jobs[0].file_id, f"{type(error).__name__}: {error}")
            self._catalog.dequeue(jobs[0].path)
            return
        ids = [j.file_id for j in jobs]
        self._vectors.add(IMAGE, ids, ids, vectors)
        self._dirty = True
        for job in jobs:
            self._catalog.finish_file(job.file_id, "full", sharpness=job.sharpness, ocr_score=job.ocr_score)
            self._catalog.dequeue(job.path)
            self._retries.pop(job.path, None)


def _stored_kind(kind: str) -> str:
    return "document" if kind == "name_only" else kind
