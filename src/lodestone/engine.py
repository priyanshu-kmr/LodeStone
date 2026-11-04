"""The background engine: one process owns indexing, watching, rescans and search."""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Callable, Iterator

from .catalog import PRIORITY_MISSING, PRIORITY_SCAN, Catalog, DatasetPrompt, WatchedFolder
from .config import (
    DEFAULT_MODEL,
    SETTINGS,
    EmbeddingModel,
    Paths,
    Settings,
    default_folders,
    get_paths,
    is_under,
    norm_path,
    seed_excludes,
)
from .extract import within_embed_limit
from .indexer import Indexer
from .office import close_shared_worker
from .scanner import FileEntry, FolderEstimate, estimate_folder, scan_root
from .search import ALL, Searcher, SearchResult
from .vectors import TEXT, VectorStore
from .watcher import Watcher

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Status:
    state: str
    queued: int
    indexed: int
    errors: int
    name_only: int
    pending_prompts: int


class Engine:
    def __init__(
        self,
        paths: Paths | None = None,
        embedder=None,
        ocr=None,
        settings: Settings = SETTINGS,
        watch: bool = True,
    ) -> None:
        self.paths = paths or get_paths()
        self.catalog = Catalog(self.paths.catalog)
        saved_limit = self.catalog.get_meta("user_token_limit")
        if saved_limit is not None:
            settings = replace(settings, embed_token_limit=int(saved_limit))
        for field in ("video_seconds", "audio_seconds"):
            saved = self.catalog.get_meta(f"user_{field}")
            if saved is not None:
                settings = replace(settings, **{field: int(saved)})
        for field in ("unload_index_model", "unload_search_model"):
            saved = self.catalog.get_meta(field)
            if saved is not None:
                settings = replace(settings, **{field: saved == "1"})
        self.settings = settings
        saved_model = self.catalog.get_meta("embedding_model")
        self.embedding_model = EmbeddingModel.from_json(saved_model) if saved_model else DEFAULT_MODEL
        self.vectors = VectorStore(
            self.paths.vectors, self.embedding_model.id, self.embedding_model.dim, settings
        )
        if embedder is None:
            from .embedder import GemmaEmbedder

            embedder = GemmaEmbedder(
                settings, self.gpu_for_search, self.gpu_for_indexing, self.embedding_model
            )
        if ocr is None:
            from .ocr import Ocr

            ocr = Ocr()
        self.embedder = embedder
        self.indexer = Indexer(
            self.catalog, self.vectors, embedder, ocr, settings, on_progress=self._notify
        )
        self.searcher = Searcher(self.catalog, self.vectors, embedder, settings)

        self._listeners: list[Callable[[], None]] = []
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._scan_wanted = threading.Event()
        self._scanning = threading.Event()
        self._paused = threading.Event()
        if self.catalog.get_meta("paused") == "1":
            self._paused.set()
        self._admin_lock = threading.Lock()
        self._batch_lock = threading.Lock()
        self._search_lock = threading.Lock()
        self._hold_worker = threading.Event()
        self._threads: list[threading.Thread] = []
        self.watcher = (
            Watcher(self.catalog, self.request_scan, self._wake.set, settings) if watch else None
        )

    # ---- lifecycle ----------------------------------------------------------------------------

    def seed_defaults(self) -> None:
        """First run only: later changes in Settings must stick."""
        if self.catalog.get_meta("seeded"):
            return
        for folder in default_folders():
            self.catalog.add_folder(str(folder), is_default=True)
        for folder in seed_excludes():
            self.catalog.add_exclude(str(folder))
        self.catalog.set_meta("seeded", "1")

    def start(self) -> None:
        self.catalog.retry_failed()
        if self.watcher:
            self.watcher.start()
            self.watcher.sync()
        for target, name in (
            (self._worker_loop, "lodestone-indexer"),
            (self._scan_loop, "lodestone-scanner"),
        ):
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self._threads.append(thread)
        # A crash or a closed app can leave work behind, so always check on startup.
        self.request_scan()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        self._scan_wanted.set()
        if self.watcher:
            self.watcher.stop()
        for thread in self._threads:
            thread.join(timeout=10)
        self._threads.clear()
        close_shared_worker()

    def add_listener(self, callback: Callable[[], None]) -> None:
        self._listeners.append(callback)

    def _notify(self) -> None:
        for callback in self._listeners:
            try:
                callback()
            except Exception:
                log.exception("status listener failed")

    # ---- background loops -----------------------------------------------------------------------

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            handled = 0
            if not self._hold_worker.is_set():
                try:
                    if self._paused.is_set():
                        # Paused still has to let go of the models, or they stay loaded until unpaused.
                        with self._admin_lock:
                            self.indexer.free_unused()
                    else:
                        with self._batch_lock:
                            handled = self.indexer.run_batch()
                        if not handled:
                            with self._admin_lock:
                                self.indexer.idle()
                except Exception:
                    log.exception("indexing batch failed")
                    self._stop.wait(1.0)
            if not handled:
                self._wake.wait(timeout=0.5)
                self._wake.clear()

    def _scan_loop(self) -> None:
        interval = self.settings.rescan_hours * 3600
        while not self._stop.is_set():
            self._scan_wanted.wait(timeout=interval)
            if self._stop.is_set():
                return
            self._scan_wanted.clear()
            self._scanning.set()
            self._notify()
            try:
                self.scan_all()
            except Exception:
                log.exception("rescan failed")
            finally:
                self._scanning.clear()
                self._wake.set()
                self._notify()

    @contextmanager
    def _worker_held(self) -> Iterator[None]:
        """Let the worker finish its batch and keep it out, so it can't write stale results after."""
        with self._admin_lock:
            self._hold_worker.set()
            try:
                with self._batch_lock:
                    yield
            finally:
                self._hold_worker.clear()
                self._wake.set()

    def request_scan(self) -> None:
        self._scan_wanted.set()

    def scan_all(self) -> None:
        """The safety net: find whatever the watcher missed, including while the app was closed."""
        self._apply_embed_limit()
        catalog = self.catalog
        folders = catalog.folders()
        enabled = [f.path for f in folders if f.enabled]
        disabled = [f.path for f in folders if not f.enabled]
        decisions = catalog.dataset_prompts()
        allowed = {norm_path(p.path).lower() for p in decisions if p.decision == "index"}

        # Keyed by lower-case path, because Windows paths ignore case.
        seen: dict[str, FileEntry] = {}
        known: dict[str, tuple[int, int, str]] = {}
        blocked: list[str] = [p.path for p in decisions if p.decision == "pending"]
        excludes = catalog.excludes() + disabled
        extra_exts = catalog.custom_exts()

        for root in enabled:
            if not os.path.isdir(root):
                # An unplugged drive must not look like every file was deleted.
                continue
            result = scan_root(root, excludes, self.settings, self._stop.is_set, extra_exts)
            if self._stop.is_set():
                return
            for candidate in result.candidates:
                if norm_path(candidate.path).lower() in allowed:
                    continue
                catalog.record_dataset_prompt(
                    candidate.path, candidate.image_count, ", ".join(candidate.signs)
                )
                blocked.append(candidate.path)
            for entry in result.files:
                seen[entry.path.lower()] = entry
            known.update(catalog.known_state_under(root))

        lookup = {path.lower(): state for path, state in known.items()}
        changed: list[str] = []
        for lower, entry in seen.items():
            if any(is_under(entry.path, folder) for folder in blocked):
                continue
            state = lookup.get(lower)
            if state is None or state[:2] != (entry.size, entry.mtime_ns) or state[2] == "pending":
                changed.append(entry.path)
        catalog.enqueue_many(changed, PRIORITY_SCAN)

        # Queued last and at the lowest priority, so moved files are matched up before anything
        # is forgotten. The worker checks each path again, so a mistake here cannot lose data.
        missing = [
            path for path in known
            if path.lower() not in seen and not any(is_under(path, d) for d in disabled)
        ]
        catalog.enqueue_many(missing, PRIORITY_MISSING)
        if self.watcher:
            self.watcher.sync()
        catalog.set_meta("last_scan", str(time.time()))

    def _apply_embed_limit(self) -> None:
        """Bring files indexed under an earlier limit in line, so ranking treats every file alike.

        A file with hundreds of vectors has many more chances to match by luck. Removing the
        extras needs no embedding, so this is quick. It runs once per limit value.
        """
        limit = str(self.settings.embed_token_limit)
        if self.catalog.get_meta("embed_token_limit") == limit:
            return
        extra: list[int] = []
        for chunks in self.catalog.text_chunks_by_file().values():
            kept = len(within_embed_limit([text for _, text in chunks], self.settings))
            extra.extend(chunk_id for chunk_id, _ in chunks[kept:])
        with self._admin_lock:
            self.vectors.delete_rows(TEXT, extra)
            self.vectors.maintain()
        self.catalog.set_meta("embed_token_limit", limit)
        log.info("removed %d vectors beyond the embedding limit", len(extra))

    # ---- search -----------------------------------------------------------------------------------

    def search(self, query: str, top_k: int | None = None, mode: str = ALL) -> list[SearchResult]:
        # Holds off a model switch, because a query vector means nothing against another model's tables.
        with self._search_lock:
            results = self.searcher.search(query, top_k, mode)
        present = []
        for result in results:
            if os.path.exists(result.path):
                present.append(result)
            else:
                # The index is out of date for this file. Hide it and fix the index.
                self.catalog.enqueue(result.path, 0)
                self._wake.set()
        return present

    # ---- folders -----------------------------------------------------------------------------------

    def folders(self) -> list[WatchedFolder]:
        return self.catalog.folders()

    def excludes(self) -> list[str]:
        return self.catalog.excludes()

    def estimate(self, path: str) -> FolderEstimate:
        return estimate_folder(path, self.catalog.excludes(), self.catalog.custom_exts())

    def add_folder(self, path: str) -> None:
        self.catalog.add_folder(path)
        self._settings_changed()

    def remove_folder(self, path: str) -> None:
        """Stop watching and forget its entries. The actual files are never touched."""
        with self._admin_lock:
            self.catalog.remove_folder(path)
            remaining = [f.path for f in self.catalog.folders()]
            self.catalog.clear_queue_under(path)
            doomed = [
                p for p in self.catalog.file_paths_under(path)
                if not any(is_under(p, r) for r in remaining)
            ]
            self.indexer.remove_paths(doomed)
        self._settings_changed()

    def set_folder_enabled(self, path: str, enabled: bool) -> None:
        self.catalog.set_folder_enabled(path, enabled)
        if not enabled:
            self.catalog.clear_queue_under(path)
        self._settings_changed()

    def add_exclude(self, path: str) -> None:
        with self._admin_lock:
            self.catalog.add_exclude(path)
            self.catalog.clear_queue_under(path)
            self.indexer.remove_paths(self.catalog.file_paths_under(path))
        self._settings_changed()

    def remove_exclude(self, path: str) -> None:
        self.catalog.remove_exclude(path)
        self._settings_changed()

    def _settings_changed(self) -> None:
        if self.watcher:
            self.watcher.sync()
        self.request_scan()
        self._wake.set()
        self._notify()

    # ---- file types, token limit and the whole index -------------------------------------------

    def custom_exts(self) -> list[str]:
        return sorted(self.catalog.custom_exts())

    def add_custom_exts(self, exts: list[str]) -> None:
        self.catalog.set_custom_exts(self.catalog.custom_exts() | set(exts))
        self._settings_changed()

    def remove_custom_ext(self, ext: str) -> None:
        """The rescan queues these files as missing, and the worker then forgets them."""
        self.catalog.set_custom_exts(self.catalog.custom_exts() - {ext})
        self._settings_changed()

    @property
    def token_limit(self) -> int:
        return self.settings.embed_token_limit

    def set_token_limit(self, limit: int) -> None:
        """Only text uses the limit, so images without any text keep their slow-to-make vectors."""
        with self._worker_held():
            self.catalog.set_meta("user_token_limit", str(limit))
            self.settings = replace(self.settings, embed_token_limit=limit)
            self.indexer.set_settings(self.settings)
            self.catalog.mark_for_reindex(text_only=True)
        self._settings_changed()

    @property
    def video_seconds(self) -> int:
        return self.settings.video_seconds

    @property
    def audio_seconds(self) -> int:
        return self.settings.audio_seconds

    def set_video_seconds(self, seconds: int) -> None:
        self._set_media_seconds("video_seconds", seconds, ["video"])

    def set_audio_seconds(self, seconds: int) -> None:
        """Videos are included, because this also decides how much of their sound gets read."""
        self._set_media_seconds("audio_seconds", seconds, ["audio", "video"])

    def _set_media_seconds(self, field: str, seconds: int, kinds: list[str]) -> None:
        with self._worker_held():
            self.catalog.set_meta(f"user_{field}", str(seconds))
            self.settings = replace(self.settings, **{field: seconds})
            self.indexer.set_settings(self.settings)
            self.catalog.mark_kinds_for_reindex(kinds)
        self._settings_changed()

    def reindex(self) -> None:
        """Read and embed every file again. Each stays searchable until its turn comes."""
        with self._worker_held():
            self.catalog.mark_for_reindex()
        self._settings_changed()

    def delete_index(self) -> None:
        """Forget every file and free the disk space. The actual files are never touched.

        Indexing stays paused afterwards, or the next scan would rebuild it all straight away.
        """
        self.set_paused(True)
        with self._worker_held():
            self.vectors.clear()
            self.catalog.clear_index()
        self._settings_changed()

    # ---- embedding model ----------------------------------------------------------------------

    def inspect_model(self, model_id: str) -> EmbeddingModel:
        """Download and try a model, without changing anything. This can take minutes."""
        return self.embedder.inspect_model(model_id)

    def set_embedding_model(self, model: EmbeddingModel) -> None:
        """Replace the vectors and queue every file to be embedded again.

        In this order so a crash at any point leaves a working index: until the model is saved,
        the app restarts on the old model, and only has to reindex.
        """
        with self._worker_held():
            self.catalog.mark_for_reindex()
            with self._search_lock:
                self.embedder.set_model(model)
                self.vectors.switch_model(model.id, model.dim)
                self.embedding_model = model
            self.catalog.set_meta("embedding_model", model.to_json())
        self._settings_changed()

    # ---- review: datasets and name-only images -----------------------------------------------

    def dataset_prompts(self) -> list[DatasetPrompt]:
        return self.catalog.dataset_prompts("pending")

    def decide_dataset(self, path: str, index_it: bool) -> None:
        self.catalog.set_dataset_decision(path, "index" if index_it else "skip")
        if not index_it:
            self.catalog.add_exclude(path)
        self._settings_changed()

    def name_only_images(self):
        return self.catalog.name_only_images()

    def index_anyway(self, path: str) -> None:
        """The escape hatch for an image the quality rules got wrong."""
        row = self.catalog.get_file(path)
        if row:
            self.catalog.set_force_full(row.id, True)
            self.catalog.enqueue(path, 0)
            self._wake.set()

    # ---- GPU ----------------------------------------------------------------------------------------

    def _gpu_choice(self, name: str) -> bool:
        """Before there were two choices there was one, so that still counts until one is made."""
        saved = self.catalog.get_meta(name)
        return (saved if saved is not None else self.catalog.get_meta("use_gpu")) == "1"

    @property
    def gpu_for_search(self) -> bool:
        return self._gpu_choice("gpu_for_search")

    @property
    def gpu_for_indexing(self) -> bool:
        return self._gpu_choice("gpu_for_indexing")

    def set_gpu_for_search(self, use_gpu: bool) -> None:
        """Remembered between runs. Falls back to the CPU if there's no usable GPU."""
        self.catalog.set_meta("gpu_for_search", "1" if use_gpu else "0")
        self.embedder.set_gpu_for_search(use_gpu)

    def set_gpu_for_indexing(self, use_gpu: bool) -> None:
        """Remembered between runs. Falls back to the CPU if there's no usable GPU."""
        self.catalog.set_meta("gpu_for_indexing", "1" if use_gpu else "0")
        self.embedder.set_gpu_for_indexing(use_gpu)

    # ---- unloading ----------------------------------------------------------------------------

    def set_unload_index_model(self, unload: bool) -> None:
        self._set_unload("unload_index_model", unload)

    def set_unload_search_model(self, unload: bool) -> None:
        self._set_unload("unload_search_model", unload)

    def _set_unload(self, field: str, unload: bool) -> None:
        with self._admin_lock:
            self.catalog.set_meta(field, "1" if unload else "0")
            self.settings = replace(self.settings, **{field: unload})
            self.indexer.set_settings(self.settings)

    @property
    def search_model_loaded(self) -> bool:
        return self.embedder.search_model_loaded

    def prepare_search(self) -> None:
        self.embedder.warm_up()

    # ---- status -----------------------------------------------------------------------------------

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    def set_paused(self, paused: bool) -> None:
        """Remembered between runs, so a deleted index isn't rebuilt by simply restarting."""
        if paused:
            self._paused.set()
        else:
            self._paused.clear()
            self._wake.set()
        self.catalog.set_meta("paused", "1" if paused else "0")
        self._notify()

    def status(self) -> Status:
        counts = self.catalog.counts()
        queued = self.catalog.active_queue_size()
        if self.paused:
            state = "Paused"
        elif self._scanning.is_set():
            state = "Checking folders"
        elif queued:
            state = "Indexing"
        else:
            state = "Up to date"
        return Status(
            state, queued, counts["indexed"], counts["errors"], counts["name_only"],
            len(self.catalog.dataset_prompts("pending")),
        )

    def wait_until_idle(self, timeout: float = 3600.0) -> bool:
        """Block until the queue drains. Used by the command line and the tests."""
        deadline = time.monotonic() + timeout
        settled = 0
        while time.monotonic() < deadline:
            quiet = (
                not self._scan_wanted.is_set()
                and not self._scanning.is_set()
                and self.catalog.queue_size() == 0
            )
            settled = settled + 1 if quiet else 0
            if settled >= 3:
                with self._admin_lock:
                    self.indexer.maintain()
                return True
            time.sleep(0.2)
        return False
