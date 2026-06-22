"""The live file watcher. Events only queue a path to re-check, because they can be wrong or lost."""

from __future__ import annotations

import logging
import os
from typing import Callable

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .catalog import PRIORITY_LIVE, PRIORITY_MISSING, Catalog
from .config import SETTINGS, Settings
from .scanner import is_temp_file, kind_of, path_allowed

log = logging.getLogger(__name__)


class _Handler(FileSystemEventHandler):
    def __init__(self, watcher: "Watcher") -> None:
        self._watcher = watcher

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._watcher.file_changed(os.fsdecode(event.src_path))
        else:
            self._watcher.request_rescan()

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._watcher.file_changed(os.fsdecode(event.src_path))

    def on_deleted(self, event: FileSystemEvent) -> None:
        self._watcher.path_gone(os.fsdecode(event.src_path))

    def on_moved(self, event: FileSystemEvent) -> None:
        source = os.fsdecode(event.src_path)
        destination = os.fsdecode(event.dest_path)
        self._watcher.path_gone(source)
        if event.is_directory:
            self._watcher.request_rescan()
        else:
            self._watcher.file_changed(destination)


class Watcher:
    def __init__(
        self,
        catalog: Catalog,
        on_rescan: Callable[[], None],
        on_event: Callable[[], None],
        settings: Settings = SETTINGS,
    ) -> None:
        self._catalog = catalog
        self._on_rescan = on_rescan
        self._on_event = on_event
        self._settings = settings
        self._observer = None
        self._watches: dict[str, object] = {}
        self._handler = _Handler(self)
        self._roots: list[str] = []
        self._excludes: list[str] = []

    def start(self) -> None:
        self._observer = Observer()
        self._observer.start()

    def stop(self) -> None:
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
        self._watches.clear()

    def sync(self) -> None:
        """Match what is being watched to the current folder settings."""
        folders = [f.path for f in self._catalog.folders() if f.enabled]
        self._roots = folders
        pending = [p.path for p in self._catalog.dataset_prompts("pending")]
        self._excludes = self._catalog.excludes() + pending
        if self._observer is None:
            return
        for path in list(self._watches):
            if path not in folders:
                try:
                    self._observer.unschedule(self._watches.pop(path))
                except (KeyError, ValueError):
                    pass
        for path in folders:
            if path not in self._watches and os.path.isdir(path):
                try:
                    self._watches[path] = self._observer.schedule(self._handler, path, recursive=True)
                except OSError as error:
                    log.warning("could not watch %s: %s", path, error)

    def file_changed(self, path: str) -> None:
        name = os.path.basename(path)
        if kind_of(name) is None or is_temp_file(name):
            return
        if path_allowed(path, self._roots, self._excludes):
            # Each new event restarts the wait, so half-copied files are left alone.
            self._catalog.enqueue(path, PRIORITY_LIVE, delay=self._settings.debounce_seconds)
            self._on_event()

    def path_gone(self, path: str) -> None:
        """Wait a little before forgetting, so a move's new path gets to keep the vectors."""
        known = self._catalog.file_paths_under(path)
        if known:
            self._catalog.enqueue_many(
                known, PRIORITY_MISSING, delay=self._settings.delete_delay_seconds
            )
            self._on_event()

    def request_rescan(self) -> None:
        self._on_rescan()
