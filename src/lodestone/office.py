"""Office conversion in a separate process, so one bad file can't block all indexing.

Python threads can't be stopped, but a process can. If a file takes too long, the process is killed
and the file is marked as failed. It stays findable by name, and the queue moves on.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import threading
from typing import Callable

from .config import SETTINGS, Settings

log = logging.getLogger(__name__)


def read_spreadsheet(path: str, settings: Settings = SETTINGS) -> str:
    """Read the first rows of each sheet with a streaming reader.

    A workbook can hold a million repetitive rows. Its opening describes it well enough for search,
    and loading all of it took over 9 minutes on a real 43 MB file.
    """
    from openpyxl import load_workbook

    workbook = load_workbook(path, read_only=True, data_only=True)
    parts: list[str] = []
    total = 0
    try:
        for sheet in workbook.worksheets:
            parts.append(f"Sheet: {sheet.title}")
            for count, row in enumerate(sheet.iter_rows(values_only=True)):
                if count >= settings.sheet_max_rows or total >= settings.sheet_max_chars:
                    break
                cells = [str(cell) for cell in row if cell is not None]
                if cells:
                    line = " | ".join(cells)
                    parts.append(line)
                    total += len(line)
            if total >= settings.sheet_max_chars:
                break
    finally:
        workbook.close()
    return "\n".join(parts)


_markitdown = None


def convert_office_file(path: str) -> str:
    global _markitdown
    if os.path.splitext(path)[1].lower() == ".xlsx":
        return read_spreadsheet(path)
    if _markitdown is None:
        # Slow to set up, so make one and keep it for this process's lifetime.
        from markitdown import MarkItDown

        _markitdown = MarkItDown(enable_plugins=False)
    return _markitdown.convert_local(path).text_content or ""


def _serve(connection, handler: Callable[[str], str]) -> None:
    while True:
        try:
            path = connection.recv()
        except EOFError:  # The app has gone, so stop rather than linger in the background.
            return
        if path is None:
            return
        try:
            connection.send((True, handler(path)))
        except Exception as error:
            connection.send((False, f"{type(error).__name__}: {error}"))


class OfficeWorker:
    def __init__(
        self,
        timeout: float = SETTINGS.office_timeout_seconds,
        handler: Callable[[str], str] = convert_office_file,
    ) -> None:
        self._timeout = timeout
        self._handler = handler
        self._process = None
        self._connection = None
        self._lock = threading.Lock()

    def _start(self) -> None:
        context = multiprocessing.get_context("spawn")
        ours, theirs = context.Pipe()
        self._process = context.Process(target=_serve, args=(theirs, self._handler), daemon=True)
        self._process.start()
        theirs.close()
        self._connection = ours

    def _kill(self) -> None:
        if self._process is not None:
            self._process.kill()
            self._process.join(timeout=5)
        if self._connection is not None:
            self._connection.close()
        self._process = None
        self._connection = None

    def convert(self, path: str) -> str:
        with self._lock:
            if self._process is None or not self._process.is_alive():
                self._kill()
                self._start()
            self._connection.send(path)
            if not self._connection.poll(self._timeout):
                self._kill()
                raise TimeoutError(f"gave up after {self._timeout:.0f}s")
            try:
                succeeded, value = self._connection.recv()
            except EOFError:
                self._kill()
                raise RuntimeError("the converter crashed") from None
        if not succeeded:
            raise ValueError(value)
        return value

    def close(self) -> None:
        with self._lock:
            self._kill()


_worker: OfficeWorker | None = None
_worker_lock = threading.Lock()


def shared_worker() -> OfficeWorker:
    global _worker
    with _worker_lock:
        if _worker is None:
            _worker = OfficeWorker()
        return _worker


def close_shared_worker() -> None:
    global _worker
    with _worker_lock:
        if _worker is not None:
            _worker.close()
            _worker = None
