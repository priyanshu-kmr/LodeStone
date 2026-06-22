"""The SQLite catalog: the source of truth for files, text, keyword search and settings."""

from __future__ import annotations

import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from .config import is_under, norm_path

SCHEMA_VERSION = "1"

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS watched_folders (
    path       TEXT PRIMARY KEY COLLATE NOCASE,
    enabled    INTEGER NOT NULL DEFAULT 1,
    is_default INTEGER NOT NULL DEFAULT 0,
    added_at   REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS excluded_paths (
    path     TEXT PRIMARY KEY COLLATE NOCASE,
    added_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS dataset_prompts (
    path        TEXT PRIMARY KEY COLLATE NOCASE,
    image_count INTEGER NOT NULL,
    signs       TEXT NOT NULL,
    decision    TEXT NOT NULL DEFAULT 'pending'
);
CREATE TABLE IF NOT EXISTS files (
    id           INTEGER PRIMARY KEY,
    path         TEXT NOT NULL UNIQUE COLLATE NOCASE,
    name         TEXT NOT NULL,
    kind         TEXT NOT NULL,
    size         INTEGER NOT NULL,
    mtime_ns     INTEGER NOT NULL,
    content_hash TEXT,
    status       TEXT NOT NULL DEFAULT 'pending',
    index_level  TEXT NOT NULL DEFAULT 'name_only',
    skip_reason  TEXT,
    sharpness    REAL,
    ocr_score    REAL,
    force_full   INTEGER NOT NULL DEFAULT 0,
    error        TEXT,
    indexed_at   REAL
);
CREATE INDEX IF NOT EXISTS files_hash ON files(content_hash, size);
CREATE INDEX IF NOT EXISTS files_status ON files(status);
CREATE TABLE IF NOT EXISTS chunks (
    id      INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    source  TEXT NOT NULL,
    text    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_file ON chunks(file_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, content='chunks', content_rowid='id', tokenize='unicode61 remove_diacritics 2'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
CREATE TABLE IF NOT EXISTS queue (
    path        TEXT PRIMARY KEY COLLATE NOCASE,
    priority    INTEGER NOT NULL,
    not_before  REAL NOT NULL,
    enqueued_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS queue_order ON queue(priority, not_before);
"""

PRIORITY_LIVE = 0
PRIORITY_SCAN = 1
PRIORITY_MISSING = 2

_FTS_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
_STOPWORDS = {
    "a", "an", "and", "are", "at", "for", "from", "has", "have", "in", "is", "it", "my", "of",
    "on", "or", "that", "the", "this", "to", "was", "which", "with",
}


@dataclass(frozen=True)
class FileRow:
    id: int
    path: str
    name: str
    kind: str
    size: int
    mtime_ns: int
    content_hash: str | None
    status: str
    index_level: str
    skip_reason: str | None
    sharpness: float | None
    ocr_score: float | None
    force_full: int
    error: str | None
    indexed_at: float | None


@dataclass(frozen=True)
class WatchedFolder:
    path: str
    enabled: bool
    is_default: bool


@dataclass(frozen=True)
class DatasetPrompt:
    path: str
    image_count: int
    signs: str
    decision: str


def name_chunk_text(path: str) -> str:
    """Filename and parent folder as plain words, so interview_notes.pdf matches 'interview notes'."""
    p = Path(path)
    words = re.sub(r"[_\-.]+", " ", p.stem)
    # Split camelCase too, so InterviewNotes also matches.
    words = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", words)
    return f"{words} {p.suffix.lstrip('.')} {p.parent.name}".strip()


def query_terms(text: str) -> list[str]:
    """The words of a query that are worth matching on."""
    tokens = _FTS_TOKEN.findall(text.lower())
    return list(dict.fromkeys([t for t in tokens if t not in _STOPWORDS] or tokens))


def term_coverage(terms: list[str], text: str) -> float:
    """The share of the query's words that appear in this text."""
    if not terms:
        return 0.0
    words = set(_FTS_TOKEN.findall(text.lower()))
    return sum(t in words for t in terms) / len(terms)


def fts_query(text: str) -> str:
    """Turn free text into a safe FTS5 query: any word may match, BM25 ranks the rest."""
    return " OR ".join(f'"{t}"' for t in query_terms(text))


class Catalog:
    """One connection per thread, because the watcher, worker and UI each run their own."""

    def __init__(self, path: Path | str) -> None:
        self._path = str(path)
        self._local = threading.local()
        self._write_lock = threading.RLock()
        self._conn.executescript(SCHEMA)
        self._conn.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)", (SCHEMA_VERSION,)
        )
        self._conn.commit()

    @property
    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._path, timeout=30)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    # ---- meta -----------------------------------------------------------------------------

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def set_meta(self, key: str, value: str) -> None:
        with self._write_lock, self._conn:
            self._conn.execute(
                "INSERT INTO meta(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    # ---- folders --------------------------------------------------------------------------

    def add_folder(self, path: str, is_default: bool = False) -> None:
        with self._write_lock, self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO watched_folders(path, enabled, is_default, added_at) "
                "VALUES (?, 1, ?, ?)",
                (norm_path(path), int(is_default), time.time()),
            )

    def remove_folder(self, path: str) -> None:
        with self._write_lock, self._conn:
            self._conn.execute("DELETE FROM watched_folders WHERE path=?", (norm_path(path),))

    def set_folder_enabled(self, path: str, enabled: bool) -> None:
        with self._write_lock, self._conn:
            self._conn.execute(
                "UPDATE watched_folders SET enabled=? WHERE path=?", (int(enabled), norm_path(path))
            )

    def folders(self) -> list[WatchedFolder]:
        rows = self._conn.execute(
            "SELECT path, enabled, is_default FROM watched_folders ORDER BY added_at, path"
        ).fetchall()
        return [WatchedFolder(r["path"], bool(r["enabled"]), bool(r["is_default"])) for r in rows]

    def add_exclude(self, path: str) -> None:
        with self._write_lock, self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO excluded_paths(path, added_at) VALUES (?, ?)",
                (norm_path(path), time.time()),
            )

    def remove_exclude(self, path: str) -> None:
        with self._write_lock, self._conn:
            self._conn.execute("DELETE FROM excluded_paths WHERE path=?", (norm_path(path),))

    def excludes(self) -> list[str]:
        rows = self._conn.execute("SELECT path FROM excluded_paths ORDER BY path").fetchall()
        return [r["path"] for r in rows]

    # ---- dataset prompts ------------------------------------------------------------------

    def record_dataset_prompt(self, path: str, image_count: int, signs: str) -> None:
        """Keep any earlier answer, so a folder is only ever asked about once."""
        with self._write_lock, self._conn:
            self._conn.execute(
                "INSERT INTO dataset_prompts(path, image_count, signs) VALUES (?, ?, ?) "
                "ON CONFLICT(path) DO UPDATE SET image_count=excluded.image_count, "
                "signs=excluded.signs",
                (norm_path(path), image_count, signs),
            )

    def set_dataset_decision(self, path: str, decision: str) -> None:
        with self._write_lock, self._conn:
            self._conn.execute(
                "UPDATE dataset_prompts SET decision=? WHERE path=?", (decision, norm_path(path))
            )

    def dataset_prompts(self, decision: str | None = None) -> list[DatasetPrompt]:
        sql = "SELECT path, image_count, signs, decision FROM dataset_prompts"
        args: tuple = ()
        if decision:
            sql += " WHERE decision=?"
            args = (decision,)
        rows = self._conn.execute(sql + " ORDER BY image_count DESC", args).fetchall()
        return [DatasetPrompt(r["path"], r["image_count"], r["signs"], r["decision"]) for r in rows]

    # ---- queue ----------------------------------------------------------------------------

    def enqueue(self, path: str, priority: int, delay: float = 0.0) -> None:
        self.enqueue_many([path], priority, delay)

    def enqueue_many(self, paths: Iterable[str], priority: int, delay: float = 0.0) -> None:
        """Re-queuing keeps the most urgent priority and restarts the debounce wait."""
        now = time.time()
        rows = [(norm_path(p), priority, now + delay, now) for p in paths]
        if not rows:
            return
        with self._write_lock, self._conn:
            self._conn.executemany(
                "INSERT INTO queue(path, priority, not_before, enqueued_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(path) DO UPDATE SET priority=MIN(priority, excluded.priority), "
                "not_before=excluded.not_before",
                rows,
            )

    def next_tasks(self, limit: int) -> list[str]:
        rows = self._conn.execute(
            "SELECT path FROM queue WHERE not_before <= ? ORDER BY priority, not_before LIMIT ?",
            (time.time(), limit),
        ).fetchall()
        return [r["path"] for r in rows]

    def dequeue(self, path: str) -> None:
        with self._write_lock, self._conn:
            self._conn.execute("DELETE FROM queue WHERE path=?", (norm_path(path),))

    def queue_size(self) -> int:
        return self._conn.execute("SELECT COUNT(*) AS n FROM queue").fetchone()["n"]

    def clear_queue_under(self, folder: str) -> None:
        pattern = _like_prefix(folder)
        with self._write_lock, self._conn:
            self._conn.execute(
                "DELETE FROM queue WHERE path = ? OR path LIKE ? ESCAPE '\\'",
                (norm_path(folder), pattern),
            )

    # ---- files ----------------------------------------------------------------------------

    def get_file(self, path: str) -> FileRow | None:
        row = self._conn.execute("SELECT * FROM files WHERE path=?", (norm_path(path),)).fetchone()
        return FileRow(**dict(row)) if row else None

    def get_file_by_id(self, file_id: int) -> FileRow | None:
        row = self._conn.execute("SELECT * FROM files WHERE id=?", (file_id,)).fetchone()
        return FileRow(**dict(row)) if row else None

    def get_files_by_ids(self, ids: Iterable[int]) -> dict[int, FileRow]:
        ids = list(ids)
        out: dict[int, FileRow] = {}
        for start in range(0, len(ids), 500):
            part = ids[start : start + 500]
            marks = ",".join("?" * len(part))
            for row in self._conn.execute(f"SELECT * FROM files WHERE id IN ({marks})", part):
                out[row["id"]] = FileRow(**dict(row))
        return out

    def files_with_hash(self, content_hash: str, size: int) -> list[FileRow]:
        rows = self._conn.execute(
            "SELECT * FROM files WHERE content_hash=? AND size=? AND status='indexed'",
            (content_hash, size),
        ).fetchall()
        return [FileRow(**dict(r)) for r in rows]

    def known_state_under(self, folder: str) -> dict[str, tuple[int, int, str]]:
        """Path -> (size, mtime_ns, status) for the rescan's fast comparison."""
        rows = self._conn.execute(
            "SELECT path, size, mtime_ns, status FROM files WHERE path LIKE ? ESCAPE '\\'",
            (_like_prefix(folder),),
        )
        return {r["path"]: (r["size"], r["mtime_ns"], r["status"]) for r in rows}

    def file_ids_under(self, folder: str) -> list[int]:
        rows = self._conn.execute(
            "SELECT id FROM files WHERE path = ? OR path LIKE ? ESCAPE '\\'",
            (norm_path(folder), _like_prefix(folder)),
        ).fetchall()
        return [r["id"] for r in rows]

    def file_paths_under(self, folder: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT path FROM files WHERE path = ? OR path LIKE ? ESCAPE '\\'",
            (norm_path(folder), _like_prefix(folder)),
        ).fetchall()
        return [r["path"] for r in rows]

    def start_file(self, path: str, kind: str, size: int, mtime_ns: int, content_hash: str | None) -> int:
        """Record the file and its name right away, so it is findable by name within seconds.

        Any earlier text is removed in the same transaction, and the status stays 'pending'
        until its vectors are written. A crash at any point just means it gets redone.
        """
        path = norm_path(path)
        name = os.path.basename(path)
        with self._write_lock, self._conn:
            row = self._conn.execute("SELECT id FROM files WHERE path=?", (path,)).fetchone()
            if row:
                file_id = row["id"]
                self._conn.execute(
                    "UPDATE files SET path=?, name=?, kind=?, size=?, mtime_ns=?, content_hash=?, "
                    "status='pending', index_level='name_only', skip_reason=NULL, sharpness=NULL, "
                    "ocr_score=NULL, error=NULL, indexed_at=NULL WHERE id=?",
                    (path, name, kind, size, mtime_ns, content_hash, file_id),
                )
                self._conn.execute("DELETE FROM chunks WHERE file_id=?", (file_id,))
            else:
                file_id = self._conn.execute(
                    "INSERT INTO files(path, name, kind, size, mtime_ns, content_hash) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (path, name, kind, size, mtime_ns, content_hash),
                ).lastrowid
            self._conn.execute(
                "INSERT INTO chunks(file_id, ordinal, source, text) VALUES (?, 0, 'name', ?)",
                (file_id, name_chunk_text(path)),
            )
        return int(file_id)

    def add_chunks(self, file_id: int, texts: list[str], source: str, start_ordinal: int = 1) -> list[int]:
        ids: list[int] = []
        with self._write_lock, self._conn:
            for i, text in enumerate(texts):
                cur = self._conn.execute(
                    "INSERT INTO chunks(file_id, ordinal, source, text) VALUES (?, ?, ?, ?)",
                    (file_id, start_ordinal + i, source, text),
                )
                ids.append(int(cur.lastrowid))
        return ids

    def finish_file(
        self,
        file_id: int,
        index_level: str,
        skip_reason: str | None = None,
        sharpness: float | None = None,
        ocr_score: float | None = None,
    ) -> None:
        with self._write_lock, self._conn:
            self._conn.execute(
                "UPDATE files SET status='indexed', index_level=?, skip_reason=?, sharpness=?, "
                "ocr_score=?, error=NULL, indexed_at=? WHERE id=?",
                (index_level, skip_reason, sharpness, ocr_score, time.time(), file_id),
            )

    def fail_file(self, file_id: int, error: str) -> None:
        with self._write_lock, self._conn:
            self._conn.execute(
                "UPDATE files SET status='error', error=?, indexed_at=? WHERE id=?",
                (error[:500], time.time(), file_id),
            )

    def retry_failed(self) -> int:
        """Give failed files another try. Some causes are temporary, like a locked file."""
        with self._write_lock, self._conn:
            paths = [r["path"] for r in self._conn.execute("SELECT path FROM files WHERE status='error'")]
            self._conn.execute("UPDATE files SET status='pending' WHERE status='error'")
        self.enqueue_many(paths, PRIORITY_SCAN)
        return len(paths)

    def touch_file(self, file_id: int, size: int, mtime_ns: int) -> None:
        with self._write_lock, self._conn:
            self._conn.execute(
                "UPDATE files SET size=?, mtime_ns=? WHERE id=?", (size, mtime_ns, file_id)
            )

    def move_file(self, file_id: int, new_path: str, size: int, mtime_ns: int) -> None:
        """Same content at a new path: keep the id, so the vectors stay valid."""
        new_path = norm_path(new_path)
        with self._write_lock, self._conn:
            self._conn.execute(
                "UPDATE files SET path=?, name=?, size=?, mtime_ns=? WHERE id=?",
                (new_path, os.path.basename(new_path), size, mtime_ns, file_id),
            )
            self._conn.execute(
                "DELETE FROM chunks WHERE file_id=? AND source='name'", (file_id,)
            )
            self._conn.execute(
                "INSERT INTO chunks(file_id, ordinal, source, text) VALUES (?, 0, 'name', ?)",
                (file_id, name_chunk_text(new_path)),
            )

    def delete_file(self, file_id: int) -> None:
        with self._write_lock, self._conn:
            self._conn.execute("DELETE FROM files WHERE id=?", (file_id,))

    def set_force_full(self, file_id: int, value: bool) -> None:
        with self._write_lock, self._conn:
            self._conn.execute(
                "UPDATE files SET force_full=? WHERE id=?", (int(value), file_id)
            )

    def name_only_images(self, limit: int = 500) -> list[FileRow]:
        rows = self._conn.execute(
            "SELECT * FROM files WHERE kind='image' AND index_level='name_only' "
            "AND status='indexed' ORDER BY skip_reason, name LIMIT ?",
            (limit,),
        ).fetchall()
        return [FileRow(**dict(r)) for r in rows]

    def counts(self) -> dict[str, int]:
        row = self._conn.execute(
            "SELECT COUNT(*) AS total, "
            "SUM(status='indexed') AS indexed, "
            "SUM(status='pending') AS pending, "
            "SUM(status='error') AS errors, "
            "SUM(kind='image' AND status='indexed' AND index_level='name_only') AS name_only "
            "FROM files"
        ).fetchone()
        return {k: int(row[k] or 0) for k in row.keys()}

    def folder_counts(self, folder: str) -> tuple[int, int]:
        """Indexed files under a folder, and how many are still queued."""
        prefix = _like_prefix(folder)
        indexed = self._conn.execute(
            "SELECT COUNT(*) AS n FROM files WHERE path LIKE ? ESCAPE '\\' AND status='indexed'",
            (prefix,),
        ).fetchone()["n"]
        queued = self._conn.execute(
            "SELECT COUNT(*) AS n FROM queue WHERE path LIKE ? ESCAPE '\\'", (prefix,)
        ).fetchone()["n"]
        return int(indexed), int(queued)

    # ---- text retrieval -------------------------------------------------------------------

    def keyword_search(self, query: str, limit: int) -> list[tuple[int, int]]:
        """BM25 search. Returns (chunk_id, file_id), best first."""
        match = fts_query(query)
        if not match:
            return []
        rows = self._conn.execute(
            "SELECT chunks.id AS chunk_id, chunks.file_id AS file_id FROM chunks_fts "
            "JOIN chunks ON chunks.id = chunks_fts.rowid "
            "WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
            (match, limit),
        ).fetchall()
        return [(r["chunk_id"], r["file_id"]) for r in rows]

    def chunk_texts(self, chunk_ids: Iterable[int]) -> dict[int, tuple[str, str]]:
        """Chunk id -> (source, text)."""
        ids = list(chunk_ids)
        out: dict[int, tuple[str, str]] = {}
        for start in range(0, len(ids), 500):
            part = ids[start : start + 500]
            marks = ",".join("?" * len(part))
            for row in self._conn.execute(
                f"SELECT id, source, text FROM chunks WHERE id IN ({marks})", part
            ):
                out[row["id"]] = (row["source"], row["text"])
        return out

    def chunks_for_file(self, file_id: int) -> list[tuple[int, str, str]]:
        rows = self._conn.execute(
            "SELECT id, source, text FROM chunks WHERE file_id=? ORDER BY ordinal", (file_id,)
        ).fetchall()
        return [(r["id"], r["source"], r["text"]) for r in rows]

    def text_chunks_by_file(self) -> dict[int, list[tuple[int, str]]]:
        """Every file's embeddable chunks in order, as (chunk id, text)."""
        grouped: dict[int, list[tuple[int, str]]] = {}
        for row in self._conn.execute(
            "SELECT id, file_id, text FROM chunks WHERE source != 'name' ORDER BY file_id, ordinal"
        ):
            grouped.setdefault(row["file_id"], []).append((row["id"], row["text"]))
        return grouped

    def all_paths(self) -> Iterator[str]:
        for row in self._conn.execute("SELECT path FROM files"):
            yield row["path"]

    def is_excluded(self, path: str) -> bool:
        return any(is_under(path, e) for e in self.excludes())


def _like_prefix(folder: str) -> str:
    base = norm_path(folder).rstrip("\\/") + os.sep
    escaped = base.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return escaped + "%"
